#!/usr/bin/env python3
"""从「普通话-粤语」句对里挖短语对，做成短语表（统计对齐，不训练模型）。

思路：同一句对里反复一起出现的 zh 片段和 yue 片段，很可能是同一个意思。
用互选（zh→yue 与 yue→zh 都是最高频）过滤噪声。

用法：python3 tools/build_yue_phrase.py [pairs.json] [--min 5]
输出：data/yue_phrase.marshal（zlib 压缩的 marshal）
"""
from __future__ import annotations

import json
import marshal
import os
import sys
import zlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "yue_phrase.marshal")
ZH_RANGE = (2, 5)
YUE_RANGE = (2, 6)
CANTONESE = set("佢哋嘅咗唔係喺冇乜嘢嗰嚟睇咁啲畀攞瞓啱攰嬲諗搵掟嘥噉咩喇囉啩啫")


NEG_ZH = set("不没无別别")
NEG_YUE = set("唔冇咪")


def load_known(path: str) -> set[str]:
    """已知的粤语词：人工词表的取值 + rime 词表（用来兜住统计噪声）。"""
    known: set[str] = set()
    tsv = os.path.join(ROOT, "data", "yue.tsv")
    if os.path.exists(tsv):
        for line in open(tsv, encoding="utf-8"):
            if "\t" in line and not line.startswith("#"):
                known.add(line.rstrip("\n").split("\t")[1])
    words = os.path.join(ROOT, "vendor", "yue_words.yaml")
    if os.path.exists(words):
        for line in open(words, encoding="utf-8"):
            if line.startswith("#") or "\t" not in line:
                continue
            piece = line.split("\t")[0]
            if len(piece) >= 2:
                known.add(piece)
    return known


def plausible(zh: str, yue: str) -> bool:
    """过滤明显错的对齐：否定词不一致、长度差太大。"""
    if abs(len(yue) - len(zh)) > 3:
        return False
    if (NEG_YUE & set(yue)) and not (NEG_ZH & set(zh)):
        return False
    if (NEG_ZH & set(zh)) and not (NEG_YUE & set(yue)):
        # 允许「不」在粤语里换成别的说法，但要求长度接近且不是纯替换
        return False
    return True


def ngrams(text: str, low: int, high: int) -> set[str]:
    """返回该文本所有 n-gram。注意：调用方必须用 sorted() 遍历，
    否则 set 的迭代顺序受 PYTHONHASHSEED 影响，Counter 的并列项会随运行变化，
    统计出来的短语表每次跑都可能差几条（实测 3972~3978 条）。"""
    out: set[str] = set()
    for n in range(low, high + 1):
        for i in range(len(text) - n + 1):
            piece = text[i:i + n]
            if all("\u4e00" <= ch <= "\u9fff" for ch in piece):
                out.add(piece)
    return out


def mine_phrases(pairs: list[dict], min_count: int = 5) -> dict[str, str]:
    """从平行句对里挖 (普通话片段 → 粤语片段)。纯统计，可被评测脚本复用。"""
    zh_freq: Counter[str] = Counter()
    yue_freq: Counter[str] = Counter()
    for item in pairs:
        zh, yue = item.get("zh", ""), item.get("yue", "")
        zh_freq.update(sorted(ngrams(zh, *ZH_RANGE)))
        yue_freq.update(sorted(ngrams(yue, *YUE_RANGE)))

    known = load_known("")
    # 只有"确实是个普通话词/短语"的 key 才收（防止 为|他 这种跨界切分）
    jieba: set[str] = set()
    jf = os.path.join(ROOT, "vendor", "jieba.txt")
    if os.path.exists(jf):
        for line in open(jf, encoding="utf-8"):
            parts = line.split()
            if parts:
                jieba.add(parts[0])
    manual: set[str] = set()
    tsv = os.path.join(ROOT, "data", "yue.tsv")
    if os.path.exists(tsv):
        for line in open(tsv, encoding="utf-8"):
            if "\t" in line and not line.startswith("#"):
                manual.add(line.rstrip("\n").split("\t")[0])
    zh_ok = {g for g, c in zh_freq.items() if c >= min_count}
    yue_ok = {g for g, c in yue_freq.items() if c >= min_count}
    print(f"句对 {len(pairs)}；候选 zh 片段 {len(zh_ok)}，yue 片段 {len(yue_ok)}")

    co: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in pairs:
        zh_z = sorted(ngrams(item.get("zh", ""), *ZH_RANGE) & zh_ok)
        yue_z = sorted(ngrams(item.get("yue", ""), *YUE_RANGE) & yue_ok)
        if not zh_z or not yue_z:
            continue
        for z in zh_z:
            co[z].update(yue_z)

    best: dict[str, str] = {}
    for z, counter in co.items():
        y, count = counter.most_common(1)[0]
        if count < min_count:
            continue
        ratio = count / zh_freq[z]
        if ratio < 0.6 or count < 8:
            continue
        if y == z or (y in z and len(y) <= len(z)):
            continue
        if z in manual:          # 人工词表优先，统计结果不许推翻
            continue
        if jieba and z not in jieba:
            continue
        if len(y) < len(z) - 1:  # 目标不能明显缩水（避免 哪里→邊）
            continue
        if not plausible(z, y):
            continue
        # 结果必须是"已知的粤语说法"，否则就是统计噪声
        if (not (CANTONESE & set(y))) and y not in known:
            continue
        # 粤语不用「們」（复数用「哋」），也不用「邊裏」这类少见写法
        if "們" in y or "裏" in y:
            continue
        best[z] = y

    # 反向检查（互选），去掉"一对多"的噪声
    reverse: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in pairs:
        zh_z = sorted(ngrams(item.get("zh", ""), *ZH_RANGE) & zh_ok)
        yue_z = sorted(ngrams(item.get("yue", ""), *YUE_RANGE) & yue_ok)
        for y in yue_z:
            reverse[y].update(zh_z)
    filtered = {}
    for z, y in best.items():
        if not reverse[y]:
            continue
        if max(reverse[y].items(), key=lambda kv: kv[1])[0] == z:
            filtered[z] = y
    return filtered


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    min_count = 5
    for arg in sys.argv[1:]:
        if arg.startswith("--min"):
            min_count = int(arg.split("=", 1)[1]) if "=" in arg else min_count
    path = args[0] if args else "/tmp/yue_pairs.json"
    pairs = json.load(open(path, encoding="utf-8"))
    filtered = mine_phrases(pairs, min_count)
    print(f"短语对 {len(filtered)} 条")
    for probe in ("给你", "我们", "知道", "这个", "没有问题"):
        if probe in filtered:
            print(f"  {probe} → {filtered[probe]}")

    with open(OUT, "wb") as fh:
        fh.write(zlib.compress(marshal.dumps(filtered), 6))
    print(f"写出 {OUT}（{os.path.getsize(OUT)/1024:.0f} KiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
