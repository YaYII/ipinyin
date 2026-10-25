#!/usr/bin/env python3
"""构建粤语神经解码的「允许替换集」（P1 约束解码的守卫数据）。

神经模型会把成语/古诗改坏（牢→監、疑是→疑係），根因是它没有一个
「该不该改」的判据。守卫的思路：只允许有**实证**的替换 ——
替换要么在人工词表/短语表里，要么在平行语料的对齐里反复出现过；
语料里从没见过的改法（牢→監）直接否掉，回退规则版。

产出 data/yue_guard.marshal：
  char_pairs  单字替换 (简→繁 + 口语字) 的出现次数（同长句对逐位对齐统计）
  left/right  带邻接语境的单字替换：left[(前字,原字)][改字] = 次数
  word_pairs  片段级替换 (2~5 字 → 2~6 字)，互选+过滤后的最高频目标
  singles     人工词表里的单字映射（yue.tsv，係/嘅/咗/喺/冇/佢…）

两种构建口径：
  --all        全量语料（出货用，默认输出 data/yue_guard.marshal）
  --split eval 训练/测试切分（seed=0 洗牌，跳过评测用的前 1/20）
               —— 评测脚本用这份，保证「守卫没见过考题」
"""
from __future__ import annotations

import argparse
import json
import marshal
import os
import random
import sys
import zlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from build_yue_phrase import CANTONESE, NEG_YUE, NEG_ZH, ngrams  # noqa: E402

ZH_RANGE = (2, 5)
YUE_RANGE = (2, 6)
# 长度收缩型替换（似的→噉、什么→咩）目标只有 1 个字，单独用更小的共现门槛挖
YUE_SINGLE_RANGE = (1, 1)


def is_cjk(text: str) -> bool:
    return all("\u4e00" <= ch <= "\u9fff" for ch in text)


def load_classical() -> set[str]:
    """典籍白名单：成语/古诗——经典文本的粤语渲染就是繁体外壳。

    口语语料统计在文言语域不适用（不觉→唔覺 在口语里是真转换），
    这类输入由守卫按白名单直接给繁体外壳。列表随仓库维护，可扩展。
    """
    lines: set[str] = set()
    for name in ("idioms.txt", "poems.txt"):
        p = os.path.join(ROOT, "tools", name)
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                s = ln.strip()
                if s and not s.startswith("#"):
                    lines.add(s)
    return lines


def load_trad_shell() -> dict[str, str]:
    """s2t 表的单字映射：纯简→繁外壳，守卫免实证放行（写→寫 这类）。"""
    import zlib
    path = os.path.join(ROOT, "data", "s2t.marshal")
    raw = open(path, "rb").read()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    table = marshal.loads(raw)
    return {s: t for s, t in table.items()
            if len(s) == 1 and len(t) == 1 and s != t
            and is_cjk(s) and is_cjk(t)}


def load_yue_tsv() -> tuple[dict[str, str], dict[str, str]]:
    """返回（全部映射，单字映射）。"""
    mapping: dict[str, str] = {}
    path = os.path.join(ROOT, "data", "yue.tsv")
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        src, dst = line.rstrip("\n").split("\t", 1)
        if src and dst:
            mapping[src] = dst
    singles = {k: v for k, v in mapping.items() if len(k) == 1}
    return mapping, singles


def known_yue_words() -> set[str]:
    """已知的粤语写法：yue.tsv 取值 + 短语表取值（过滤统计噪声用）。"""
    known: set[str] = set()
    for _, dst in load_yue_tsv()[0].items():
        known.add(dst)
    phrase_path = os.path.join(ROOT, "data", "yue_phrase.marshal")
    if os.path.exists(phrase_path):
        raw = open(phrase_path, "rb").read()
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        known.update(marshal.loads(raw).values())
    return known


def split_pairs(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """复刻 tools/eval_ime_lengths.load_pairs 的切分，train 与评测测试集互斥。"""
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
    pairs = pairs[:150000]
    n_test = max(500, len(pairs) // 20)
    return pairs[n_test:], pairs[:n_test]


def mine_word_pairs(train: list[dict], known: set[str], manual_keys: set[str],
                    min_count: int) -> dict[str, list[str]]:
    """片段级替换：与 build_yue_phrase 同源的统计对齐，但保留守卫需要的更宽口径。

    与短语表的两点差别：
    - 每个来源保留**所有**达标的写法（什么→乜嘢 / 什么→咩 都是合法粤语，
      只留 top1 会把模型的合法输出误杀成无实证）；
    - 额外挖「目标收缩成单字」的替换（似的→噉），目标字限于粤语口语字集。

    不要 jieba 过滤（vendor 里没有）、不要互选（守卫宁可宽一点，由语境检查兜底）；
    保留否定词一致、长度差、已知粤语词这些降噪。
    """
    zh_freq: Counter[str] = Counter()
    yue_freq: Counter[str] = Counter()
    for item in train:
        zh_freq.update(sorted(ngrams(item.get("zh", ""), *ZH_RANGE)))
        yue_freq.update(sorted(ngrams(item.get("yue", ""), *YUE_RANGE)))
    zh_ok = {g for g, c in zh_freq.items() if c >= min_count}
    yue_ok = {g for g, c in yue_freq.items() if c >= min_count}
    # 单字目标只收口语字，避免 任何字→的 这种垃圾对齐
    yue_single_ok = {g for g, c in yue_freq.items()
                     if c >= min_count and len(g) == 1 and g in CANTONESE}

    co: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in train:
        zh_z = sorted(ngrams(item.get("zh", ""), *ZH_RANGE) & zh_ok)
        yue_z = sorted(ngrams(item.get("yue", ""), *YUE_RANGE) & yue_ok)
        yue_s = sorted(ngrams(item.get("yue", ""), *YUE_SINGLE_RANGE) & yue_single_ok)
        if not zh_z or not (yue_z or yue_s):
            continue
        for z in zh_z:
            co[z].update(yue_z)
            co[z].update(yue_s)

    out: defaultdict[str, list[str]] = defaultdict(list)
    for z, counter in co.items():
        total = zh_freq[z]
        for y, count in counter.most_common():
            if count < min_count:
                break                          # most_common 递减，后面只会更小
            if count / total < 0.25:
                continue
            if y == z or (y in z and len(y) <= len(z)):
                continue
            if z in manual_keys:               # 人工词表已覆盖，不重复存
                continue
            if abs(len(y) - len(z)) > 3 or len(y) < len(z) - 1:
                continue
            if (NEG_YUE & set(y)) and not (NEG_ZH & set(z)):
                continue
            if (NEG_ZH & set(z)) and not (NEG_YUE & set(y)):
                continue
            # 结果必须是已知的粤语说法（含粤语口语字或词表里出现过）
            if not (CANTONESE & set(y)) and y not in known:
                continue
            if "們" in y or "裏" in y:
                continue
            if y not in out[z]:
                out[z].append(y)
    return dict(out)


def mine_vocab(train: list[dict], min_freq: int) -> set[str]:
    """目标侧（粤语书面）的已知写法：2~6 字 n-gram，句频达标。

    长句模式用它做「输出是不是已知粤语」的校验 —— 比替换对实证宽得多：
    难道→唔通、兄弟→巴打 这类合法转换未必凑得出替换对，但 唔通/巴打
    本身就是语料里常见的粤语词。
    """
    freq: Counter[str] = Counter()
    for item in train:
        yue = item.get("yue", "")
        freq.update(sorted(ngrams(yue, 2, 6)))
    return {g for g, c in freq.items() if c >= min_freq}


def mine_char_pairs(train: list[dict]) -> tuple[Counter, Counter, Counter]:
    """单字替换：同长的纯汉字句对逐位对齐，统计 (原字→改字) 与带语境的版本。"""
    char_pairs: Counter = Counter()
    left: Counter = Counter()
    right: Counter = Counter()
    for item in train:
        zh, yue = item.get("zh", ""), item.get("yue", "")
        if len(zh) != len(yue) or not is_cjk(zh) or not is_cjk(yue):
            continue
        for i, (s, d) in enumerate(zip(zh, yue)):
            if s == d:
                continue
            char_pairs[(s, d)] += 1
            if i > 0:
                left[(zh[i - 1], s, d)] += 1
            if i + 1 < len(zh):
                right[(s, d, zh[i + 1])] += 1
    return char_pairs, left, right


def prune_counter(counter: Counter, min_count: int) -> dict:
    return {k: v for k, v in counter.items() if v >= min_count}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--all", action="store_true", help="全量语料（出货口径），默认 eval 切分")
    ap.add_argument("--min-pair", type=int, default=5, help="片段替换的最低共现次数")
    ap.add_argument("--min-char", type=int, default=10, help="单字替换的最低次数")
    ap.add_argument("--min-ctx", type=int, default=3, help="语境单字替换的最低次数")
    ap.add_argument("--min-vocab", type=int, default=10, help="已知粤语写法的最低句频")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    if args.all:
        train = [p for p in pairs if p["zh"] and p["yue"]
                 and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
    else:
        train, _ = split_pairs(pairs)

    manual, singles = load_yue_tsv()
    known = known_yue_words()
    print(f"训练句对 {len(train)}")

    mined = mine_word_pairs(train, known, set(manual), args.min_pair)
    # 人工词表并入多目标集：yue.tsv 的写法是额外合法目标，不是唯一目标
    word_pairs: dict[str, list[str]] = {}
    for z, ys in mined.items():
        word_pairs[z] = list(ys)
    for z, y in manual.items():
        ys = word_pairs.setdefault(z, [])
        if y not in ys:
            ys.insert(0, y)                    # 人工写法排最前
    char_pairs, left, right = mine_char_pairs(train)
    vocab = mine_vocab(train, args.min_vocab)
    data = {
        "word_pairs": word_pairs,
        "singles": singles,
        "char_pairs": prune_counter(char_pairs, args.min_char),
        "left": prune_counter(left, args.min_ctx),
        "right": prune_counter(right, args.min_ctx),
        "vocab": sorted(vocab),
        "colloquial": "".join(sorted(CANTONESE)),
        "trad": load_trad_shell(),
        "classical": sorted(load_classical()),
        "meta": {
            "pairs": len(train),
            "min_pair": args.min_pair,
            "min_char": args.min_char,
            "min_ctx": args.min_ctx,
            "min_vocab": args.min_vocab,
            "all": bool(args.all),
        },
    }
    print(f"word_pairs {len(word_pairs)}，char_pairs {len(data['char_pairs'])}，"
          f"left {len(data['left'])}，right {len(data['right'])}，vocab {len(vocab)}")

    out = args.out or os.path.join(
        ROOT, "data", "yue_guard.marshal" if args.all else "yue_guard.eval.marshal")
    with open(out, "wb") as fh:
        fh.write(zlib.compress(marshal.dumps(data), 6))
    print(f"写出 {out}（{os.path.getsize(out) / 1024:.0f} KiB）")

    # 几个探针：成语的错误替换必须不在表里，口语替换必须在表里
    def in_words(s: str, d: str) -> bool:
        return d in word_pairs.get(s, [])
    for s, d in (("牢", "監"), ("弓", "蛇"), ("是", "係"), ("不", "唔"), ("懂", "識"), ("没", "冇")):
        print(f"  {s}→{d}: char {data['char_pairs'].get((s, d), 0)}, "
              f"left {sum(v for (p, a, b), v in data['left'].items() if a == s and b == d)}, "
              f"right {sum(v for (a, b, n), v in data['right'].items() if a == s and b == d)}, "
              f"word {in_words(s, d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
