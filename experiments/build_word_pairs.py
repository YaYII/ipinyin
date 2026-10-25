#!/usr/bin/env python3
"""从 train 切分挖「真词级」平行对（zh→yue），给神经训练补词级信号。

与 tools/build_yue_phrase.mine_phrases 的差别（那是给评测题面用的，宁缺毋滥）：
  - key 必须是 data/lexicon.tsv 里的真词（≥2 字）——防跨界碎片（得他/了很）；
  - 保留 top-K 变体（什么→乜嘢 / 什么→咩 都是合法粤语，分开成多条训练对）；
  - 门槛放宽（训练容忍统计噪声，评测不容忍）；
  - 排除成语/古诗 key（tools/idioms.txt、poems.txt——正确行为是照抄，不是转换）。

产出 JSON（默认 data/yue_word_pairs.json）：[{"zh": "...", "yue": "..."}]
切分与 train_yue_seq2seq.py 完全一致（seed=0），只用 train 侧——考题互斥。

用法：python3 experiments/build_word_pairs.py [--min-count 3] [--variants 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ZH_RANGE = (2, 5)
YUE_RANGE = (2, 6)


def ngrams(text: str, lo: int, hi: int) -> set[str]:
    return {text[i:i + k] for i in range(len(text))
            for k in range(lo, min(hi, len(text) - i) + 1)}


def load_lexicon_words() -> set[str]:
    """输入法词库的真词（≥2 字）。jieba 词表存在则并进来。"""
    words: set[str] = set()
    lf = os.path.join(ROOT, "data", "lexicon.tsv")
    if os.path.exists(lf):
        for line in open(lf, encoding="utf-8"):
            if "\t" in line and not line.startswith("#"):
                for w in line.rstrip("\n").split("\t")[1].split():
                    if len(w) >= 2:
                        words.add(w)
    jf = os.path.join(ROOT, "vendor", "jieba.txt")
    if os.path.exists(jf):
        for line in open(jf, encoding="utf-8"):
            parts = line.split()
            if parts and len(parts[0]) >= 2:
                words.add(parts[0])
    return words


def load_protected() -> set[str]:
    """正确行为是照抄的 key（成语/古诗）——训练对里不许出现。"""
    prot: set[str] = set()
    for name in ("idioms.txt", "poems.txt"):
        p = os.path.join(ROOT, "tools", name)
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                s = ln.strip()
                if s and not s.startswith("#"):
                    prot.add(s)
    return prot


def split_train(pairs: list[dict]) -> list[dict]:
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"]
             and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
    pairs = pairs[:150000]
    n_test = max(500, len(pairs) // 20)
    return pairs[2 * n_test:]            # train 侧（与 trainer / 守卫 builder 一致）


def extract_anchored(train: list[dict], words: set[str], prot: set[str]):
    """锚定对齐提取：zh key 在 yue 侧的对应区间用「两侧稳定字符」卡边界。

    纯共现统计会把 帶眼识人→帶眼識人 里的 key 眼识 对到整条 帶眼識人
    （越界扩伸）；锚定后 span 只能是 眼識。对齐用 difflib 的 equal 块当锚
    （imperfect 但求边界，不求逐字对）。
    span 长度限 [len(key)-1, len(key)]：缩一格（知道→知）合法，扩一格
    多是边界污染（漢字嘅/秋天係——比 key 长且含其译形）。
    返回 (zh频次, key→Counter[span])。
    """
    import difflib
    zh_freq: Counter[str] = Counter()
    co: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in train:
        zh, yue = item["zh"], item["yue"]
        keys = {g for g in ngrams(zh, *ZH_RANGE) if g in words and g not in prot}
        if not keys:
            continue
        zh2yue: dict[int, int] = {}
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, zh, yue, autojunk=False).get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    zh2yue[i1 + k] = j1 + k
        for key in keys:
            start = zh.find(key)
            end = start + len(key)
            left = max((i for i in range(start) if i in zh2yue), default=None)
            right = min((i for i in range(end, len(zh)) if i in zh2yue), default=None)
            ys = 0 if left is None else zh2yue[left] + 1
            ye = len(yue) if right is None else zh2yue[right]
            span = yue[ys:ye]
            if not span or not (len(key) - 1 <= len(span) <= len(key)):
                continue
            zh_freq[key] += 1
            co[key][span] += 1
    return zh_freq, co


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "yue_word_pairs.json"))
    ap.add_argument("--min-count", type=int, default=3, help="zh key 的最低出现句数")
    ap.add_argument("--min-co", type=int, default=2, help="key→目标 的最低共现次数")
    ap.add_argument("--variants", type=int, default=3, help="每个 key 保留的变体数")
    ap.add_argument("--ratio", type=float, default=0.5, help="主导目标的最低占比")
    ap.add_argument("--variant-ratio", type=float, default=0.15,
                    help="变体的最低占比（合法变体天然低于主导目标）")
    args = ap.parse_args()

    words = load_lexicon_words()
    prot = load_protected()
    train = split_train(json.load(open(args.pairs, encoding="utf-8")))
    print(f"train 句对 {len(train)}；真词表 {len(words)}；保护 key {len(prot)}")

    zh_freq, co = extract_anchored(train, words, prot)
    out: list[dict] = []
    n_keys = 0
    for key, counter in co.items():
        total = zh_freq[key]
        if total < args.min_count:
            continue
        cands = []
        for y, c in counter.most_common():
            if y == key:
                continue
            share = c / max(1, total)
            if share >= args.ratio or (share >= args.variant_ratio and c >= args.min_co):
                cands.append((y, share))
            if len(cands) >= args.variants:
                break
        if not cands:
            continue
        n_keys += 1
        for y, _share in cands:
            out.append({"zh": key, "yue": y})
    random.seed(0)
    random.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=0)
    lens = Counter(len(p["zh"]) for p in out)
    print(f"挖出 {len(out)} 条词级平行对（{n_keys} 个 key，长度分布 {dict(sorted(lens.items()))}）")
    print(f"已写 {args.out}（{os.path.getsize(args.out) // 1024} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
