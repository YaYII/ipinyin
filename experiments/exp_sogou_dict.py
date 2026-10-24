#!/usr/bin/env python3
"""对照评测：现词库 vs 加上搜狗细胞词库，看输入效果差多少。

两个口径都要跑：
  1. **通用语料**（vendor/yue_pairs）—— 看搜狗那些成语/历史词会不会挤掉常用词；
  2. **你的文档**（≤8 字）—— 看搜狗词库能不能补上缺词。

用法：
  python3 experiments/exp_sogou_dict.py --lex /tmp/lex_sogou/lexicon.tsv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.lm import CharLM  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402


def sentences_from_text(path: str, max_len: int = 8) -> list[str]:
    conv = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
    out: list[str] = []
    for name in sorted(glob.glob(os.path.join(path, "*.txt"))):
        text = conv.to_simplified(open(name, encoding="utf-8", errors="ignore").read())
        for piece in re.findall(r"[\u4e00-\u9fff]+", text):
            for sent in piece.split():
                if 4 <= len(sent) <= max_len:
                    out.append(sent)
    return out


def run(lexicon: Lexicon, cases: list[tuple[str, str]]) -> tuple[int, int]:
    """cases = [(拼音键, 目标词)]，返回 (有效数, Top1 命中数)。"""
    ime = ImeState(lexicon)
    valid = hit = 0
    for key, want in cases:
        ime.clear()
        for ch in key:
            ime.insert(ch)
        words = ime.view().words
        if not words:
            continue
        valid += 1
        if words[:1] == [want]:
            hit += 1
    return valid, hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lex", default="/tmp/lex_sogou/lexicon.tsv", help="加了搜狗词的词库")
    ap.add_argument("--base", default=os.path.join(ROOT, "data", "lexicon.tsv"))
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    from eval_sentence import build_word_map, to_pinyin

    base = Lexicon.load(args.base)
    merged = Lexicon.load(args.lex)
    print(f"词库：{len(base)} 词 → {len(merged)} 词（+{len(merged) - len(base)}）\n")

    # 通用语料：词级 Top1（整句题面太严，词级更能反映「能不能打出来」）
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    import random

    random.seed(0)
    random.shuffle(pairs)
    word_map = build_word_map(base)
    gen_cases = []
    for p in pairs:
        zh = (p.get("zh") or "").strip()
        if not zh or not all("\u4e00" <= c <= "\u9fff" for c in zh):
            continue
        r = to_pinyin(zh, word_map, lexicon=base)
        if r:
            gen_cases.append((r[0], zh))
        if len(gen_cases) >= args.n:
            break

    # 你的文档：≤8 字
    doc_sents = sentences_from_text(os.path.join(ROOT, "vendor", "personal", "_clean"))
    doc_cases = []
    for s in doc_sents:
        r = to_pinyin(s, word_map, lexicon=base)
        if r:
            doc_cases.append((r[0], s))

    print(f"{'题库':<14}{'词库':<12}{'题数':>7}{'Top1':>9}{'变化':>9}")
    for label, cases in (("通用语料", gen_cases), ("你的文档", doc_cases)):
        base_v, base_h = run(base, cases)
        new_v, new_h = run(merged, cases)
        print(f"{label:<14}{'现词库':<12}{base_v:>7}{base_h / base_v:>9.1%}{'':>9}")
        print(f"{label:<14}{'+搜狗':<12}{new_v:>7}{new_h / new_v:>9.1%}"
              f"{(new_h - base_h) / base_v * 100:>+8.1f}pt")

    # 关键补充：搜狗新词本身能不能一次打出来。
    # 上面两个基准几乎不动，不是「词库没用」，而是这些新词在语料里占比太低
    # （你的 17,988 句里只有 217 句含新词 = 1.2%）。
    have = {w for k in base.iter_keys() for w, _ in base.entries(k)}
    new_words = []
    for line in open(args.lex, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] not in have:
            new_words.append((parts[0], parts[1]))
    import random

    random.seed(2)
    sample = random.sample(new_words, min(800, len(new_words)))
    print(f"\n搜狗独有词 {len(new_words)} 个，抽查 {len(sample)} 个「能不能一次打出来」：")
    for label, lex in (("现词库", base), ("+搜狗", merged)):
        ime = ImeState(lex)
        hit = page = 0
        for key, want in sample:
            ime.clear()
            for ch in key:
                ime.insert(ch)
            words = ime.view().words
            hit += words[:1] == [want]
            page += want in words[:9]
        n = len(sample)
        print(f"  {label:<8} 首候选 {hit / n:5.1%}   前 9 {page / n:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
