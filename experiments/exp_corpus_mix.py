#!/usr/bin/env python3
"""实验：不同语料组合训出来的 LM，在「通用整句」和「你自己的文档」上分别如何。

动机：语言模型的收益取决于语域，而不是字数。这个脚本把「多加一批语料到底有没有用」
变成一张可比的表 —— 同一套参数、同一批测试题，只换训练语料。

用法：
  python3 experiments/exp_corpus_mix.py \
      --pairs vendor/yue_pairs.json \
      --extra 小说=vendor/novels 文档=vendor/personal \
      --test-text vendor/personal
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.lm import CharLM  # noqa: E402

DATA = os.path.join(ROOT, "data")


def sentences_from_text(path: str) -> list[str]:
    import glob

    files: list[str] = []
    if os.path.isdir(path):
        for pattern in ("**/*.txt", "**/*.md"):
            files.extend(sorted(glob.glob(os.path.join(path, pattern), recursive=True)))
    elif os.path.exists(path):
        files = [path]
    out: list[str] = []
    import re

    for name in files:
        text = open(name, encoding="utf-8", errors="ignore").read()
        for piece in re.findall(r"[\u4e00-\u9fff]+", text):
            if 4 <= len(piece) <= 40:
                out.append(piece)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--extra", nargs="*", default=[], help="额外语料：名字=路径")
    ap.add_argument("--test-text", default="", help="用这份文本的后 30% 当领域测试集")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--weight", type=float, default=0.8)
    args = ap.parse_args()

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    n_test = max(500, len(pairs) // 20)
    test, train = pairs[:n_test], pairs[2 * n_test:]        # 与训练脚本同一套切分
    base_sents = [p["zh"] for p in train if p.get("zh")]
    print(f"基础语料 {len(base_sents)} 句")

    extras: dict[str, list[str]] = {}
    for spec in args.extra:
        name, _, path = spec.partition("=")
        sents = sentences_from_text(path)
        extras[name] = sents
        print(f"额外语料 {name}: {len(sents)} 句 / {sum(len(s) for s in sents)} 字")

    # 领域测试集：同一份文本的后 30%，且这些句子不进任何训练语料
    domain_test: list[str] = []
    if args.test_text:
        all_sents = sentences_from_text(args.test_text)
        cut = int(len(all_sents) * 0.7)
        domain_test = all_sents[cut:]
        if "文档" in extras:                      # 只让「文档」这一类保留训练部分
            extras["文档"] = all_sents[:cut]

    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    ime = ImeState(lexicon)

    def pinyin_of(text: str) -> str | None:
        # 最长匹配切词再拼拼音（与 tools/eval_sentence.py 同一套逻辑）
        from eval_sentence import build_word_map, to_pinyin

        global _WORD_MAP
        try:
            _WORD_MAP
        except NameError:
            _WORD_MAP = build_word_map(lexicon)
        r = to_pinyin(text, _WORD_MAP, lexicon=lexicon)
        return r[0] if r else None

    def evaluate(sents: list[str]) -> tuple[int, int]:
        """返回 (有效题数, Top1 命中数)。"""
        valid = hit = 0
        for s in sents[: args.n]:
            key = pinyin_of(s)
            if not key:
                continue
            valid += 1
            ime.clear()
            for ch in key:
                ime.insert(ch)
            if ime.view().words[:1] == [s]:
                hit += 1
        return valid, hit

    combos: list[tuple[str, list[str]]] = [("不用 LM", [])]
    combos.append(("基础语料", base_sents))
    for name, sents in extras.items():
        combos.append((f"基础 + {name}", base_sents + sents))
    if len(extras) > 1:
        combos.append(("基础 + 全部", base_sents + [s for v in extras.values() for s in v]))

    rows = []
    for label, sents in combos:
        if sents:
            lm = CharLM(3).fit(sents, min_count=2)
        else:
            lm = None
        ime.attach_lm(lm, args.weight)

        g_valid, g_hit = evaluate([p["zh"] for p in test if p.get("zh")])
        d_valid, d_hit = evaluate(domain_test) if domain_test else (0, 0)
        rows.append((label, g_valid, g_hit, d_valid, d_hit))

    print(f"\n{'语料组合':<14}{'通用 Top1':>12}{'文档 Top1':>12}")
    for label, gv, gh, dv, dh in rows:
        g = f"{gh / gv:.1%} ({gv})" if gv else "—"
        d = f"{dh / dv:.1%} ({dv})" if dv else "—"
        print(f"{label:<14}{g:>12}{d:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
