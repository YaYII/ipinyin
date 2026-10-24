#!/usr/bin/env python3
"""训练字级语言模型（给整句候选重排用）。

语料来源（按优先级）：
  1. --corpus 指定的纯文本（一行一句）
  2. --pairs 平行语料，只取中文侧（默认 vendor/yue_pairs.json）
  3. tools/user_corpus.txt

用法：
  python3 tools/build_lm.py                      # 用 vendor/yue_pairs.json
  python3 tools/build_lm.py --corpus 我的语料.txt
  python3 tools/build_lm.py --order 4 --min 3    # 更大/更小
输出：data/lm_char.marshal（zlib 压缩，默认约 10MB，别提交进仓库）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lm import CharLM  # noqa: E402

CN = re.compile(r"[\u4e00-\u9fff]+")


def sentences_from_text(path: str) -> list[str]:
    """文件或目录（递归收 .txt/.md）。按标点断句，只留 4~40 字的句子。"""
    import glob

    files: list[str] = []
    if os.path.isdir(path):
        for pattern in ("**/*.txt", "**/*.md"):
            files.extend(sorted(glob.glob(os.path.join(path, pattern), recursive=True)))
    elif os.path.exists(path):
        files = [path]
    else:
        print(f"  跳过不存在的路径 {path}")
        return []

    out: list[str] = []
    for name in files:
        with open(name, encoding="utf-8") as fh:
            text = fh.read()
        for piece in CN.findall(text):
            for sent in piece.split():
                if 4 <= len(sent) <= 40:
                    out.append(sent)
    return out


def sentences_from_pairs(path: str) -> list[str]:
    pairs = json.load(open(path, encoding="utf-8"))
    return [p["zh"] for p in pairs if p.get("zh")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=[], help="纯文本语料：文件或目录，可多个（如你自己的文档）")
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--min", type=int, default=2, help="长 gram 的最小出现次数（剪枝）")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "lm_char.marshal"))
    args = ap.parse_args()

    if args.corpus:
        sents = []
        for path in args.corpus:
            if path.endswith(".json"):          # 平行语料，只取中文侧
                got = sentences_from_pairs(path)
            else:
                got = sentences_from_text(path)
            print(f"  语料 {path}: {len(got)} 句")
            sents.extend(got)
    elif os.path.exists(args.pairs):
        sents = sentences_from_pairs(args.pairs)
        print(f"语料：{args.pairs}")
    else:
        sents = sentences_from_text(os.path.join(ROOT, "tools", "user_corpus.txt"))
        print("语料：tools/user_corpus.txt（平行语料不在，只能用自带的 10 句）")

    sents = [s for s in sents if s]
    print(f"训练 {len(sents)} 句 / {sum(len(s) for s in sents)} 字")
    lm = CharLM(args.order).fit(sents, min_count=args.min)
    lm.save(args.out)
    print(f"写出 {args.out}（{os.path.getsize(args.out) / 1e6:.1f} MB，"
          f"1-gram {len(lm.counts[1])} / 2-gram {len(lm.counts[2])} / "
          f"{args.order}-gram {len(lm.counts[args.order])}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
