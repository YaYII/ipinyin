#!/usr/bin/env python3
"""中文输入质量基准：用你自己的语料当考题，测"打对了拼音，词能不能一次出来"。

指标：
  Top1  第一个候选就是目标词
  Top3  前三里有
  Top9  第一页里有（换句话说：不用翻页）

用法：
  python3 tools/eval_quality.py [题数]                    # 用 tools/user_corpus.txt
  python3 tools/eval_quality.py --log                     # 用真机输入日志（最真实）
  python3 tools/eval_quality.py --log ~/my_history.tsv    # 指定日志
  python3 tools/eval_quality.py --corpus 我的语料.txt
"""
from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.ime import ImeState            # noqa: E402
from ipinyin.lexicon import Lexicon          # noqa: E402
from ipinyin.user_model import UserModel     # noqa: E402

DATA = os.environ.get("IPINYIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".local/share/ipinyin/data")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")      # 没装过的机器（如 macOS 开发环境）直接用仓库里的数据
CORPUS = os.path.join(ROOT, "tools", "user_corpus.txt")


def build_cases(lexicon: Lexicon, limit: int, corpus: str = CORPUS) -> list[tuple[str, str]]:
    """从语料里取"词 + 它的拼音"，作为输入题面。"""
    word_key: dict[str, str] = {}
    for key in lexicon.iter_keys():
        for word, _ in lexicon.entries(key):
            word_key.setdefault(word, key)

    import re
    text = open(corpus, encoding="utf-8").read()
    words: list[str] = []
    seen: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for length in (2, 3, 4):
            for start in range(0, len(run) - length + 1):
                word = run[start:start + length]
                if word in word_key and word not in seen:
                    seen.add(word)
                    words.append(word)
    random.Random(7).shuffle(words)
    return [(word_key[w], w) for w in words[:limit]]


def build_cases_from_log(path: str) -> list[tuple[str, str]]:
    """用真机输入日志当题库：user_model.tsv 一行是「拼音键 \t 词 \t 选中次数 \t 展示次数」。

    这是最真实的考题 —— 都是你自己打过、并且真的选中的词。
    """
    cases: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            picks = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
            if picks < 1:
                continue
            item = (parts[0], parts[1])
            if item in seen:
                continue
            seen.add(item)
            cases.append(item)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=400, help="题数上限")
    ap.add_argument("--log", nargs="?", const="__default__", default="",
                    help="用真机输入日志当题库（默认找 ~/.local/share/ipinyin/data/user_model.tsv）")
    ap.add_argument("--corpus", default=CORPUS, help="语料文件")
    args = ap.parse_args()
    limit = args.n
    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))

    def valid(key: str, word: str) -> bool:
        if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
            return True
        return len(word) in lexicon.syllable_counts(key)

    model = UserModel(
        os.path.join(DATA, "user_model.tsv"),
        os.path.join(DATA, "user_seed.tsv"),
        os.path.join(DATA, "user_bigram.tsv"),
        os.path.join(DATA, "user_bigram_seed.tsv"),
        validator=valid,
    )
    ime = ImeState(lexicon, user_model=model)

    if args.log:
        path = args.log
        if path == "__default__":
            path = os.path.join(DATA, "user_model.tsv")
        if not os.path.exists(path):
            print(f"找不到输入日志 {path}", file=sys.stderr)
            return 1
        cases = build_cases_from_log(path)[:limit]
        print(f"题库：真机输入日志 {path}")
    else:
        cases = build_cases(lexicon, limit, args.corpus)
    hit1 = hit3 = hit9 = 0
    misses: list[tuple[str, str, str]] = []
    for key, word in cases:
        ime.clear()
        for char in key:
            ime.insert(char)
        # 用 words（实际上屏内容），不是 candidates（四列展示串 "词  繁  粤  粤拼"）
        candidates = ime.view().words
        if candidates[:1] == [word]:
            hit1 += 1
        elif word in candidates[:3]:
            hit3 += 1
        if word in candidates[:9]:
            hit9 += 1
        else:
            misses.append((key, word, "、".join(candidates[:4])))

    total = len(cases)
    print(f"考题 {total} 条" + ("（真机输入日志里你选过的词）" if args.log else "（语料里的 2~4 字词）"))
    print(f"  Top1（一次就对） : {hit1:4d}  {hit1/total:6.1%}")
    print(f"  前3             : {hit3:4d}  {hit3/total:6.1%}")
    print(f"  第一页（<=9）    : {hit9:4d}  {hit9/total:6.1%}")
    print(f"  综合一次命中率   : {(hit1 + hit3)/total:6.1%}")
    if misses:
        print("\n掉出第一页的（前 15 个）：")
        for key, word, got in misses[:15]:
            print(f"  {key:<14} 应为 {word}  实际 {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
