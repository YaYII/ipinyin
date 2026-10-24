#!/usr/bin/env python3
"""从你自己的文档里挖「词库没有的领域词」，生成 user_words.txt 的补充条目。

为什么：长句打不出来的主因不是模型，是**词库里没有你的领域词**。
实测 16 字以上的句子 77% 失败，失败样本里大量出现「系统根据、导入规则、自动生成、
日排班表、服务供应」这类词库缺词。补进去，整句一次命中才能上去。

判据（避免把跨词接缝当词报出来）：
  - 2~4 字片段，出现次数 >= --min；
  - 词库里没有整词；
  - **不能**被两个以内的已知多字词拼出来（`系统根据` = 系统 + 根据 → 拼得出，丢掉；
    `排班表` 拼不出 → 真缺词，留下）；
  - 不含虚词开头/结尾。

拼音：按单字读音拼（多音字取该字最高频的读音），所以生成的结果要**人工过一眼**，
多音字可能猜错（如「行」在「日排班表」里读 háng，但单字最高频是 xing）。

用法：
  python3 tools/suggest_words.py                       # 只打印，不动文件
  python3 tools/suggest_words.py --write               # 追加进 tools/user_words.txt
  python3 tools/suggest_words.py --min 4 --top 50
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402

DATA = os.environ.get("IPINYIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".local/share/ipinyin/data")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")

STOP = set("的了着过是在和与及或就都也还很太更最不没别被把给让对从向往于")
CJK = re.compile(r"[\u4e00-\u9fff]+")


def char_pinyin(lexicon: Lexicon) -> dict[str, str]:
    """单字 → 拼音键（取最高频那个读音）。"""
    out: dict[str, str] = {}
    for key in lexicon.iter_keys():
        for word, freq in lexicon.entries(key):
            if len(word) == 1 and word not in out:
                out[word] = key
    return out


def composable(piece: str, known: set[str], max_word: int = 4) -> bool:
    i = 0
    pieces = 0
    size = len(piece)
    while i < size:
        for length in range(min(max_word, size - i), 1, -1):
            if piece[i:i + length] in known:
                i += length
                pieces += 1
                break
        else:
            i += 1
            pieces += 1
    return pieces <= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "vendor", "personal", "_clean"))
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--write", action="store_true", help="追加进 tools/user_words.txt")
    args = ap.parse_args()

    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    conv = Converter.load(os.path.join(DATA, "s2t.marshal"))
    known: set[str] = set()
    for key in lexicon.iter_keys():
        for word, _freq in lexicon.entries(key):
            known.add(word)
    py = char_pinyin(lexicon)

    freq: Counter[str] = Counter()
    for path in sorted(glob.glob(os.path.join(args.src, "*.txt"))):
        text = conv.to_simplified(open(path, encoding="utf-8", errors="ignore").read())
        for run in CJK.findall(text):
            for n in (2, 3, 4):
                for i in range(len(run) - n + 1):
                    piece = run[i:i + n]
                    if piece[0] in STOP or piece[-1] in STOP:
                        continue
                    freq[piece] += 1

    # 两道降噪：
    # 1. 含虚词（的了和是就不…）的片段基本是跨词接缝，不是词；
    # 2. 是更高频片段的子串的丢掉（「下拉选」是「下拉选择」的一部分）。
    maximal: set[str] = set()
    for word, count in freq.items():
        if count < args.min:
            continue
        longer = [w for w in freq
                  if len(w) > len(word) and word in w and freq[w] >= count and freq[w] >= args.min]
        if not longer:
            maximal.add(word)

    missing = [(w, c) for w, c in freq.items()
               if c >= args.min and w not in known and w in maximal
               and not (STOP & set(w)) and not composable(w, known)]
    missing.sort(key=lambda kv: (-kv[1], -len(kv[0])))

    rows = []
    for word, count in missing[: args.top]:
        key = "".join(py.get(ch, "") for ch in word)
        if not key or not key.isalpha():
            continue
        rows.append((word, key, count))

    print(f"扫 {args.src}：缺词候选 {len(missing)} 个，取前 {len(rows)}\n")
    print(f"{'词':<10}{'拼音（自动猜）':<16}{'出现':>5}")
    for word, key, count in rows:
        print(f"{word:<10}{key:<16}{count:>5}")

    if args.write and rows:
        path = os.path.join(ROOT, "tools", "user_words.txt")
        existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        have = {ln.split()[0] for ln in existing.splitlines()
                if ln.strip() and not ln.startswith("#") and len(ln.split()) == 2}
        new = [(w, k) for w, k, _c in rows if w not in have]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n# --- 由 tools/suggest_words.py 从 vendor/personal 生成（多音字需人工核对）---\n")
            for word, key in new:
                fh.write(f"{word} {key}\n")
        print(f"\n已追加 {len(new)} 条到 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
