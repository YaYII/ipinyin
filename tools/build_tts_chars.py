#!/usr/bin/env python3
"""字级音库：按词频建常用字单元，字齐了任意词都能拼出来。

A 级（本脚本）解决「每个字都有音」；词级库只是锦上添花——词再全也追不上自由输入，
字是有限集（常用 ~8000），建一次就够。

用法：
    python3 tools/build_tts_chars.py                  # 前 8000 字
    python3 tools/build_tts_chars.py --chars 4000 --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from build_tts_units import build  # noqa: E402
from ipinyin import units  # noqa: E402


def char_freq() -> collections.Counter:
    freq: collections.Counter = collections.Counter()
    path = os.path.join(ROOT, "data", "lexicon.tsv")
    if not os.path.exists(path):
        return freq
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                for ch in parts[1]:
                    if "\u4e00" <= ch <= "\u9fff":
                        freq[ch] += 1
    return freq


def main() -> int:
    ap = argparse.ArgumentParser(description="按词频预生成字级音库")
    ap.add_argument("--chars", type=int, default=8000, help="字上限（默认 8000）")
    ap.add_argument("--workers", type=int, default=8, help="并发数（默认 8）")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--skip-syllables", action="store_true",
                    help="不并入音节代表字（普通话音库用，粤语音节表与它无关）")
    args = ap.parse_args()

    out = args.out or units.default_root()
    freq = char_freq()
    common = [c for c, _ in freq.most_common(args.chars)]

    # 音节代表字也要覆盖，否则音节兜底会缺音
    rep_path = os.path.join(out, "syllables.json")
    rep: dict[str, str] = {}
    if not args.skip_syllables and os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as fh:
            rep = json.load(fh)

    seen: set[str] = set()
    chars: list[str] = []
    for ch in common + list(rep.values()):      # 高频在前：常用字先建好，生僻字失败不影响
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
    print(f"字库目标 {len(chars)} 个（词频前 {args.chars} + 音节代表字）", flush=True)
    asyncio.run(build(chars, out, workers=args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
