#!/usr/bin/env python3
"""B 级音库：每个粤语音节录一个「代表字」单元，任意词都能拼出来。

A 级是词级（覆盖常用词），B 级是音节级（1967 个音节 → 有限集，拼得出来就念得出来）。
单元仍是 edge 录的真音，只是按音节取字：粤拼 si1 → 代表字「诗」→ 该字的单元。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from build_tts_units import build  # noqa: E402
from ipinyin import config, units, yue  # noqa: E402


def representative_chars() -> dict[str, str]:
    """粤拼音节 → 代表字：优先词频高的字，其次码位小的（常用字）。"""
    conv = yue.YueConverter.load_dict(config.yue_dict_path())
    freq: dict[str, int] = {}
    lex = os.path.join(ROOT, "data", "lexicon.tsv")
    if os.path.exists(lex):
        with open(lex, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    for ch in parts[1]:
                        freq[ch] = freq.get(ch, 0) + 1
    buckets: dict[str, list[str]] = {}
    for ch, reading in conv._chars.items():
        buckets.setdefault(reading.strip(), []).append(ch)
    return {s: max(cs, key=lambda c: (freq.get(c, 0), -ord(c)))
            for s, cs in buckets.items()}


def main() -> int:
    out = units.default_root()
    rep = representative_chars()
    path = os.path.join(out, "syllables.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=0, sort_keys=True)
    print(f"音节 {len(rep)} 个 → 代表字表 {path}")
    todo = sorted(set(rep.values()))
    asyncio.run(build(todo, out, workers=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
