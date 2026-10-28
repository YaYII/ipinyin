#!/usr/bin/env python3
"""构建简拼索引：词的每个音节取首字母（如果 → rg）。

搜狗/谷歌/Rime 都是这个原理：把「全拼」和「首字母缩写」两套码都指向同一个词，
按词频排序、再用个人词频调整。产出 data/initials.marshal（zlib 压缩）。
"""
from __future__ import annotations

import marshal
import os
import sys
import zlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lexicon import Lexicon  # noqa: E402

OUT = os.path.join(ROOT, "data", "initials.marshal")
MIN_LEN, MAX_LEN = 2, 6


def main() -> int:
    lex = Lexicon.load(os.path.join(ROOT, "data", "lexicon.tsv"))
    table: defaultdict[str, dict[str, int]] = defaultdict(dict)
    keys = 0
    for key in lex.iter_keys():
        syllables = lex.split_syllables(key)
        if not syllables or not (MIN_LEN <= len(syllables) <= MAX_LEN):
            continue
        initials = "".join(s[0] for s in syllables)
        if len(initials) != len(syllables):
            continue
        for word, freq in lex.entries(key):
            if len(word) != len(syllables):     # 字数必须和音节数一致，否则不是简拼码
                continue
            if freq > table[initials].get(word, 0):
                table[initials][word] = freq
        keys += 1

    compact = {
        code: "|".join(f"{w}:{f}" for w, f in sorted(items.items(), key=lambda kv: -kv[1])[:12])
        for code, items in table.items()
    }

    # 混拼码：首音节取首字母 + 其余音节全拼（wmen→我们、rguo→如果、nhao→你好）
    mixed: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for key in lex.iter_keys():
        syllables = lex.split_syllables(key)
        if not syllables or not (2 <= len(syllables) <= 4) or len(syllables) != len(key) - sum(len(s) - 1 for s in syllables):
            pass
        if not syllables or not (2 <= len(syllables) <= 4):
            continue
        code = syllables[0][0] + "".join(syllables[1:])
        for word, freq in lex.entries(key):
            if len(word) != len(syllables):
                continue
            if freq > mixed[code].get(word, 0):
                mixed[code][word] = freq
    compact_mixed = {
        code: "|".join(f"{w}:{f}" for w, f in sorted(items.items(), key=lambda kv: -kv[1])[:8])
        for code, items in mixed.items() if max(items.values()) >= 50
    }
    with open(OUT, "wb") as fh:
        fh.write(zlib.compress(marshal.dumps({"initials": compact, "mixed": compact_mixed}), 6))
    print(f"简拼码 {len(compact)} 个 / 混拼码 {len(compact_mixed)} 个（来自 {keys} 个拼音键）"
          f"→ {OUT}（{os.path.getsize(OUT)/1024:.0f} KiB）")
    for probe in ("wmen", "rguo", "nhao"):
        print(f"  混拼 {probe}: {compact_mixed.get(probe, '—')[:50]}")
    for probe in ("rg", "nh", "wm", "zhg"):
        print(f"  {probe}: {compact.get(probe, '—')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
