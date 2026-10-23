#!/usr/bin/env python3
"""把 OpenCC 的简→繁映射压成 data/s2t.marshal（单字映射，够输入法用）。"""
from __future__ import annotations

import marshal
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "vendor", "STCharacters.txt")
OUT = os.path.join(ROOT, "data", "s2t.marshal")


def main() -> int:
    mapping: dict[str, str] = {}
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            simp, targets = parts[0], parts[1].split()
            if simp and targets:
                mapping[simp] = targets[0].replace("爲", "為")   # 统一用「為」
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        marshal.dump(mapping, fh)
    print(f"写出 {OUT}：{len(mapping)} 个简体字有繁体写法")
    for word in ("国", "们", "这", "说", "时"):
        print(f"  {word} → {mapping.get(word, word)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
