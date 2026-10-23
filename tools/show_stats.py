#!/usr/bin/env python3
"""查看个人词频的命中率。

用法：
  python3 tools/show_stats.py            看整体（按选中次数排序的前 30 条）
  python3 tools/show_stats.py huankuan   看某个拼音键下的所有候选及命中率
"""
from __future__ import annotations

import os
import sys

DATA = os.path.join(os.path.expanduser("~"), ".local/share/ipinyin/data", "user_model.tsv")


def load(path: str) -> dict[str, list[tuple[str, int, int]]]:
    table: dict[str, list[tuple[str, int, int]]] = {}
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            key, word = parts[0], parts[1]
            picks = int(parts[2]) if parts[2].isdigit() else 0
            exposures = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            table.setdefault(key, []).append((word, picks, exposures))
    for items in table.values():
        items.sort(key=lambda row: -(row[1] + 2 * row[1] / max(row[2], 1)))
    return table


def main() -> int:
    table = load(DATA)
    if not table:
        print("还没有学习记录（先打几个字）")
        return 0

    if len(sys.argv) > 1:
        key = sys.argv[1]
        rows = table.get(key, [])
        if not rows:
            print(f"没有 {key} 的记录")
            return 1
        print(f"{key}:")
        for word, picks, exposures in rows[:12]:
            rate = picks / exposures if exposures else 0.0
            shown = f"{rate:6.1%}" if exposures else "     -"
        print(f"  {word:<8} 选中 {picks:>4}  曝光 {exposures:>5}  命中率 {shown}")
        return 0

    flat = [(word, key, picks, exposures) for key, rows in table.items() for word, picks, exposures in rows]
    flat.sort(key=lambda row: -row[2])
    print(f"共 {len(table)} 个拼音键 / {len(flat)} 个词条\n")
    print(f"{'拼音键':<16}{'词':<8}{'选中':>6}{'曝光':>8}{'命中率':>10}")
    for word, key, picks, exposures in flat[:30]:
        rate = f"{picks / exposures:>9.1%}" if exposures else "        -"
        print(f"{key:<16}{word:<8}{picks:>6}{exposures:>8}{rate:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
