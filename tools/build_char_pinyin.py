"""生成「汉字 → 带调号拼音」表（data/chars.marshal）。

为什么需要它：espeak-ng 的普通话声线（cmn）读汉字会**逐字拼英文字母**
（`我`→`w'o`、`们`→`m-e-n-f-a-i-v`、`音`→`y-i-n`），听起来是乱码；
但读拼音是对的（`nǐ hǎo` → `n'i35 X'Au214`）。所以普通话朗读走
「汉字 → 拼音 → espeak」，粤语则直接给汉字（yue 声线有完整字典，简繁都认）。

数据源：vendor/pinyin.txt（mozillazg/pinyin-data，MIT）。
每个字取第一个读音（该字的常用读音），多音字会读错——预览用途可接受。
"""
from __future__ import annotations

import marshal
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "vendor", "pinyin.txt")
OUT = os.path.join(ROOT, "data", "chars.marshal")

PATTERN = re.compile(r"^U\+([0-9A-Fa-f]+):\s*([^#]+)")


def load(path: str) -> dict[str, str]:
    table: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            matched = PATTERN.match(line)
            if not matched:
                continue
            try:
                code = int(matched.group(1), 16)
            except ValueError:
                continue
            # 只收基本汉字区（含〇）；扩展区生僻字对朗读是噪音
            if not (0x4E00 <= code <= 0x9FFF or code == 0x3007):
                continue
            readings = [item.strip() for item in matched.group(2).split(",") if item.strip()]
            if readings:
                table[chr(code)] = readings[0]
    return table


def main() -> int:
    if not os.path.exists(SOURCE):
        print(f"缺 {SOURCE}，先跑 tools/fetch_data.sh", file=sys.stderr)
        return 1
    table = load(SOURCE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as handle:
        marshal.dump(table, handle)
    print(f"{len(table)} 字 → {OUT}（{os.path.getsize(OUT) / 1024:.0f}KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
