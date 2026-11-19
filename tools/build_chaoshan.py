"""生成潮州话（汕头音）字音/词音表：data/chaoshan.marshal。

数据源：p1an-lin-jung/teochew-g2p（BSD-3-Clause），底本是府城音，
按 accent_convert/to_Swatow.txt 转成**汕头音**（用户指定）。

产物结构：
    {"chars": {字: "pengim"}, "words": {词: "pengim pengim"}, "accent": "st"}

约定：只取每个字的第一个读音（最常用）。多音字按词表优先——`word_common.txt`
里的词条直接给整词读音，比逐字拼准。
"""
from __future__ import annotations

import marshal
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "vendor", "teochew")
OUT = os.path.join(ROOT, "data", "chaoshan.marshal")

# 字#读音1|读音2  /  词#读音1 读音2
LINE = re.compile(r"^(.+?)#(.+)$")


def read_pairs(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            matched = LINE.match(line)
            if matched:
                pairs.append((matched.group(1), matched.group(2)))
    return pairs


def load_accent(path: str) -> dict[tuple[str, str], str]:
    """汕头音替换表：字_府城音节#汕头音节。"""
    table: dict[tuple[str, str], str] = {}
    for head, target in read_pairs(path):
        if "_" not in head:
            continue
        char, _, syllable = head.partition("_")
        table[(char, syllable)] = target.strip()
    return table


def to_swatow(char: str, syllable: str, accent: dict[tuple[str, str], str]) -> str:
    return accent.get((char, syllable), syllable)


def build() -> dict:
    accent = load_accent(os.path.join(SRC, "to_Swatow.txt"))

    chars: dict[str, str] = {}
    for name in ("char_main.txt", "char_low_frequency.txt", "char_extension.txt"):
        for char, readings in read_pairs(os.path.join(SRC, name)):
            if len(char) != 1 or char in chars:
                continue          # 先到的优先：char_main 是常用字表
            first = readings.split("|")[0].strip().rstrip("*")
            if not first:
                continue
            chars[char] = to_swatow(char, first, accent)

    words: dict[str, str] = {}
    for word, readings in read_pairs(os.path.join(SRC, "word_common.txt")):
        syllables = readings.split()
        if len(word) != len(syllables):
            continue              # 音节数和字数对不上（多音/合音），跳过
        words[word] = " ".join(
            to_swatow(char, syllable, accent) for char, syllable in zip(word, syllables)
        )

    return {"chars": chars, "words": words, "accent": "st"}


def main() -> int:
    if not os.path.exists(os.path.join(SRC, "char_main.txt")):
        print(f"缺 {SRC}/char_main.txt，先跑 tools/fetch_teochew.sh", file=sys.stderr)
        return 1
    table = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as handle:
        marshal.dump(table, handle)
    size = os.path.getsize(OUT) / 1024
    print(f"潮州话（汕头音）：{len(table['chars'])} 字 / {len(table['words'])} 词 → {OUT}（{size:.0f}KB）")
    for probe in ("你好", "食饭", "自己", "潮州", "汕头"):
        readings = table["words"].get(probe) or "".join(
            table["chars"].get(ch, "?") for ch in probe
        )
        print(f"  {probe} → {readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
