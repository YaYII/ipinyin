#!/usr/bin/env python3
"""编译粤语发音表：字音 + 词音（数据来自 rime-cantonese，CC BY 4.0）。

输出 data/yue.marshal：
  {"chars": {字: 粵拼}, "words": {詞: 粵拼}}
词的收录范围：出现在 data/yue.tsv 里的粤语词 + 含粤语口语字的词。
"""
from __future__ import annotations

import marshal
import os
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARS = os.path.join(ROOT, "vendor", "yue_chars.yaml")
WORDS = os.path.join(ROOT, "vendor", "yue_words.yaml")
YUE_TSV = os.path.join(ROOT, "data", "yue.tsv")
OUT = os.path.join(ROOT, "data", "yue.marshal")

# 只在粤语里常用的口语字：用它们判断一个词是不是"粤语词"
COLLOQUIAL = set("佢哋嘅咗唔係喺冇乜嘢嗰嚟睇咁啲畀攞瞓啱攰嬲諗搵掟嘥冚唞啩啫咩噉嚿")


def load_pairs(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def load_readings(path: str) -> dict[str, list[tuple[str, float | None]]]:
    """字 → [(读音, 百分比)]。rime-cantonese 里变体带百分比，主读音不带。"""
    out: dict[str, list[tuple[str, float | None]]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            pct: float | None = None
            if len(parts) >= 3 and parts[2].endswith("%"):
                try:
                    pct = float(parts[2].rstrip("%"))
                except ValueError:
                    pct = None
            out.setdefault(parts[0], []).append((parts[1], pct))
    return out


def pick_main(entries: list[tuple[str, float | None]]) -> str:
    """多读音字取主读音：不带百分比的那条优先，没有就取百分比最高的。

    不能直接取第一行——源的排序不按常用度：`去` 的第一行是 heoi2 3%（异读），
    主读音 heoi3 在第二行。实测 5423 个多读音字里 1194 个（22%）会被取错。
    """
    plain = [reading for reading, pct in entries if pct is None]
    if plain:
        return plain[0]
    return max(entries, key=lambda entry: entry[1] or 0.0)[0]


def main() -> int:
    chars: dict[str, str] = {
        char: pick_main(entries) for char, entries in load_readings(CHARS).items()
    }
    words: dict[str, str] = {}
    for word, jyutping in load_pairs(WORDS):
        words.setdefault(word, jyutping)      # 全量词音，注音才准（如 中國 zung1 gwok3）

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(zlib.compress(marshal.dumps({"chars": chars, "words": words}), 6))
    print(f"字音 {len(chars)} 条，词音 {len(words)} 条 → {OUT}")
    for probe in ("我哋", "佢哋", "唔該", "而家", "点解", "冇"):
        print(f"  {probe}: {words.get(probe) or '（按字拼）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
