#!/usr/bin/env python3
"""编译粤语发音表：字音 + 词音（数据来自 rime-cantonese，CC BY 4.0）。

输出 data/yue.marshal：
  {"chars": {字: 粵拼}, "words": {詞: 粵拼}}
词的收录范围：出现在 data/yue.tsv 里的粤语词 + 含粤语口语字的词。
"""
from __future__ import annotations

import marshal
import os

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


def main() -> int:
    chars = dict(load_pairs(CHARS))
    wanted = {dst for _src, dst in load_pairs(YUE_TSV)}
    words: dict[str, str] = {}
    for word, jyutping in load_pairs(WORDS):
        if word in wanted or (COLLOQUIAL & set(word)):
            words.setdefault(word, jyutping)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        marshal.dump({"chars": chars, "words": words}, fh)
    print(f"字音 {len(chars)} 条，词音 {len(words)} 条 → {OUT}")
    for probe in ("我哋", "佢哋", "唔該", "而家", "点解", "冇"):
        print(f"  {probe}: {words.get(probe) or '（按字拼）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
