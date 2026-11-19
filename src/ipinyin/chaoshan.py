"""潮州话（汕头音）字音/词音查询。

数据来自 `tools/build_chaoshan.py` 生成的 data/chaoshan.marshal：
`{"chars": {字: 潮拼}, "words": {词: 潮拼...}, "accent": "st"}`。

只做「读音」不做「改写」：潮州话没有成规模的普通话-潮州话平行语料，
整句翻译做不了（粤语那条线有 21 万句对，这里没有）。所以这里给的是
「这个字/词潮州话怎么念」，也就是候选窗里的第五列。
"""
from __future__ import annotations

import marshal
import os
from typing import Optional


class ChaoshanConverter:
    __slots__ = ("_chars", "_words", "_m2c", "_cache", "_rewrite_cache", "accent")

    CACHE_SIZE = 512

    def __init__(self, chars: dict[str, str], words: dict[str, str], accent: str = "st",
                 m2c: dict[str, str] | None = None):
        self._chars = chars
        self._words = words
        self._m2c = m2c or {}
        self._cache: dict[str, str] = {}
        self._rewrite_cache: dict[str, str] = {}
        self.accent = accent

    @classmethod
    def load(cls, path: str) -> Optional["ChaoshanConverter"]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as handle:
                table = marshal.load(handle)
        except (OSError, ValueError, EOFError):
            return None
        if not isinstance(table, dict) or not table.get("chars"):
            return None
        return cls(table.get("chars", {}), table.get("words", {}),
                   table.get("accent", "st"), table.get("m2c", {}))

    def pengim(self, text: str) -> str:
        """汉字 → 潮拼（汕头音）。词表优先，查不到再逐字；都没有就跳过该字。"""
        if not text:
            return ""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        out: list[str] = []
        index = 0
        size = len(text)
        while index < size:
            matched = ""
            # 词表最长匹配（最多 6 字，够覆盖常用词）
            for length in range(min(6, size - index), 0, -1):
                piece = text[index:index + length]
                reading = self._words.get(piece)
                if reading:
                    out.append(reading)
                    matched = piece
                    break
            if matched:
                index += len(matched)
                continue
            char = text[index]
            reading = self._chars.get(char)
            if reading:
                out.append(reading)
            index += 1

        result = " ".join(" ".join(out).split())
        if len(self._cache) >= self.CACHE_SIZE:
            self._cache.clear()
        self._cache[text] = result
        return result

    def rewrite(self, text: str) -> str:
        """普通话 → 潮汕话书面词（词级映射，最长匹配优先）。

        这是粤语那套「嘅/唔/咗」的对应物：`吃饭`→`食饭`、`我们`→`阮`、
        `什么`→`乜个`。语序仍是普通话语序——潮州话没有平行语料，
        做不到整句改写（粤语那边靠 21 万句对训练的模型）。
        """
        if not text or not self._m2c:
            return text
        cached = self._rewrite_cache.get(text)
        if cached is not None:
            return cached
        out: list[str] = []
        index = 0
        size = len(text)
        while index < size:
            matched = ""
            for length in range(min(6, size - index), 0, -1):
                piece = text[index:index + length]
                replaced = self._m2c.get(piece)
                if replaced:
                    out.append(replaced)
                    matched = piece
                    break
            if matched:
                index += len(matched)
                continue
            out.append(text[index])
            index += 1
        result = "".join(out)
        if len(self._rewrite_cache) >= self.CACHE_SIZE:
            self._rewrite_cache.clear()
        self._rewrite_cache[text] = result
        return result

    def __len__(self) -> int:
        return len(self._chars) + len(self._words)
