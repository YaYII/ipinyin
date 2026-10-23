"""普通话 → 粤语书面用字（词级最长匹配，表在 data/yue.tsv）。"""
from __future__ import annotations

import marshal
import os
import zlib


class YueConverter:
    __slots__ = ("_map", "_max_len", "_chars", "_words", "_max_word")

    def __init__(self, mapping: dict[str, str] | None = None,
                 chars: dict[str, str] | None = None,
                 words: dict[str, str] | None = None):
        self._map = mapping or {}
        self._max_len = max((len(k) for k in self._map), default=1)
        self._chars = chars or {}
        self._words = words or {}
        self._max_word = max((len(k) for k in self._words), default=1)

    @classmethod
    def load(cls, path: str) -> "YueConverter":
        mapping: dict[str, str] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#") or "\t" not in line:
                        continue
                    src, dst = line.split("\t", 1)
                    if src:
                        mapping[src] = dst
        return cls(mapping)

    @classmethod
    def load_dict(cls, path: str, mapping: dict[str, str] | None = None) -> "YueConverter":
        """加载粤拼表（字音 + 词音）。"""
        chars: dict[str, str] = {}
        words: dict[str, str] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    pass                     # 兼容未压缩的旧文件
                data = marshal.loads(raw)
                if isinstance(data, dict):
                    chars = data.get("chars", {}) or {}
                    words = data.get("words", {}) or {}
            except (OSError, ValueError, EOFError):
                pass
        return cls(mapping, chars, words)

    def jyutping(self, text: str) -> str:
        """给一段粤语文字注音：优先整词，其次按字拼。"""
        if not text or (not self._chars and not self._words):
            return ""
        out: list[str] = []
        index = 0
        size = len(text)
        missing = 0
        while index < size:
            for length in range(min(self._max_word, size - index), 0, -1):
                piece = text[index:index + length]
                reading = self._words.get(piece)
                if reading:
                    out.extend(reading.split())
                    index += length
                    break
            else:
                char = text[index]
                reading = self._chars.get(char)
                if reading:
                    out.append(reading)
                else:
                    missing += 1
                index += 1
        if missing >= size:
            return ""
        return " ".join(out)

    def convert(self, text: str) -> str:
        if not self._map:
            return text
        out: list[str] = []
        index = 0
        size = len(text)
        while index < size:
            for length in range(min(self._max_len, size - index), 0, -1):
                piece = text[index : index + length]
                replacement = self._map.get(piece)
                if replacement is not None:
                    out.append(replacement)
                    index += length
                    break
            else:
                out.append(text[index])
                index += 1
        return "".join(out)

    def __len__(self) -> int:
        return len(self._map)
