"""简体 → 繁体（单字映射，表来自 OpenCC 的 STCharacters，4012 字）。"""
from __future__ import annotations

import marshal
import os


class Converter:
    __slots__ = ("_map",)

    def __init__(self, mapping: dict[str, str] | None = None):
        self._map = mapping or {}

    @classmethod
    def load(cls, path: str) -> "Converter":
        if not path or not os.path.exists(path):
            return cls({})
        try:
            with open(path, "rb") as fh:
                table = marshal.load(fh)
            return cls(table if isinstance(table, dict) else {})
        except (OSError, ValueError, EOFError):
            return cls({})

    def to_traditional(self, text: str) -> str:
        if not self._map:
            return text
        return "".join(self._map.get(ch, ch) for ch in text)

    def __len__(self) -> int:
        return len(self._map)
