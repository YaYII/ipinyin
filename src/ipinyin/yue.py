"""普通话 → 粤语书面用字（词级最长匹配，表在 data/yue.tsv）。"""
from __future__ import annotations

import os


class YueConverter:
    __slots__ = ("_map", "_max_len")

    def __init__(self, mapping: dict[str, str] | None = None):
        self._map = mapping or {}
        self._max_len = max((len(k) for k in self._map), default=1)

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
