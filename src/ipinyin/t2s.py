"""简体 ↔ 繁体单字映射（表来自 OpenCC 的 STCharacters，4012 字）。

两条线要分清（架构上的区别，不是实现细节）：

- **简体打字**：拼音 → 简体字。繁体只是**显示层的字符映射**（`to_traditional`），
  一个字换一个字，不需要模型。所以简中语言模型的语料必须是简体；
  繁体文本要先过 `to_simplified` 再用，否则模型会去学一批简体引擎永远见不到的字形。
- **粤语**：不是字形问题。`早上好 → 早晨`、`我们 → 我哋` 是口语表达不同，
  字面映射做不出来，必须靠规则表 + 神经模型（见 yue.py）。

简→繁是一对多（一个简体可能对应多个繁体），所以反向映射会有多个繁体指向同一个简体；
繁→简把任一繁体映射回它的简体，不需要区分。
"""
from __future__ import annotations

import marshal
import os


class Converter:
    __slots__ = ("_map", "_reverse")

    def __init__(self, mapping: dict[str, str] | None = None):
        self._map = mapping or {}
        reverse: dict[str, str] = {}
        for simp, trad in self._map.items():
            for ch_s, ch_t in zip(simp, trad):
                if ch_s == ch_t or ch_t in reverse:
                    continue
                # 表里有扩展区异体字（如 㓰 是 划 的异体），它们会污染反向映射：
                # 劃 本该回 划，却因为 㓰→劃 这条被映射成 㓰。只接受基本区汉字。
                if not ("\u4e00" <= ch_s <= "\u9fff"):
                    continue
                reverse[ch_t] = ch_s
        self._reverse = reverse

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

    def to_simplified(self, text: str) -> str:
        """繁 → 简。语料归一化用：简中语言模型只该见简体。"""
        if not self._reverse:
            return text
        return "".join(self._reverse.get(ch, ch) for ch in text)

    def __len__(self) -> int:
        return len(self._map)
