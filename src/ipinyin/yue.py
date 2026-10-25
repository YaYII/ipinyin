"""普通话 → 粤语书面用字（词级最长匹配，表在 data/yue.tsv）。"""
from __future__ import annotations

import marshal
import os
import zlib

from . import yue_rules


class YueConverter:
    __slots__ = ("_map", "_max_len", "_chars", "_words", "_max_word", "_phrase",
                 "_max_phrase", "nn", "guard")

    def __init__(self, mapping: dict[str, str] | None = None,
                 chars: dict[str, str] | None = None,
                 words: dict[str, str] | None = None,
                 phrase: dict[str, str] | None = None):
        self._map = mapping or {}
        self._max_len = max((len(k) for k in self._map), default=1)
        self._phrase = phrase or {}
        self._max_phrase = max((len(k) for k in self._phrase), default=1)
        self.nn = None          # 可选：NeuralYue，存在时优先用于长句
        self.guard = None       # 可选：YueGuard，约束神经解码（P1）
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
        phrase: dict[str, str] = {}
        ppath = os.path.join(os.path.dirname(path), "yue_phrase.marshal") if path else ""
        if ppath and os.path.exists(ppath):
            try:
                with open(ppath, "rb") as fh:
                    raw = fh.read()
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    pass
                loaded = marshal.loads(raw)
                if isinstance(loaded, dict):
                    phrase = loaded
            except (OSError, ValueError, EOFError):
                pass
        return cls(mapping, chars, words, phrase)

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

    def convert(self, text: str, rules: bool = True, neural: bool | None = None) -> str:
        # neural=None 时默认走神经（装了模型就用）。但调用方要注意：
        # 输入法的候选列表里每个候选都会调一次 convert，2~4 字词实测单次 7~10ms，
        # 9 个候选全算就是 60ms+/键。正确用法是只对「首选候选/上屏内容」传 neural=True，
        # 其余候选走规则（0.02ms）。见 src/ipinyin/ime.py 的调用点。
        if neural is None:
            neural = True
        if neural and self.nn is not None:
            try:
                result = self.nn.translate(text)
            except Exception:
                result = ""
            if result:
                if self.guard is not None:
                    # P1 约束解码：无实证的改动回退规则版（成语/古诗防误改）
                    result = self.guard.check(
                        text, result, repair=lambda seg: self.convert(seg, neural=False))
                if result:
                    return result
        if rules:
            text = yue_rules.apply(text)      # 先调语序，再换字
        if not self._map:
            return text
        out: list[str] = []
        index = 0
        size = len(text)
        while index < size:
            matched = False
            # 1) 统计得到的短语（学到的地道说法，最长优先）
            for length in range(min(self._max_phrase, size - index), 1, -1):
                replacement = self._phrase.get(text[index:index + length])
                if replacement is not None:
                    out.append(replacement)
                    index += length
                    matched = True
                    break
            if matched:
                continue
            # 2) 人工词表
            for length in range(min(self._max_len, size - index), 0, -1):
                replacement = self._map.get(text[index:index + length])
                if replacement is not None:
                    out.append(replacement)
                    index += length
                    matched = True
                    break
            if matched:
                continue
            out.append(text[index])
            index += 1
        return "".join(out)

    def __len__(self) -> int:
        return len(self._map)
