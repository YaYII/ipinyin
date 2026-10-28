"""词库：拼音键 → 候选词。

磁盘上有两种格式：
  lexicon.tsv      人手可读/可编辑（拼音键 \t 词 \t 词频）
  lexicon.marshal  紧凑快照（拼音键 → "词:词频|词:词频"），启动只读它

快照只做一件事：把「启动时构造 16 万个元组」的开销降下来（880ms → 200ms），
因为引擎启动慢会卡住 IBus 主循环，用户感受到的是「键盘卡住，然后飙出一堆字」。
每个键的内容按需解析并做少量缓存，单个键约 20µs。
"""
from __future__ import annotations

import bisect
import math
import marshal
import os
from typing import Iterable

_CACHE_LIMIT = 4096


class Lexicon:
    __slots__ = (
        "_raw", "_prefix_table", "_keys", "max_key_len", "log_total", "total_freq",
        "_cache", "_syllables", "_syllable_prefixes", "_split_cache", "_initials",
    )

    def __init__(self, raw: dict[str, str], total_freq: int, prefix_table: dict[str, str] | None = None):
        self._raw = raw
        self._prefix_table = prefix_table or {}
        self._keys = sorted(raw)
        self.max_key_len = max((len(k) for k in raw), default=0)
        self.total_freq = max(total_freq, 1)
        self.log_total = math.log(self.total_freq)
        self._cache: dict[str, list[tuple[str, int]]] = {}
        self._split_cache: dict[str, set[int]] = {}
        # 合法音节表：单字条目的键就是一个音节
        self._syllables = {
            key for key in self._keys
            if any(len(word) == 1 for word, _ in self.entries(key))
        }
        # 所有音节的前缀（用来判断"这串字母像不像打了一半的拼音"）
        self._initials: dict[str, list[tuple[str, int]]] = {}
        self._syllable_prefixes = {
            syllable[:length]
            for syllable in self._syllables
            for length in range(1, min(3, len(syllable)) + 1)
        }

    # ---------- 构造 ----------
    @classmethod
    def load(cls, path: str) -> "Lexicon":
        snapshot = os.path.join(os.path.dirname(path), "lexicon.marshal")
        if os.path.exists(snapshot):
            try:
                with open(snapshot, "rb") as fh:
                    raw, total = marshal.load(fh)
                if isinstance(raw, dict) and raw:
                    return cls(raw, int(total), cls._load_prefix_table(path))
            except (OSError, ValueError, EOFError, TypeError):
                pass  # 快照损坏或 Python 版本不兼容，退回解析文本
        return cls.from_tsv(path)

    @staticmethod
    def _load_prefix_table(path: str) -> dict[str, str]:
        prefix_file = os.path.join(os.path.dirname(path), "prefix.marshal")
        try:
            with open(prefix_file, "rb") as fh:
                table = marshal.load(fh)
            return table if isinstance(table, dict) else {}
        except (OSError, ValueError, EOFError, TypeError):
            return {}

    @classmethod
    def from_tsv(cls, path: str) -> "Lexicon":
        table: dict[str, list[str]] = {}
        total = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, word, freq = line.rstrip("\n").split("\t")
                weight = int(freq)
                table.setdefault(key, []).append(f"{word}:{weight}")
                total += weight
        raw = {key: "|".join(items) for key, items in table.items()}
        return cls(raw, total)

    # ---------- 查询 ----------
    def entries(self, key: str) -> list[tuple[str, int]]:
        """整串精确匹配，按词频降序（快照里已经排好序）。"""
        payload = self._raw.get(key)
        if payload is None:
            return []
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        parsed = self._parse(payload)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = parsed
        return parsed

    @staticmethod
    def _parse(payload: str) -> list[tuple[str, int]]:
        parsed: list[tuple[str, int]] = []
        for segment in payload.split("|"):
            word, _, freq = segment.rpartition(":")
            parsed.append((word, int(freq)))
        return parsed

    def has(self, key: str) -> bool:
        return key in self._raw

    def score(self, freq: int) -> float:
        """一元语言模型打分：log P(word)，越大越常见。"""
        return math.log(max(freq, 1)) - self.log_total

    def syllable_counts(self, key: str) -> set[int]:
        """这个拼音串能切成几个音节（用于校验 拼音↔汉字 数量是否对得上）。"""
        cached = self._split_cache.get(key)
        if cached is not None:
            return cached
        size = len(key)
        reachable: list[set[int]] = [set() for _ in range(size + 1)]
        reachable[0].add(0)
        for start in range(size):
            if not reachable[start]:
                continue
            for end in range(start + 1, min(size, start + 6) + 1):
                if key[start:end] in self._syllables:
                    for count in reachable[start]:
                        reachable[end].add(count + 1)
        result = reachable[size]
        if len(self._split_cache) > 4096:
            self._split_cache.clear()
        self._split_cache[key] = result
        return result

    def load_initials(self, path: str) -> "Lexicon":
        """载入简拼索引（如果→rg）。文件不存在就只是不启用，不影响其它功能。"""
        if not path or not os.path.exists(path):
            return self
        try:
            import zlib

            with open(path, "rb") as fh:
                raw = fh.read()
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass
            table = marshal.loads(raw)
        except (OSError, ValueError, EOFError):
            return self
        parsed: dict[str, list[tuple[str, int]]] = {}
        for code, payload in table.items():
            items: list[tuple[str, int]] = []
            for piece in payload.split("|"):
                if ":" not in piece:
                    continue
                word, _, freq = piece.rpartition(":")
                try:
                    items.append((word, int(freq)))
                except ValueError:
                    continue
            if items:
                parsed[code] = items
        self._initials = parsed
        return self

    def initials_entries(self, code: str) -> list[tuple[str, int]]:
        return self._initials.get(code, [])

    def is_syllable_prefix(self, text: str) -> bool:
        return text in self._syllable_prefixes

    def split_syllables(self, key: str) -> list[str]:
        """把拼音串切成音节（最长匹配），切不开返回空表。"""
        out: list[str] = []
        index = 0
        size = len(key)
        while index < size:
            for length in range(min(6, size - index), 0, -1):
                piece = key[index:index + length]
                if piece in self._syllables:
                    out.append(piece)
                    index += length
                    break
            else:
                return []
        return out

    def prefix_entries(
        self, key: str, key_limit: int = 80, word_limit: int = 300
    ) -> list[tuple[str, str, int]]:
        """和 prefix 一样，但把拼音键一起返回（用来判断"是不是只差一两个字母"）。"""
        if not key:
            return []
        if len(key) <= 3:
            payload = self._prefix_table.get(key)
            if not payload:
                return []
            out: list[tuple[str, str, int]] = []
            for segment in payload.split("|"):
                word, _, freq = segment.rpartition(":")
                out.append((key, word, int(freq)))
            return out[:word_limit]
        start = bisect.bisect_left(self._keys, key)
        found: list[tuple[str, str, int]] = []
        scanned = 0
        for index in range(start, len(self._keys)):
            candidate = self._keys[index]
            if not candidate.startswith(key):
                break
            scanned += 1
            if scanned > key_limit:
                break
            if candidate == key:
                continue
            for word, freq in self.entries(candidate):
                found.append((candidate, word, freq))
            if len(found) > word_limit * 3:
                break
        found.sort(key=lambda item: -item[2])
        return found[:word_limit]

    def prefix(self, key: str, key_limit: int = 80, word_limit: int = 300) -> list[tuple[str, int]]:
        """以 key 开头的词，按词频降序返回（不含精确匹配本身）。"""
        if not key:
            return []
        # 短前缀查预计算表（整表按词频排好），避免"按字母序扫前 150 个键"这种偏差
        if len(key) <= 3:
            payload = self._prefix_table.get(key)
            return self._parse(payload)[:word_limit] if payload else []
        start = bisect.bisect_left(self._keys, key)
        found: list[tuple[str, int]] = []
        scanned = 0
        for index in range(start, len(self._keys)):
            candidate = self._keys[index]
            if not candidate.startswith(key):
                break
            scanned += 1
            if scanned > key_limit:
                break
            if candidate == key:
                continue
            found.extend(self.entries(candidate))
            if len(found) > word_limit * 3:
                break
        found.sort(key=lambda item: -item[1])
        return found[:word_limit]

    def iter_keys(self) -> Iterable[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._raw)
