"""个人词频模型：每个「拼音键 → 词」都带命中率。

统计三样东西：
  picks      你最终选中它多少次（空格接受记 1 票，数字/点击选词记 3 票）
  exposures  它被展示给你多少次
  hit rate   picks / exposures

排序分 = picks + 2 × 命中率：常用的必然靠前，命中率高的在同频时胜出。
数据落在 ~/.local/share/ipinyin/data/user_model.tsv，一行：
  拼音键 \t 词 \t picks \t exposures
"""
from __future__ import annotations

import os
import threading

MAX_WORDS_PER_KEY = 12


class UserModel:
    def __init__(self, path: str | None = None, seed_path: str | None = None,
                 bigram_path: str | None = None, bigram_seed_path: str | None = None,
                 validator=None):
        # key -> word -> [picks, exposures]
        self._data: dict[str, dict[str, list[int]]] = {}
        self._bigrams: dict[str, dict[str, int]] = {}
        self._path = path
        self._bigram_path = bigram_path
        self._validator = validator
        self._dropped = 0
        self._dirty = False
        self._learned: set[tuple[str, str]] = set()
        self.version = 0  # 每次学到新词 +1，用来让候选缓存失效
        self._lock = threading.Lock()
        if seed_path and os.path.exists(seed_path):
            self._load_file(seed_path, learned=False)
        if path and os.path.exists(path):
            self._load_file(path, learned=True)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if bigram_seed_path and os.path.exists(bigram_seed_path):
            self._load_bigrams(bigram_seed_path)
        if bigram_path and os.path.exists(bigram_path):
            self._load_bigrams(bigram_path)
        if self._dropped:
            self.compact()  # 把脏学习从磁盘上清掉

    # ---------- 读 ----------
    def _load_file(self, path: str, learned: bool = True) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    key, word = parts[0], parts[1]
                    if not key or not word:
                        continue
                    if self._validator and not self._validator(key, word):
                        self._dropped += 1
                        continue
                    try:
                        picks = int(parts[2])
                    except ValueError:
                        continue
                    exposures = 0
                    if len(parts) >= 4:
                        try:
                            exposures = int(parts[3])
                        except ValueError:
                            exposures = 0
                    bucket = self._data.setdefault(key, {})
                    cell = bucket.setdefault(word, [0, 0, 0])
                    if learned:
                        cell[0] += max(picks, 1)
                        cell[1] += max(exposures, 0)
                        self._learned.add((key, word))
                    else:
                        cell[2] += max(picks, 1)   # 种子只做先验，不算"你选过"
        except OSError:
            pass

    def _load_bigrams(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    prev, nxt = parts[0], parts[1]
                    try:
                        count = int(parts[2])
                    except ValueError:
                        continue
                    bucket = self._bigrams.setdefault(prev, {})
                    bucket[nxt] = bucket.get(nxt, 0) + max(count, 1)
        except OSError:
            pass

    # ---------- 查 ----------
    def exposures(self, key: str, word: str) -> int:
        cell = self._data.get(key, {}).get(word)
        return cell[1] if cell else 0

    def hit_rate(self, key: str, word: str) -> float:
        cell = self._data.get(key, {}).get(word)
        if not cell:
            return 0.0
        picks, exposures, _prior = cell
        return picks / exposures if exposures else 0.0

    def picks(self, key: str, word: str) -> int:
        cell = self._data.get(key, {}).get(word)
        return cell[0] if cell else 0

    def score(self, key: str, word: str) -> float:
        cell = self._data.get(key, {}).get(word)
        if not cell:
            return 0.0
        picks, exposures, prior = cell
        rate = picks / exposures if exposures else 0.0
        return prior + picks + 2.0 * rate

    def preferred(self, key: str, limit: int = MAX_WORDS_PER_KEY) -> list[str]:
        bucket = self._data.get(key)
        if not bucket:
            return []
        ranked = sorted(bucket.items(), key=lambda item: -self.score(key, item[0]))
        return [word for word, _ in ranked[:limit]]

    def after(self, prev: str | None, limit: int = 9) -> list[str]:
        """紧跟着 prev 之后最常出现的词——用来"猜你下一句要打什么"。"""
        if not prev:
            return []
        bucket = self._bigrams.get(prev)
        if not bucket:
            return []
        ranked = sorted(bucket.items(), key=lambda item: -item[1])
        return [word for word, _ in ranked[:limit]]

    # ---------- 学 ----------
    def observe(self, key: str, words: list[str]) -> None:
        """候选展示给用户时记一次曝光（只记内存，压缩时落盘）。

        只记「已知的词」（种子/学过的）：从未选中的词不会落盘、得分恒为 0，
        给它们建 cell 纯属泄漏 —— 每次按键都会调这里，评测脚本全量跑
        曾因此 OOM（每 400 句 ~3.4MB 线性增长）。
        """
        if not key or not words:
            return
        bucket = self._data.get(key)
        if not bucket:
            return
        changed = False
        for word in words:
            cell = bucket.get(word)
            if cell is not None:
                cell[1] += 1
                changed = True
        if changed:
            self._dirty = True

    def learn(self, key: str, word: str, weight: int = 1) -> None:
        if not key or not word or key == word:
            return
        if len(word) > 24 or len(key) > 32:
            return
        if self._validator and not self._validator(key, word):
            return
        with self._lock:
            bucket = self._data.setdefault(key, {})
            cell = bucket.setdefault(word, [0, 0, 0])
            cell[0] += weight
            cell[1] = max(cell[1], 1)
            self._learned.add((key, word))
            self.version += 1
            if self._path is None:
                return
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(f"{key}\t{word}\t{weight}\t{cell[1]}\n")
            except OSError:
                pass

    def learn_bigram(self, prev: str | None, word: str) -> None:
        if not prev or not word or prev == word:
            return
        with self._lock:
            bucket = self._bigrams.setdefault(prev, {})
            bucket[word] = bucket.get(word, 0) + 1
            if self._bigram_path is None:
                return
            try:
                with open(self._bigram_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{prev}\t{word}\t1\n")
            except OSError:
                pass

    # ---------- 维护 ----------
    def compact(self) -> None:
        """压实成每键每词一条记录（含曝光数），供人直接查看命中率。"""
        if not self._path:
            return
        with self._lock:
            tmp = self._path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write("# 拼音键\t词\t选中次数\t曝光次数\n")
                    # 只落盘"学到的"，不把种子词写回去——否则每次重启种子权重会翻倍
                    for key, word in sorted(self._learned):
                        if word not in self._data.get(key, {}):
                            continue
                        picks, exposures, _prior = self._data[key][word]
                        fh.write(f"{key}\t{word}\t{picks}\t{exposures}\n")
                os.replace(tmp, self._path)
                self._dirty = False
            except OSError:
                pass

    def maybe_compact(self, threshold: int = 5000) -> None:
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                lines = sum(1 for _ in fh)
        except OSError:
            return
        if self._dirty or lines > threshold:
            self.compact()

    def flush_if_dirty(self) -> None:
        if self._dirty:
            self.compact()

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._data.values()) + sum(
            len(bucket) for bucket in self._bigrams.values()
        )
