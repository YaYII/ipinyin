"""整句切分：把「woxihuanni」切成「我 / 喜欢 / 你」。

用一元语言模型做动态规划：得分 = Σ log P(词) - 每词固定惩罚。
词频来自词库，常用组合会自然胜出。

`best_paths()` 额外给出前 k 条路径 —— 只有一个候选意味着「打错一个字就没有第二选择」，
而上下文重排（词二元 / 神经语言模型）必须先在候选里挑，所以 n-best 是前置条件。
"""
from __future__ import annotations

from .lexicon import Lexicon

NEG_INF = float("-inf")
WORD_PENALTY = 0.35


def best_path(lexicon: Lexicon, text: str, max_words: int = 12) -> list[str] | None:
    """返回最优切分结果；无法完整覆盖时返回 None。"""
    paths = best_paths(lexicon, text, k=1, max_words=max_words)
    return paths[0] if paths else None


def best_paths(lexicon: Lexicon, text: str, k: int = 3, max_words: int = 12,
               prune: int = 6) -> list[list[str]]:
    """返回前 k 条切分（按得分从高到低，同一结果只留一条）。"""
    return [path for _score, path in best_paths_scored(lexicon, text, k, max_words, prune)]


def best_paths_scored(lexicon: Lexicon, text: str, k: int = 3, max_words: int = 12,
                      prune: int = 6) -> list[tuple[float, list[str]]]:
    """同 best_paths，但带上一元得分 —— 上下文重排需要「词库先验 + 上下文分」插值，
    只有排名没法插值（实测拿排名当先验的测法会得出「上下文毫无作用」的错误结论）。"""
    if not text or k <= 0:
        return []
    size = len(text)
    keep = max(k, prune)
    # 每个位置：[(得分, 词序列), ...]，按得分降序，最多 keep 条
    frontier: list[list[tuple[float, list[str]]]] = [[] for _ in range(size + 1)]
    frontier[0] = [(0.0, [])]
    max_len = min(lexicon.max_key_len, size)

    for start in range(size):
        if not frontier[start]:
            continue
        for end in range(start + 1, min(size, start + max_len) + 1):
            entries = lexicon.entries(text[start:end])
            if not entries:
                continue
            word, freq = entries[0]
            add = lexicon.score(freq) - WORD_PENALTY
            bucket = frontier[end]
            for prev_score, path in frontier[start]:
                if len(path) + 1 > max_words:
                    continue
                bucket.append((prev_score + add, path + [word]))
            if len(bucket) > keep:
                bucket.sort(key=lambda kv: -kv[0])
                del bucket[keep:]

    if not frontier[size]:
        return []
    frontier[size].sort(key=lambda kv: -kv[0])
    out: list[tuple[float, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for score, path in frontier[size]:
        key = tuple(path)
        if key in seen or not path:
            continue
        seen.add(key)
        out.append((score, path))
        if len(out) >= k:
            break
    return out
