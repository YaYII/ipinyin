"""整句切分：把「woxihuanni」切成「我 / 喜欢 / 你」。

用一元语言模型做动态规划：得分 = Σ log P(词) - 每词固定惩罚。
词频来自词库，常用组合会自然胜出。
"""
from __future__ import annotations

from .lexicon import Lexicon

NEG_INF = float("-inf")
WORD_PENALTY = 0.35


def best_path(lexicon: Lexicon, text: str, max_words: int = 12) -> list[str] | None:
    """返回最优切分结果；无法完整覆盖时返回 None。"""
    if not text:
        return None
    size = len(text)
    best_score = [NEG_INF] * (size + 1)
    best_prev = [-1] * (size + 1)
    best_word = [""] * (size + 1)
    best_words = [0] * (size + 1)
    best_score[0] = 0.0

    max_len = min(lexicon.max_key_len, size)
    for start in range(size):
        if best_score[start] == NEG_INF:
            continue
        for end in range(start + 1, min(size, start + max_len) + 1):
            entries = lexicon.entries(text[start:end])
            if not entries:
                continue
            word, freq = entries[0]
            score = best_score[start] + lexicon.score(freq) - WORD_PENALTY
            if best_words[start] + 1 > max_words:
                continue
            if score > best_score[end]:
                best_score[end] = score
                best_prev[end] = start
                best_word[end] = word
                best_words[end] = best_words[start] + 1

    if best_score[size] == NEG_INF:
        return None

    words: list[str] = []
    cursor = size
    while cursor > 0:
        words.append(best_word[cursor])
        cursor = best_prev[cursor]
    words.reverse()
    return words
