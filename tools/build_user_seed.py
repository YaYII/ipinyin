#!/usr/bin/env python3
"""从用户的语料里挖词，生成 data/user_seed.tsv（预置的个人词频）。

做两件事：
  1. 用词库做中文分词（按词频的动态规划），统计你用过的词；
  2. 反查词库得到拼音键，词库里没有的生词按单字读音拼出拼音键。

用法：python3 tools/build_user_seed.py [语料文件]
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lexicon import Lexicon  # noqa: E402

LEXICON = os.path.join(ROOT, "data", "lexicon.tsv")
OUT = os.path.join(ROOT, "data", "user_seed.tsv")
CHINESE = re.compile(r"[\u4e00-\u9fff]+")
NEG_INF = float("-inf")


def build_index(lexicon: Lexicon) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    """返回 (词→频次, 词→拼音键, 字→拼音)。"""
    word_freq: dict[str, int] = {}
    word_key: dict[str, str] = {}
    char_key: dict[str, str] = {}
    for key in lexicon.iter_keys():
        for word, freq in lexicon.entries(key):
            if word_freq.get(word, 0) < freq:
                word_freq[word] = freq
            word_key.setdefault(word, key)
            if len(word) == 1:
                char_key.setdefault(word, key)
    return word_freq, word_key, char_key


def segment_maxmatch(text: str, word_freq: dict[str, int], max_len: int) -> list[str]:
    """正向最大匹配：优先切最长的已知词。

    这里刻意不用词频 DP——单字词频（几十万）天然压制词组频（几百），
    一元模型会把「你好」切成「你/好」。挖词场景下，最大匹配更可靠。
    """
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        for length in range(min(max_len, size - index), 1, -1):
            word = text[index : index + length]
            if word in word_freq:
                out.append(word)
                index += length
                break
        else:
            out.append(text[index])
            index += 1
    return out


def segment(text: str, word_freq: dict[str, int], max_len: int) -> list[str]:
    """按词频做动态规划分词；词库没有的字按单字处理。"""
    size = len(text)
    best = [NEG_INF] * (size + 1)
    prev = [-1] * (size + 1)
    pick = [""] * (size + 1)
    best[0] = 0.0
    for start in range(size):
        if best[start] == NEG_INF:
            continue
        for end in range(start + 1, min(size, start + max_len) + 1):
            word = text[start:end]
            freq = word_freq.get(word)
            if freq is None:
                if end != start + 1:  # 不在词库里，只允许单字
                    continue
                freq = 1
            score = best[start] + math.log(freq + 1)
            if score > best[end]:
                best[end] = score
                prev[end] = start
                pick[end] = word
    words: list[str] = []
    cursor = size
    while cursor > 0:
        words.append(pick[cursor])
        cursor = prev[cursor]
    words.reverse()
    return words


def main() -> int:
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tools", "user_corpus.txt")
    corpus = open(corpus_path, encoding="utf-8").read()

    lexicon = Lexicon.from_tsv(LEXICON)
    word_freq, word_key, char_key = build_index(lexicon)
    max_len = max(len(word) for word in word_freq)

    counts: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for match in CHINESE.finditer(corpus):
        words = segment_maxmatch(match.group(0), word_freq, max_len)
        for word in words:
            if len(word) >= 2:
                counts[word] += 1
        # 相邻词对：用来"猜你下一句"
        for left, right in zip(words, words[1:]):
            if len(left) + len(right) >= 3:
                pairs[(left, right)] += 1

    rows: list[tuple[str, str, int]] = []
    missed: list[str] = []
    for word, count in counts.most_common(300):
        key = word_key.get(word)
        if key is None:
            parts = [char_key.get(ch) for ch in word]
            key = "".join(parts) if all(parts) else None
        if key is None:
            missed.append(word)
            continue
        rows.append((key, word, count))

    # 从语料里补 2 字组合：词库没有、但你在文本里反复用的（只会/之会这类）
    char_counts: Counter[str] = Counter(ch for run in CHINESE.findall(corpus) for ch in run)
    pair_counts: Counter[str] = Counter()
    for run in CHINESE.findall(corpus):
        for left, right in zip(run, run[1:]):
            if char_counts[left] >= 5 and char_counts[right] >= 5:
                pair_counts[left + right] += 1
    for pair, count in pair_counts.most_common(80):
        if count < 4 or pair in word_freq:
            continue
        parts = [char_key.get(ch) for ch in pair]
        if not all(parts):
            continue
        key = "".join(parts)
        if (key, pair) in {(k, w) for k, w, _ in rows}:
            continue
        rows.append((key, pair, min(count, 40)))

    # 手写的常用词表：词库没有的词（如"卡顿"）靠它补进来，权重给高，排最前
    extra_path = os.path.join(ROOT, "tools", "user_words.txt")
    seen_keys = {(key, word) for key, word, _ in rows}
    if os.path.exists(extra_path):
        for line in open(extra_path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or len(line.split()) != 2:
                continue
            word, key = line.split()
            if (key, word) in seen_keys:
                continue
            rows.append((key, word, 60))
            seen_keys.add((key, word))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# 由 tools/build_user_seed.py 生成：拼音键 词 次数\n")
        for key, word, count in rows:
            fh.write(f"{key}\t{word}\t{min(count * 2, 200)}\n")

    bigram_out = os.path.join(ROOT, "data", "user_bigram_seed.tsv")
    with open(bigram_out, "w", encoding="utf-8") as fh:
        fh.write("# 由 tools/build_user_seed.py 生成：前一个词 后一个词 次数\n")
        for (left, right), count in pairs.most_common(5000):
            if count < 1:
                continue
            fh.write(f"{left}\t{right}\t{count}\n")

    print(f"写出 {bigram_out}：{min(len(pairs), 5000)} 个词对")
    print(f"写出 {OUT}：{len(rows)} 个词")
    if missed:
        print(f"无法定音跳过：{'、'.join(missed[:10])}")
    print("示例：", "、".join(f"{w}({c})" for _, w, c in rows[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
