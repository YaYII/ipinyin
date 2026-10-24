#!/usr/bin/env python3
"""实验：整句候选从 1 条变 k 条，再用上下文（词二元）重排，看能涨多少。

现状：segmenter.best_path 用一元词频做一次 Viterbi，只产出**一个**整句候选，
      同音字全靠词频硬拼 —— 这是整句命中率低的根因。
本实验：k-best Viterbi 出 k 条候选，用词二元 LM 重排。

用法：python3 experiments/exp_sentence_lm.py [--k 8] [--n 300]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.segmenter import WORD_PENALTY, best_path  # noqa: E402

DATA = os.path.join(ROOT, "data")
NEG_INF = float("-inf")


def build_word_map(lexicon: Lexicon) -> dict[str, str]:
    best: dict[str, tuple[int, str]] = {}
    for key in lexicon.iter_keys():
        for word, freq in lexicon.entries(key):
            old = best.get(word)
            if old is None or freq > old[0]:
                best[word] = (freq, key)
    return {w: k for w, (_, k) in best.items()}


def to_pinyin(sentence: str, word_map: dict[str, str], max_word: int = 6) -> tuple[str, list[str]] | None:
    out: list[str] = []
    words: list[str] = []
    i = 0
    size = len(sentence)
    while i < size:
        for length in range(min(max_word, size - i), 0, -1):
            piece = sentence[i:i + length]
            key = word_map.get(piece)
            if key:
                out.append(key)
                words.append(piece)
                i += length
                break
        else:
            return None
    return "".join(out), words


def kbest_paths(lexicon: Lexicon, text: str, k: int = 8, max_words: int = 16) -> list[tuple[float, list[str]]]:
    """每个位置保留前 k 条（得分, 词序列），最后取终点的前 k 条。"""
    size = len(text)
    best: list[list[tuple[float, list[str]]]] = [[] for _ in range(size + 1)]
    best[0] = [(0.0, [])]
    max_len = min(lexicon.max_key_len, size)
    for start in range(size):
        if not best[start]:
            continue
        for end in range(start + 1, min(size, start + max_len) + 1):
            entries = lexicon.entries(text[start:end])
            if not entries:
                continue
            word, freq = entries[0]
            add = lexicon.score(freq) - WORD_PENALTY
            for score, path in best[start]:
                if len(path) + 1 > max_words:
                    continue
                best[end].append((score + add, path + [word]))
            if len(best[end]) > k * 3:          # 剪枝，防爆炸
                best[end].sort(key=lambda kv: -kv[0])
                best[end] = best[end][:k * 3]
    out = sorted(best[size], key=lambda kv: -kv[0])[:k]
    return out


class CharNgram:
    """字级 3-gram LM（stupid backoff）。中文同音字的区分主要靠字上下文，
    词级二元看不出来的「是/事/时」，字 trigram 能看出来。"""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: list[Counter] = [Counter() for _ in range(order + 1)]
        self.ctx: list[Counter] = [Counter() for _ in range(order + 1)]

    def fit(self, sentences: list[str]) -> None:
        for s in sentences:
            padded = "^" * (self.order - 1) + s + "$"
            for i, ch in enumerate(padded):
                if i < self.order - 1:
                    continue
                for n in range(1, self.order + 1):
                    gram = padded[i - n + 1:i + 1]
                    self.counts[n][gram] += 1
                    if n > 1:
                        self.ctx[n][gram[:-1]] += 1
        self.total = sum(self.counts[1].values())

    def logp(self, sentence: str) -> float:
        padded = "^" * (self.order - 1) + sentence + "$"
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            for n in range(self.order, 0, -1):
                gram = padded[i - n + 1:i + 1]
                if n == 1:
                    total += math.log((self.counts[1].get(gram, 0) + 0.1) /
                                      (self.total + 0.1 * 30000))
                    break
                c = self.ctx[n].get(gram[:-1], 0)
                num = self.counts[n].get(gram, 0)
                if c and num:
                    total += math.log(num / c)
                    break
        return total


class Bigram:
    """词二元 LM，Kneser-Ney 太复杂，这里用简单插值 + 回退。"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.uni: Counter[str] = Counter()
        self.bi: dict[str, Counter[str]] = defaultdict(Counter)
        self.total = 0
        self.vocab = 0

    def fit(self, sentences: list[list[str]]) -> None:
        for words in sentences:
            prev = "<s>"
            for w in words:
                self.uni[w] += 1
                self.bi[prev][w] += 1
                prev = w
                self.total += 1
        self.vocab = len(self.uni) + 1

    def logp(self, prev: str, word: str) -> float:
        ctx = sum(self.bi[prev].values()) if prev in self.bi else 0
        p_uni = (self.uni.get(word, 0) + 1) / (self.total + self.vocab)
        if ctx == 0:
            return math.log(p_uni)
        p_bi = (self.bi[prev].get(word, 0) + self.alpha * self.vocab * p_uni) / (ctx + self.alpha * self.vocab)
        return math.log(max(p_bi, 1e-12))

    def score(self, words: list[str]) -> float:
        prev = "<s>"
        s = 0.0
        for w in words:
            s += self.logp(prev, w)
            prev = w
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--train", type=int, default=60000)
    args = ap.parse_args()

    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    word_map = build_word_map(lexicon)

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    seen: set[str] = set()
    sentences: list[str] = []
    for p in pairs:
        zh = (p.get("zh") or "").strip()
        if not zh or zh in seen or not all("\u4e00" <= c <= "\u9fff" for c in zh):
            continue
        seen.add(zh)
        sentences.append(zh)

    # 前 20000 句留作测试，后面 6 万句训练 LM
    test_pool, train_pool = sentences[:20000], sentences[20000:20000 + args.train]

    lm = Bigram()
    train_segs = [r[1] for s in train_pool if (r := to_pinyin(s, word_map))]
    lm.fit(train_segs)
    clm = CharNgram(order=3)
    clm.fit([s for s in train_pool if all("\u4e00" <= c <= "\u9fff" for c in s)])
    print(f"LM 训练：{len(train_segs)} 句 / {lm.total} 词 / 词表 {lm.vocab}；字 trigram 已训练")

    buckets: dict[str, list[str]] = {"2-4字": [], "5-7字": [], "8-15字": [], "16+字": []}
    for s in test_pool:
        n = len(s)
        key = "2-4字" if n <= 4 else "5-7字" if n <= 7 else "8-15字" if n <= 15 else "16+字"
        if len(buckets[key]) < args.n:
            buckets[key].append(s)

    print(f"{'长度':<8}{'有效题':>6}{'现引擎':>9}{'+词二元':>10}{'+字trigram':>12}{'两者相加':>10}")
    tot = base_ok = lm_ok = ch_ok = mix_ok = 0
    for name, items in buckets.items():
        n1 = n2 = n3 = n4 = valid = 0
        for s in items:
            r = to_pinyin(s, word_map)
            if not r:
                continue
            pinyin, _ = r
            valid += 1
            path = best_path(lexicon, pinyin)
            if path and "".join(path) == s:
                n1 += 1
            cands = kbest_paths(lexicon, pinyin, k=args.k)
            if cands:
                def text_of(path_words):
                    return "".join(path_words)
                r2 = sorted(cands, key=lambda kv: -lm.score(kv[1]))
                if text_of(r2[0][1]) == s:
                    n2 += 1
                r3 = sorted(cands, key=lambda kv: -clm.logp(text_of(kv[1])))
                if text_of(r3[0][1]) == s:
                    n3 += 1
                r4 = sorted(cands, key=lambda kv: -(0.5 * lm.score(kv[1]) / max(len(kv[1]), 1)
                                                    + clm.logp(text_of(kv[1])) / max(len(text_of(kv[1])), 1)))
                if text_of(r4[0][1]) == s:
                    n4 += 1
        if not valid:
            print(f"{name:<8}{0:>6}  （没有合法题面）")
            continue
        print(f"{name:<8}{valid:>6}{n1 / valid:>9.1%}{n2 / valid:>10.1%}{n3 / valid:>12.1%}{n4 / valid:>10.1%}")
        tot += valid
        base_ok += n1
        lm_ok += n2
        ch_ok += n3
        mix_ok += n4
    if tot:
        print(f"{'合计':<8}{tot:>6}{base_ok / tot:>9.1%}{lm_ok / tot:>10.1%}"
              f"{ch_ok / tot:>12.1%}{mix_ok / tot:>10.1%}")
        print(f"相对现引擎：词二元 {(lm_ok - base_ok) / tot * 100:+.1f}pt | "
              f"字trigram {(ch_ok - base_ok) / tot * 100:+.1f}pt | "
              f"相加 {(mix_ok - base_ok) / tot * 100:+.1f}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
