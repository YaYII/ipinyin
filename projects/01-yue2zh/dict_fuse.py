#!/usr/bin/env python3
"""P2 词典融合（最小可行版）：反向词典 + beam 候选重排。

思路：把仓库现成的两个「普→粤」词表反转成「粤→普」词典，只保留**无冲突**
（同一粤语词在所有来源里都映射到同一个普通话词）的条目。对 beam search 的
候选句打一个「词典覆盖度」分：输入句按最长匹配切出词典词，候选里包含其
锁定译文的越多，覆盖度越高。最终排序 = 长度归一的 beam 分 + λ × 覆盖度。

这正是正向 YueGuard 思路的反向松版：不做解码期硬约束（字符级 seq2seq 无对齐
信息，硬锁位置做不到），改为候选级软融合——查表命中的词天然免疫同音错字与
语气词摇摆，与 --typo 增强正交可叠加。

数据源（全部本地、只读）：
  data/yue.tsv            规则词表（zh→yue，注释行 # 开头）
  data/yue_word_pairs.json 词级平行对（experiments/build_word_pairs.py 挖矿）

纯函数、不依赖 torch。
"""
from __future__ import annotations

import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YUE_TSV = os.path.join(ROOT, "data", "yue.tsv")
YUE_WORD_PAIRS = os.path.join(ROOT, "data", "yue_word_pairs.json")


def load_tsv_pairs(path: str) -> list[tuple[str, str]]:
    """yue.tsv → [(zh, yue)]；跳过注释/段头，无 tab 的行。"""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            zh, yue = line.split("\t", 1)
            zh, yue = zh.strip(), yue.strip()
            if zh and yue:
                pairs.append((zh, yue))
    return pairs


def load_json_pairs(path: str) -> list[tuple[str, str]]:
    """yue_word_pairs.json → [(zh, yue)]。"""
    data = json.load(open(path, encoding="utf-8"))
    return [(r["zh"], r["yue"]) for r in data if r.get("zh") and r.get("yue")]


def build_reverse_dict(pair_lists: list[list[tuple[str, str]]],
                       corpus_pairs: list[dict] | None = None,
                       min_seen: int = 3, min_ratio: float = 0.5,
                       min_len: int = 1, max_len: int = 12) -> dict[str, str]:
    """多源反转：粤语词 → 普通话词。

    两级策略：
    1. 有语料（corpus_pairs = [{zh, yue}]）时用**共现校验**：对每个粤语词的
       各候选译文，数它在语料里与该粤语词的同句共现率，取最高者，且共现率须
       ≥ min_ratio。这一步同时解决两个问题——同义多映射（我们/咱们→我哋：
       语料里「我们」与「我哋」共现占主导，正确保留）和挖矿反义噪声
       （唔得閒→有空：共现率≈0，整条否决）。
    2. 无语料或语料见次 < min_seen（生僻词）时退回**全票一致**规则。

    恒等映射（zh == yue）不携带信息，一律丢弃。
    """
    votes: dict[str, Counter] = {}
    for pairs in pair_lists:
        for zh, yue in pairs:
            if not (min_len <= len(yue) <= max_len):
                continue
            votes.setdefault(yue, Counter())[zh] += 1

    stats: dict[str, tuple[int, Counter]] = {}      # yue -> (出现句数, 各候选共现数)
    if corpus_pairs:
        keys = {w for w in votes if len(w) <= max_len}
        max_key = max((len(k) for k in keys), default=0)
        for p in corpus_pairs:
            yue, zh = p.get("yue", ""), p.get("zh", "")
            if not yue or not zh:
                continue
            subs = {yue[i:i + n] for n in range(1, min(max_key, len(yue)) + 1)
                    for i in range(len(yue) - n + 1)}
            for w in subs & keys:
                seen_n, co = stats.get(w, (0, Counter()))
                seen_n += 1
                for cand in votes[w]:
                    if cand in zh:
                        co[cand] += 1
                stats[w] = (seen_n, co)

    out: dict[str, str] = {}
    for yue_word, counter in votes.items():
        cands = {zh for zh in counter if zh != yue_word}
        if not cands:
            continue
        seen_n, co = stats.get(yue_word, (0, Counter()))
        if seen_n >= min_seen:
            best = max(cands, key=lambda c: co.get(c, 0))
            if co.get(best, 0) >= min_seen and co[best] / seen_n >= min_ratio:
                out[yue_word] = best
        else:
            # 语料证据不足：只信全票一致（yue.tsv 人工维护的单票可信）
            unanimous = [c for c in cands if counter[c] == sum(counter.values())]
            if len(unanimous) == 1:
                out[yue_word] = unanimous[0]
    return out


def segment(text: str, rdict: dict[str, str]) -> list[tuple[int, int, str]]:
    """最长匹配切分：返回 [(start, end, 词典译文)]，未匹配字符跳过（1 字步进）。"""
    max_key = max((len(k) for k in rdict), default=0)
    hits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        for length in range(min(max_key, len(text) - i), 0, -1):
            word = text[i:i + length]
            zh = rdict.get(word)
            if zh is not None:
                hits.append((i, i + length, zh))
                i += length
                break
        else:
            i += 1
    return hits


def dict_coverage(yue_text: str, candidate: str, rdict: dict[str, str]) -> float:
    """候选句对输入句词典词的覆盖度：命中译文（子串出现）的输入字符占比。

    无词典词命中时返回 0.0（不加不减，纯靠 beam 分）。
    """
    hits = segment(yue_text, rdict)
    if not hits:
        return 0.0
    total = sum(e - s for s, e, _ in hits)
    covered = sum(e - s for s, e, zh in hits if zh in candidate)
    return covered / total


def rerank(candidates: list[tuple[float, str]], yue_text: str,
           rdict: dict[str, str], lam: float) -> list[tuple[float, str]]:
    """按 归一beam分 + λ×覆盖度 重排候选；返回降序。candidates 本身应已降序。"""
    scored = [(score + lam * dict_coverage(yue_text, cand, rdict), cand)
              for score, cand in candidates]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return scored


def load_default_reverse_dict(corpus_pairs: list[dict] | None = None) -> dict[str, str]:
    """两个本地源合并构建，仓库内文件缺失的源自动跳过。

    corpus_pairs 传入平行语料时启用共现校验（推荐传 data/merged_pairs.json）。
    """
    pair_lists = []
    if os.path.exists(YUE_TSV):
        pair_lists.append(load_tsv_pairs(YUE_TSV))
    if os.path.exists(YUE_WORD_PAIRS):
        pair_lists.append(load_json_pairs(YUE_WORD_PAIRS))
    return build_reverse_dict(pair_lists, corpus_pairs=corpus_pairs)
