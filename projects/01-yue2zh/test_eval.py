"""eval.py 纯函数单测：chrF 指标的手工可口算样例（不加载模型、不依赖 torch）。

跑法：/path/to/python -m pytest projects/01-yue2zh/test_eval.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval as ev


def test_chrf_identical_is_one():
    assert ev.chrf_score(["你好世界"], ["你好世界"]) == 1.0
    assert ev.chrf_score(["ab", "cd"], ["ab", "cd"]) == 1.0


def test_chrf_no_overlap_is_zero():
    assert ev.chrf_score([""], ["你好"]) == 0.0
    assert ev.chrf_score(["甲乙"], ["丙丁"]) == 0.0


def test_chrf_hand_computed_partial_overlap():
    # hyp="abc" vs ref="abd"，max_n=2：
    #   n=1: P=R=2/3 → F2 = 5·(2/3)(2/3) / (4·(2/3)+(2/3)) = 2/3
    #   n=2: {ab,bc} vs {ab,bd} 命中 1 → P=R=1/2 → F2 = 1/2
    #   平均 = (2/3 + 1/2)/2 = 7/12
    got = ev.chrf_score(["abc"], ["abd"], max_n=2)
    assert abs(got - 7 / 12) < 1e-9


def test_chrf_ignores_whitespace():
    assert ev.chrf_score(["a b c"], ["abc"]) == 1.0
    assert ev.chrf_score(["你 好"], ["你好"]) == 1.0


def test_chrf_perfect_shorter_than_ngram_order():
    # 句长 2 < max_n=6：3..6 阶双侧皆空 → 跳过，只用 1..2 阶 → 仍满分
    assert ev.chrf_score(["ab"], ["ab"], max_n=6) == 1.0


def test_chrf_corpus_aggregation():
    # 两对：一对全对、一对全错 → 各阶 P=R=1/2 → F=1/2
    assert ev.chrf_score(["你好", "甲甲"], ["你好", "乙乙"]) == 0.5


def test_bucket_of():
    assert ev.bucket_of(1) == "≤8"
    assert ev.bucket_of(8) == "≤8"
    assert ev.bucket_of(9) == "9-15"
    assert ev.bucket_of(15) == "9-15"
    assert ev.bucket_of(16) == "16+"
    assert ev.bucket_of(38) == "16+"


def test_char_overlap_position_based():
    assert ev.char_overlap("abcd", "abce") == 0.75
    assert ev.char_overlap("", "abc") == 0.0
