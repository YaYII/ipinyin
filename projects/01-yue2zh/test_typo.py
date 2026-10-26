"""typo.py 同音错字模块单测：纯函数，不依赖 torch；data/yue.marshal 缺失时跳过集成项。

跑法：/path/to/python -m pytest projects/01-yue2zh/test_typo.py
"""
from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import typo

YUE_MARSHAL = os.path.join(ROOT, "data", "yue.marshal")


def test_strip_tone():
    assert typo.strip_tone("lei5") == "lei"
    assert typo.strip_tone("ngo4") == "ngo"
    assert typo.strip_tone("aa") == "aa"          # 无声调数字
    assert typo.strip_tone("ng") == "ng"


def test_build_homophones_groups_by_syllable():
    # aa1/aa2 同音（去调后同声韵），bb3 无伴
    chars = {"甲": "aa1", "乙": "aa2", "丙": "bb3"}
    h = typo.build_homophones(chars)
    assert h == {"甲": ("乙",), "乙": ("甲",)}


def test_build_homophones_whitelist_binds_keys_and_candidates():
    chars = {"甲": "aa1", "乙": "aa2", "丙": "aa3", "丁": "bb1"}
    h = typo.build_homophones(chars, allowed={"甲", "丁"})
    # 键字丙不在白名单 → 无条目；候选乙不在白名单 → 甲也无可换字
    assert h == {}
    h2 = typo.build_homophones(chars, allowed={"甲", "乙", "丁"})
    assert h2 == {"甲": ("乙",), "乙": ("甲",)}


def test_inject_typos_zero_prob_unchanged():
    h = typo.build_homophones({"甲": "aa1", "乙": "aa2"})
    s = "甲甲乙乙，丙丙。"
    assert typo.inject_typos(s, h, random.Random(0), p=0.0) == s


def test_inject_typos_deterministic_same_seed():
    h = typo.build_homophones({"甲": "aa1", "乙": "aa2", "丙": "aa3"})
    s = "甲乙丙甲乙丙甲乙丙"
    a = typo.inject_typos(s, h, random.Random(42), p=0.5)
    b = typo.inject_typos(s, h, random.Random(42), p=0.5)
    assert a == b


def test_inject_typos_single_candidate_and_full_prob():
    # 只有一个同音候选 + p=1 → 每个表内汉字必然被换；标点/非汉字不动
    h = {"係": ("喺",)}
    s = "係真係，OK 係"
    out = typo.inject_typos(s, h, random.Random(7), p=1.0)
    assert out == "喺真喺，OK 喺"


def test_inject_typos_only_touches_table_chars():
    h = {"係": ("喺",)}
    s = "粵語好難"
    assert typo.inject_typos(s, h, random.Random(7), p=1.0) == s


def test_real_yue_marshal_loads():
    if not os.path.exists(YUE_MARSHAL):
        return  # 数据文件不在就跳过（仓库内应恒在）
    chars = typo.load_yue_chars(YUE_MARSHAL)
    assert len(chars) > 20000
    assert "係" in chars
    allowed = {"係", "喺", "系"}
    h = typo.build_homophones(chars, allowed=allowed)
    # 係/喺/系 同音（hai），候选必须落在白名单内且必含 喺
    cands = h.get("係", ())
    assert "喺" in cands
    assert set(cands) <= allowed
