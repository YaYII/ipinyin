#!/usr/bin/env python3
"""G2P 管线的确定性单测：数字转换、粤拼切分、音素拆分、端到端。

全部走规则版转换（不加载神经模型），只依赖仓库内的数据文件，
同一输入永远同一输出。运行：

  /path/to/python -m pytest projects/03-tts-g2p/test_g2p.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import g2p as g2p_mod                       # noqa: E402
from g2p import G2P, normalize, int_to_cn, digits_to_cn  # noqa: E402
from phonemes import (FINAL_SET, INITIALS, TONES, build_inventory,  # noqa: E402
                      split_syllable, syllable_to_tokens)

import pytest                               # noqa: E402


@pytest.fixture(scope="module")
def engine() -> G2P:
    return G2P(neural=False)


# ---- 数字 → 数量读法 --------------------------------------------------------
@pytest.mark.parametrize("num,expect", [
    (0, "零"), (4, "四"), (9, "九"),
    (10, "十"), (11, "十一"), (19, "十九"),
    (20, "二十"), (25, "二十五"), (99, "九十九"),
    (100, "一百"), (105, "一百零五"), (110, "一百一十"),
    (123, "一百二十三"), (200, "二百"), (909, "九百零九"), (999, "九百九十九"),
    (1000, "一千"), (1005, "一千零五"), (1010, "一千零一十"),
    (1100, "一千一百"), (2000, "二千"), (9999, "九千九百九十九"),
    (10000, "一万"), (10010, "一万零一十"), (12345, "一万二千三百四十五"),
    (105000, "十万五千"), (123456, "十二万三千四百五十六"),
    (1000000, "一百万"), (10000000, "一千万"),
    (100000000, "一亿"), (100000010, "一亿零一十"),
    (123456789, "一亿二千三百四十五万六千七百八十九"),
])
def test_int_to_cn(num: int, expect: str):
    assert int_to_cn(num) == expect


def test_digits_to_cn_year():
    assert digits_to_cn("2026") == "二零二六"
    assert digits_to_cn("1997") == "一九九七"


# ---- normalize：数字形态 + 标点 ---------------------------------------------
def test_normalize_year():
    assert normalize("2026年") == "二零二六年"
    assert normalize("1997年回归。") == "一九九七年回归."


def test_normalize_month_day_clock():
    assert normalize("10月25日") == "十月二十五日"
    assert normalize("9点30分") == "九点三十分"
    assert normalize("7点半") == "七点半"


def test_normalize_quantity():
    assert normalize("123") == "一百二十三"
    assert normalize("1,234元") == "一千二百三十四元"
    assert normalize("每15分钟一班") == "每十五分钟一班"
    assert normalize("票价88块") == "票价八十八块"


def test_normalize_two_before_measure():
    # 单个二 + 量词 → 两；进位组合（十二）与序数（第二）不动
    assert normalize("2个苹果") == "两个苹果"
    assert normalize("2次") == "两次"
    assert normalize("12个") == "十二个"
    assert normalize("第二年") == "第二年"
    assert normalize("老二") == "老二"


def test_normalize_leaves_todo_forms_alone():
    # TODO 形态：小数 / 百分数 / 电话号码，保持原样不猜
    assert normalize("3.5公斤") == "3.5公斤"
    assert normalize("50%") == "50%"
    assert normalize("13800138000") == "13800138000"


def test_normalize_punct():
    assert normalize("你好，世界。") == "你好,世界."
    assert normalize("什么？！") == "什么?!"
    assert normalize("他说「你好」。") == "他说你好."
    assert normalize("；：、") == ",,,"


# ---- 粤拼切分：词级最长匹配 / 字级兜底 / 缺字报告 -----------------------------
def test_segment_word_first(engine: G2P):
    tokens = engine.segment("我哋去公園")
    words = [t.surface for t in tokens if t.kind == "word"]
    assert "我哋" in words and "公園" in words


def test_segment_char_fallback(engine: G2P):
    tokens = engine.segment("傘")           # 傘 不在词表，走字表
    assert tokens[0].kind == "char"
    assert tokens[0].syllables == ["saan3"]


def test_segment_missing_reported(engine: G2P):
    res = engine("乄係乜嘢？")
    assert "乄" in res.missing
    kinds = [t.kind for t in res.tokens]
    assert "missing" in kinds


def test_segment_ext_b_chars(engine: G2P):
    # 𠝹 是扩展B汉字，粤拼表里有（表带 1981 个 Ext-B 条目）
    res = engine("𠝹開個箱。")
    assert res.missing == []
    assert "gaai3" in res.jyutping          # 按表注音（表值如此）


def test_segment_bad_syllable_reported(engine: G2P):
    # 円 的表值 jen1 不是合法粤拼（应为 jyun1），要被点名而不是悄悄吞掉
    res = engine("日円")
    assert "日円:jen1" in res.missing
    assert "jen1" not in res.phonemes


def test_segment_punct_and_latin(engine: G2P):
    tokens = engine.segment("你好,OK.")
    kinds = [t.kind for t in tokens]
    assert kinds.count("punct") == 2
    assert any(t.kind == "latin" and t.surface == "OK" for t in tokens)


# ---- 音素拆分 ---------------------------------------------------------------
@pytest.mark.parametrize("syl,expect", [
    ("aa1", ("", "aa", "1")),
    ("ngo5", ("ng", "o", "5")),
    ("ng5", ("", "ng", "5")),
    ("m4", ("", "m", "4")),
    ("gwong2", ("gw", "ong", "2")),
    ("kwaak5", ("kw", "aak", "5")),
    ("syu1", ("s", "yu", "1")),
    ("jyu4", ("j", "yu", "4")),
    ("oeng6", ("", "oeng", "6")),
])
def test_split_syllable(syl: str, expect: tuple):
    assert split_syllable(syl) == expect


@pytest.mark.parametrize("syl", ["jen1", "xyz9", "abc1", "aa7", "", "ngo"])
def test_split_syllable_rejects_bad(syl: str):
    assert split_syllable(syl) is None


def test_syllable_to_tokens():
    assert syllable_to_tokens("ngo5") == ["ŋ", "ɔː", "T5"]
    assert syllable_to_tokens("aa1") == ["aː", "T1"]      # 零声母不发空 token
    assert syllable_to_tokens("gwong2") == ["kʷ", "ɔːŋ", "T2"]
    assert syllable_to_tokens("jen1") is None


# ---- 音素清单 ---------------------------------------------------------------
def test_inventory_shape():
    inv = build_inventory()
    assert inv["initials"] == sorted(INITIALS) and len(inv["initials"]) == 19
    assert inv["finals"] == sorted(FINAL_SET) and len(inv["finals"]) == 59
    assert inv["tones"] == list(TONES) == ["1", "2", "3", "4", "5", "6"]
    # 数据里唯一的坏音节：円→jen1（两词：日円/日本円）
    assert inv["bad_syllables"] == ["jen1"]
    assert set(inv["ipa_initial"]) == set(INITIALS)


# ---- 端到端 -----------------------------------------------------------------
def test_e2e_jyutping(engine: G2P):
    res = engine("我不知道他在说什么。")
    assert res.yue == "我唔知佢喺講乜嘢."
    assert res.jyutping == ["ngo5", "m4", "zi1", "heoi5", "hai2",
                            "gong3", "mat1", "je5"]
    assert res.missing == []


def test_e2e_numbers(engine: G2P):
    res = engine("2026年10月25日是星期日。")
    assert res.yue.startswith("二零二六年十月二十五日係")
    assert res.jyutping[:10] == ["ji6", "lin4", "ji6", "liu1", "nin4",
                                 "sap6", "jyut6", "ji6", "sap6", "ng5"]
    assert res.phonemes[:6] == ["j", "iː", "T6", "l", "iːn", "T4"]


def test_e2e_deterministic(engine: G2P):
    text = "今天天气很好，我们去公园散步。机场快线每15分钟一班，票价120元。"
    a, b = engine(text), engine(text)
    assert (a.yue, a.jyutping, a.phonemes, a.missing) == \
           (b.yue, b.jyutping, b.phonemes, b.missing)


def test_module_count_sanity():
    # 防呆：引擎里词表/字表必须已装载
    conv = g2p_mod.G2P().conv
    assert len(conv._words) > 100000
    assert len(conv._chars) > 20000
