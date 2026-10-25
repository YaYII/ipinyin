#!/usr/bin/env python3
"""convert.py 纯逻辑单测：清洗、分句、编排。不加载模型、不依赖 torch。

跑法：/path/to/python -m pytest projects/01-yue2zh/test_convert.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert as cv


def test_clean_text_unifies_newlines():
    assert cv.clean_text("a\r\nb\rc") == "a\nb\nc"


def test_clean_text_strips_zero_width():
    assert cv.clean_text("你\u200b好\u200d，\ufeff世界") == "你好，世界"


def test_segment_sentences_basic():
    text = "今日唔得閒。聽日先講！\n第二段冇標點"
    assert cv.segment_sentences(text) == ["今日唔得閒。", "聽日先講！", "第二段冇標點"]


def test_segment_sentences_punctuation_follows_sentence():
    # 连续终止标点整块跟随前句；每个标点块都是一次分句边界
    assert cv.segment_sentences("真係㗎？！唔係啩……") == ["真係㗎？！", "唔係啩……"]


def test_segment_sentences_drops_empty_and_punct_only():
    assert cv.segment_sentences("。\n！！\n  \n有嘢。") == ["有嘢。"]


def test_segment_sentences_keeps_commas_inside():
    # 逗号、顿号不是终止符，整句保留
    assert cv.segment_sentences("你，我，佢。") == ["你，我，佢。"]


def test_segment_sentences_english_words_survive():
    assert cv.segment_sentences("我用OK咗啦。") == ["我用OK咗啦。"]


def test_segment_sentences_empty_input():
    assert cv.segment_sentences("") == []
    assert cv.segment_sentences("\n\n") == []


def test_convert_text_orders_and_maps():
    def fake_decode(sents):
        return [s.replace("唔", "不") for s in sents]

    out = cv.convert_text("唔好咁啦。\n唔該。", fake_decode)
    assert out == ["不好咁啦。", "不該。"]


def test_convert_text_empty_returns_empty():
    assert cv.convert_text("   \n", lambda s: s) == []
