"""简拼：qingkx → 情况下、banbl → 办不了（首音节全拼 + 后续首字母）。"""
import os

import pytest

from ipinyin import config
from ipinyin.ime import ImeState
from ipinyin.lexicon import Lexicon
from ipinyin.lm import CharLM
from ipinyin.user_model import UserModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "data", "lexicon.tsv")
LM = os.path.join(ROOT, "data", "lm_char.marshal")


@pytest.fixture(scope="module")
def lexicon():
    if not os.path.exists(config.initials_path()):
        pytest.skip("缺 initials.marshal，先跑 tools/build_initials.py")
    lex = Lexicon.load(LEXICON).load_initials(config.initials_path())
    lex.warm_options()
    return lex


@pytest.fixture()
def ime(lexicon, tmp_path):
    state = ImeState(lexicon, user_model=UserModel(str(tmp_path / "m.tsv")))
    if os.path.exists(LM):
        state.attach_lm(CharLM.load(LM))
    return state


def type_text(ime, text):
    for char in text:
        ime.insert(char)
    return ime.view().words


def test_head_full_abbreviation_hits_phrase(ime):
    """qingkx = qing + k + x：情况 + 下 应该排在前面。"""
    words = type_text(ime, "qingkx")
    assert "情况下" in words[:3], words


def test_abbreviation_can_be_learned(ime):
    """banbl 第一次可能不是首选，选中一次后必须稳定第一。"""
    words = type_text(ime, "banbl")
    assert "办不了" in words, words
    ime.select_page_index(words.index("办不了"))
    for key in ime.learn_keys(ime.last_commit_typed, ime.last_commit_key,
                              ime.last_commit_full):
        ime.user_model.learn(key, "办不了", 3)
    ime.clear()
    assert type_text(ime, "banbl")[0] == "办不了"


def test_normal_pinyin_is_not_touched(ime):
    """普通整拼不能被简拼展开污染。"""
    assert type_text(ime, "nihao")[0] == "你好"


def test_partial_typing_does_not_expand(ime):
    """nih 这种打一半的，交给前缀补全，不触发展开（避免每键多花几毫秒）。"""
    assert ime._abbrev_segments("nih") is None
    assert ime._abbrev_segments("qingkx") is not None
    assert ime._abbrev_segments("qingkuangxia") is None   # 整拼不展开


def test_mixed_abbreviation_three_segments(ime):
    """zenmhuituran / huichux：中间夹一个首字母的混合式，不能退化成原文。

    回归：曾经因为「英文直出」的判定过宽，这类输入被当成英文直接上屏，
    打出来的就是拼音字母本身。
    """
    for text, want in (("zenmhuituran", "怎么会突然"), ("huichux", "会出现")):
        ime.clear()
        for char in text:
            ime.insert(char)
        assert ime.view().words[0] == want


def test_normal_prefix_not_broken(ime):
    """nih 这种正常打一半的，仍然走前缀补全。"""
    for char in "nih":
        ime.insert(char)
    assert ime.view().words[0] == "你好"
