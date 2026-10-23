"""词库 / 切句 / 状态机 的行为测试。用真实词库，所以这些断言就是产品行为。"""
import os

import pytest

from ipinyin.ime import ImeState
from ipinyin.lexicon import Lexicon
from ipinyin.segmenter import best_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "data", "lexicon.tsv")


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load(LEXICON)


@pytest.fixture()
def ime(lexicon):
    return ImeState(lexicon)


def type_text(ime, text):
    for char in text:
        ime.insert(char)
    return ime.view()


def test_lexicon_has_common_words(lexicon):
    assert lexicon.entries("nihao")[0][0] == "你好"
    assert lexicon.entries("zhongguo")[0][0] == "中国"
    assert lexicon.entries("wo")[0][0] == "我"


def test_prefix_expands_incomplete_syllable(lexicon):
    words = [word for word, _ in lexicon.prefix("nih")]
    assert "你好" in words


def test_segment_full_sentence(lexicon):
    assert best_path(lexicon, "woxihuanni") == ["我", "喜欢", "你"]
    assert "".join(best_path(lexicon, "jintiantianqi")) == "今天天气"


def test_segment_rejects_garbage(lexicon):
    assert best_path(lexicon, "zzzz") is None


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("nihao", "你好"),
        ("woxihuanni", "我喜欢你"),
        ("zhongguoren", "中国人"),
        ("shubiao", "鼠标"),
    ],
)
def test_first_candidate(ime, typed, expected):
    view = type_text(ime, typed)
    assert view.candidates[0] == expected


def test_commit_first_candidate(ime):
    type_text(ime, "nihao")
    assert ime.commit_current() == "你好"
    assert ime.buffer == ""


def test_commit_raw_keeps_letters(ime):
    type_text(ime, "abc")
    assert ime.commit_raw() == "abc"


def test_digit_selects_second_candidate(ime):
    type_text(ime, "shi")
    second = ime.view().candidates[1]
    assert ime.select(2) == second


def test_backspace_shortens_buffer(ime):
    type_text(ime, "nih")
    assert ime.backspace() is True
    assert ime.buffer == "ni"
    assert ime.view().candidates[0] == "你"


def test_escape_clears(ime):
    type_text(ime, "nihao")
    ime.clear()
    assert ime.view().visible is False


def test_paging_moves_between_pages(ime):
    type_text(ime, "shi")
    assert ime.view().page_count > 1
    assert ime.page_by(1) is True
    assert ime.view().page == 1
    assert ime.page_by(-1) is True
    assert ime.view().page == 0


def test_move_beyond_last_candidate_stops(ime):
    type_text(ime, "shi")
    view = ime.view()
    for _ in range(view.page_count * ime.page_size):
        if not ime.move(1):
            break
    assert ime.view().highlighted >= 0


def test_empty_buffer_has_no_view(ime):
    assert ime.view().visible is False
    assert ime.commit_current() is None


def test_bracket_paging_selects_from_second_page(ime):
    type_text(ime, "shi")
    first_page = list(ime.view().candidates)
    assert ime.page_by(1) is True
    second_page = ime.view().candidates
    assert second_page and second_page != first_page
    picked = ime.select(1)          # 第 2 页第 1 个
    assert picked == second_page[0]


def test_left_right_moves_pinyin_segment(ime):
    type_text(ime, "nihaoma")
    assert ime.view().candidates[0] == "你号码"
    assert ime.move_segment(-1) is True          # 左：切到 nihao
    assert ime.active_text() == "nihao"
    assert ime.view().candidates[0] == "你好"
    assert ime.move_segment(-1) is True          # 再左：切到 ni
    assert ime.active_text() == "ni"
    assert ime.view().candidates[0] == "你"
    assert ime.move_segment(1) is True           # 右：回到 nihao
    assert ime.active_text() == "nihao"


def test_commit_segment_keeps_remaining_pinyin(ime):
    type_text(ime, "nihaoma")
    ime.move_segment(-1)                          # 只转 nihao
    assert ime.commit_current() == "你好"
    assert ime.buffer == "ma"
    assert ime.view().candidates[0] == "吗"


def test_tab_cycles_simplified_traditional_cantonese(lexicon):
    from ipinyin.t2s import Converter
    from ipinyin.yue import YueConverter

    ime = ImeState(
        lexicon,
        converter=Converter.load("data/s2t.marshal"),
        yue=YueConverter.load_dict("data/yue.marshal", YueConverter.load("data/yue.tsv")._map),
    )
    for char in "women":
        ime.insert(char)
    assert ime.view().candidates[0].startswith("我们")
    assert ime.cycle_mode() == "trad"
    assert ime.view().candidates[0].startswith("我們")
    assert ime.cycle_mode() == "yue"
    first = ime.view().candidates[0]
    assert first.startswith("我哋")
    assert "ngo5 dei6" in first          # 粤拼注音
    assert ime.select(1) == "我哋"
    assert ime.cycle_mode() == "simp"
