"""潮州话（汕头音）第五列：字音查询 + Tab 档位。"""
import pytest

from ipinyin.chaoshan import ChaoshanConverter
from ipinyin.ime import ImeState
from ipinyin.lexicon import Lexicon


@pytest.fixture(scope="module")
def chao():
    converter = ChaoshanConverter.load("data/chaoshan.marshal")
    if converter is None:
        pytest.skip("缺 data/chaoshan.marshal（先跑 tools/build_chaoshan.py）")
    return converter


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load("data/lexicon.tsv")


def test_config_points_at_installed_table():
    from ipinyin import config

    path = config.chaoshan_path()
    assert path.endswith("chaoshan.marshal")
    assert __import__("os").path.exists(path), f"{path} 不存在（跑 tools/build_chaoshan.py）"


def test_loads_shantou_table(chao):
    assert chao.accent == "st"
    assert len(chao) > 10_000


def test_word_level_reading_wins(chao):
    # 词表优先：整词读音比逐字拼准
    assert chao.pengim("你好") == "le2 ho2"
    assert chao.pengim("汕头") == "suan1 tao5"


def test_empty_and_unmapped(chao):
    assert chao.pengim("") == ""
    # 字表里连字母都有读音（A → êi1），所以用不可能收录的字符测跳过
    assert chao.pengim("\U000E0001") == ""


def test_mode_cycle_reaches_chao(lexicon, chao):
    ime = ImeState(lexicon, chao=chao)
    assert ime.cycle_mode() == "trad"
    assert ime.cycle_mode() == "yue"
    assert ime.cycle_mode() == "chao"
    assert ime.cycle_mode() == "en"
    assert ime.cycle_mode() == "simp"


def test_chao_mode_outputs_written_form(lexicon, chao):
    """潮汕档上屏的是文字（像粤语），不是拼音。"""
    ime = ImeState(lexicon, chao=chao)
    ime.mode = "chao"
    for char in "chifan":
        ime.insert(char)
    assert ime.view().words[0] == "食饭"


def test_rewrite_maps_mandarin_words(chao):
    assert chao.rewrite("我们") == "阮"
    assert chao.rewrite("什么") == "乜个"
    assert chao.rewrite("吃饭") == "食饭"
    assert chao.rewrite("不要") == "嫑"
    assert chao.rewrite("你好") == "你好"      # 没有映射就原样保留


def test_chao_column_shows_written_form(lexicon, chao):
    ime = ImeState(lexicon, chao=chao)
    for char in "chifan":
        ime.insert(char)
    assert "食饭" in ime.view().candidates[0]


def test_chao_mode_shows_pengim(lexicon, chao):
    ime = ImeState(lexicon, chao=chao)
    ime.mode = "chao"
    for char in "nihao":
        ime.insert(char)
    assert "le2 ho2" in ime.view().candidates[0]
