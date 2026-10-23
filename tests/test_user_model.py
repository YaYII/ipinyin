"""个人词频：学到的词要真的排到前面，而且要能存下来。"""
import pytest

from ipinyin.ime import ImeState
from ipinyin.lexicon import Lexicon
from ipinyin.user_model import UserModel


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load("data/lexicon.tsv")


def test_learn_moves_word_to_front(lexicon, tmp_path):
    model = UserModel(str(tmp_path / "model.tsv"))
    ime = ImeState(lexicon, user_model=model)
    ime.insert("s")
    ime.insert("h")
    ime.insert("i")
    default_first = ime.view().candidates[0]
    second = ime.view().candidates[1]
    assert default_first != second

    model.learn("shi", second)
    ime.clear()
    for char in "shi":
        ime.insert(char)
    assert ime.view().candidates[0] == second


def test_learning_persists_across_restart(tmp_path):
    path = str(tmp_path / "model.tsv")
    UserModel(path).learn("nihao", "拟好")
    reopened = UserModel(path)
    assert "拟好" in reopened.preferred("nihao")
    assert reopened.picks("nihao", "拟好") == 1


def test_seed_is_loaded(tmp_path):
    seed = tmp_path / "seed.tsv"
    seed.write_text("shurufa\t输入法\t120\n", encoding="utf-8")
    model = UserModel(str(tmp_path / "model.tsv"), str(seed))
    assert model.preferred("shurufa") == ["输入法"]


def test_learning_ignores_raw_pinyin(tmp_path):
    model = UserModel(str(tmp_path / "model.tsv"))
    model.learn("nihaoma", "nihaoma")  # 回车原文上屏，不该学
    assert model.preferred("nihaoma") == []


def test_compact_rewrites_without_loss(tmp_path):
    path = tmp_path / "model.tsv"
    model = UserModel(str(path))
    model.learn("shi", "事")
    model.learn("shi", "事")
    model.compact()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert lines == ["shi\t事\t2\t1"]


def test_bigram_puts_expected_next_word_first(lexicon, tmp_path):
    model = UserModel(str(tmp_path / "model.tsv"), bigram_path=str(tmp_path / "bigram.tsv"))
    model.learn_bigram("我", "呢")
    ime = ImeState(lexicon, user_model=model)
    ime.context = "我"
    for char in "ni":
        ime.insert(char)
    assert ime.view().candidates[0] == "呢"


def test_bigram_survives_restart(tmp_path):
    path = str(tmp_path / "bigram.tsv")
    UserModel(str(tmp_path / "model.tsv"), bigram_path=path).learn_bigram("我", "喜欢")
    reopened = UserModel(str(tmp_path / "model.tsv"), bigram_path=path)
    assert "喜欢" in reopened.after("我")


def test_fuzzy_pinyin_rescues_fast_typo(lexicon):
    ime = ImeState(lexicon)
    for char in "pingying":  # ing/ying 打混了，拼音打成 pingying
        ime.insert(char)
    assert ime.view().candidates[0] == "拼音"
