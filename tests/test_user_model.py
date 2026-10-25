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
    default_first = ime.view().words[0]
    second = ime.view().words[1]
    assert default_first != second

    model.learn("shi", second)
    ime.clear()
    for char in "shi":
        ime.insert(char)
    assert ime.view().words[0] == second


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
    assert ime.view().words[0] == "呢"


def test_bigram_survives_restart(tmp_path):
    path = str(tmp_path / "bigram.tsv")
    UserModel(str(tmp_path / "model.tsv"), bigram_path=path).learn_bigram("我", "喜欢")
    reopened = UserModel(str(tmp_path / "model.tsv"), bigram_path=path)
    assert "喜欢" in reopened.after("我")


def test_fuzzy_pinyin_rescues_fast_typo(lexicon):
    ime = ImeState(lexicon)
    for char in "pingying":  # ing/ying 打混了，拼音打成 pingying
        ime.insert(char)
    assert ime.view().words[0] == "拼音"


def test_picked_word_survives_fuzzy(lexicon, tmp_path):
    """用户选过的词，不能被模糊音挤下去（才对 vs 柴堆 的真实 bug）。"""
    from ipinyin.ime import ImeState
    from ipinyin.user_model import UserModel

    model = UserModel(str(tmp_path / "m.tsv"), None, str(tmp_path / "b.tsv"), None,
                      validator=lambda key, word: True)
    ime = ImeState(lexicon, user_model=model)
    for char in "caidui":
        ime.insert(char)
    target = "才对"
    index = ime.view().words.index(target)
    ime.select(index + 1)
    model.learn("caidui", target)      # 引擎在上屏时会学习，这里直接模拟
    ime.clear()
    for char in "caidui":
        ime.insert(char)
    assert ime.view().words[0] == target


def test_observe_does_not_grow_on_unknown_words(tmp_path):
    """曝光只记已知的词：从未选中的候选不建 cell（每次按键都调 observe，
    之前会给所有候选建记录，评测全量跑曾因此 OOM）。"""
    from ipinyin.user_model import UserModel

    model = UserModel(str(tmp_path / "m.tsv"), None, None, None,
                      validator=lambda key, word: True)
    model.learn("nihao", "你好")
    model.observe("nihao", ["你好", "你号", "尼豪"])
    assert len(model) == 1                    # 只有学过的词有记录
    assert model.exposures("nihao", "你好") == 2   # learn 置 1 + 展示 1 次
    model.observe("nihao", ["你号"])          # 键存在但词未知 → 什么都不记
    assert len(model) == 1
    assert model.exposures("nihao", "你好") == 2
    model.observe("zenme", ["怎么"])           # 键都不存在 → 不建键
    assert len(model) == 1
