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


def test_learned_word_appears_even_if_not_in_candidates(lexicon, tmp_path):
    """学过的词即使词库/组合都拼不出来，也要补进候选（饿了吗 的真实 bug）。"""
    from ipinyin.ime import ImeState
    from ipinyin.user_model import UserModel

    model = UserModel(str(tmp_path / "m.tsv"), None, str(tmp_path / "b.tsv"), None,
                      validator=lambda key, word: True)
    ime = ImeState(lexicon, user_model=model)
    for char in "elema":
        ime.insert(char)
    assert "饿了吗" not in ime.view().words[:3]
    model.learn("elema", "饿了吗")
    ime.clear()
    for char in "elema":
        ime.insert(char)
    assert ime.view().words[0] == "饿了吗"


def test_exposures_are_not_double_counted(tmp_path):
    """曝光是累计值：文件里多行记录不能相加（旧版每重启一次翻一倍）。"""
    path = tmp_path / "model.tsv"
    path.write_text(
        "# 拼音键\t词\t选中次数\t曝光次数\n"
        "de\t的\t10\t500\n"      # 一次 compact 的快照
        "de\t的\t1\t0\n"         # 之后的增量行（曝光写 0）
        "de\t的\t1\t0\n",
        encoding="utf-8",
    )
    model = UserModel(str(path))
    assert model.picks("de", "的") == 12          # 选中次数按增量累加
    assert model.exposures("de", "的") == 500     # 曝光取最大值，不相加
    assert 0.0 < model.hit_rate("de", "的") <= 1.0


def test_absurd_exposures_are_repaired(tmp_path):
    """历史脏数据（2.5e15 那种）加载时自愈，命中率不能恒为 0。"""
    path = tmp_path / "model.tsv"
    path.write_text(
        "# 拼音键\t词\t选中次数\t曝光次数\n"
        "de\t的\t6461\t2548047116115232\n",
        encoding="utf-8",
    )
    model = UserModel(str(path))
    exposures = model.exposures("de", "的")
    assert exposures < 6461 * 1000, "异常曝光应被重置"
    assert model.hit_rate("de", "的") > 0.01, "命中率应恢复成可用值"


def test_learn_keys_cover_abbreviation(lexicon):
    """上屏要记三个键：敲的、完整的、它的简拼。"""
    ime = ImeState(lexicon)
    assert ime.learn_keys("jianp", "jianp", "jianpin") == ["jianp", "jp"]
    assert ime.learn_keys("jianpin", "jianpin", "jianpin") == ["jianpin", "jp"]
    # 单音节不生成简拼，避免 d → 的 这种噪音
    assert ime.learn_keys("de", "de", "de") == ["de"]


def test_initials_hit_after_picking_full_pinyin(tmp_path):
    """打全拼选过「简拼」，下次打 jp 必须能命中——这就是「简拼命不中」的修复。"""
    from ipinyin import config

    lex = Lexicon.load("data/lexicon.tsv").load_initials(config.initials_path())
    model = UserModel(str(tmp_path / "m.tsv"))
    ime = ImeState(lex, user_model=model)

    for ch in "jianpin":
        ime.insert(ch)
    index = ime.view().words.index("简拼")
    ime.select_page_index(index)
    for key in ime.learn_keys(ime.last_commit_typed, ime.last_commit_key,
                              ime.last_commit_full):
        model.learn(key, "简拼", 3)

    ime.clear()
    for ch in "jp":
        ime.insert(ch)
    assert ime.view().words[0] == "简拼"
