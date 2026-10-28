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
    assert view.words[0] == expected


def test_commit_first_candidate(ime):
    type_text(ime, "nihao")
    assert ime.commit_current() == "你好"
    assert ime.buffer == ""


def test_commit_raw_keeps_letters(ime):
    type_text(ime, "abc")
    assert ime.commit_raw() == "abc"


def test_digit_selects_second_candidate(ime):
    type_text(ime, "shi")
    second = ime.view().words[1]
    assert ime.select(2) == second


def test_backspace_shortens_buffer(ime):
    type_text(ime, "nih")
    assert ime.backspace() is True
    assert ime.buffer == "ni"
    assert ime.view().words[0] == "你"


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
    second_page = ime.view().words
    assert second_page and second_page != first_page
    picked = ime.select(1)          # 第 2 页第 1 个
    assert picked == second_page[0]


def test_left_right_moves_pinyin_segment(ime):
    type_text(ime, "nihaoma")
    assert ime.view().words[0] == "你号码"
    assert ime.move_segment(-1) is True          # 左：切到 nihao
    assert ime.active_text() == "nihao"
    assert ime.view().words[0] == "你好"
    assert ime.move_segment(-1) is True          # 再左：切到 ni
    assert ime.active_text() == "ni"
    assert ime.view().words[0] == "你"
    assert ime.move_segment(1) is True           # 右：回到 nihao
    assert ime.active_text() == "nihao"


def test_commit_segment_keeps_remaining_pinyin(ime):
    type_text(ime, "nihaoma")
    ime.move_segment(-1)                          # 只转 nihao
    assert ime.commit_current() == "你好"
    assert ime.buffer == "ma"
    assert ime.view().words[0] == "吗"


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
    assert "我們" in ime.view().candidates[0]
    assert ime.cycle_mode() == "yue"
    first = ime.view().candidates[0]
    assert "我哋" in first                # 四列里始终有粤语
    assert "ngo5 dei6" in first           # 粤拼注音
    assert ime.select(1) == "我哋"
    assert ime.cycle_mode() == "simp"


def test_best_paths_returns_ordered_distinct_paths(lexicon):
    """n-best 是上下文重排的前置：多于一条、按分降序、第一条必须等于旧的最优解。"""
    from ipinyin.segmenter import best_path, best_paths

    text = "woxihuannizhongguo"
    paths = best_paths(lexicon, text, k=3)
    assert len(paths) >= 2, "长拼音应该有多个切分"
    assert paths[0] == best_path(lexicon, text)
    joined = ["".join(p) for p in paths]
    assert len(set(joined)) == len(joined), "同一结果不应重复出现"
    assert all("".join(p) for p in paths)


def test_best_paths_handles_short_and_unknown(lexicon):
    from ipinyin.segmenter import best_paths

    assert best_paths(lexicon, "", k=3) == []
    assert best_paths(lexicon, "zzzz", k=3) == []
    single = best_paths(lexicon, "women", k=3)
    assert single and single[0] == ["我们"]


def test_sentence_candidates_expose_alternatives(lexicon):
    """整句候选要有备选：打错一个字时用户能直接选第二条，不用重打。"""
    ime = ImeState(lexicon)
    for char in "woxihuannizhongguo":
        ime.insert(char)
    words = ime.view().words
    assert len(words) >= 2
    assert len(set(words)) == len(words)


def test_attach_lm_reranks_sentence_candidates(lexicon):
    """挂上语言模型后，整句候选顺序由「词库先验 + λ·上下文分」决定。

    这里用桩模型验证接线（真实 LM 的效果见 docs/nn-chinese.md 的 +6.3pt）。
    """
    from ipinyin.segmenter import best_paths_scored

    text = "woxihuannizhongguo"
    paths = best_paths_scored(lexicon, text, k=3)
    assert len(paths) >= 2
    target = "".join(paths[-1][1])          # 先验排最后的那条整句

    ime = ImeState(lexicon)
    for char in text:
        ime.insert(char)
    before = ime.view().words
    assert target in before, "目标整句应该出现在候选里"

    class StubLM:
        def logp(self, self_text):
            return 100.0 if self_text == target else 0.0

    ime.attach_lm(StubLM(), weight=1.0)
    after = ime.view().words
    # 候选里除了整句还有前缀词，所以看「位置前移」而不是「必须第一」
    assert after.index(target) < before.index(target)


def test_t2s_converter_roundtrip():
    """简↔繁只是显示层映射，必须可逆：繁→简不能产出异体字（㓰 那类）。"""
    from ipinyin.t2s import Converter

    conv = Converter.load("data/s2t.marshal")
    assert len(conv) > 3000
    assert conv.to_traditional("这个问题") == "這個問題"
    assert conv.to_simplified("這個問題") == "这个问题"
    # 划/劃 曾因扩展区异体字 㓰 污染反向表，映射成「计㓰」
    assert conv.to_simplified("計劃") == "计划"
    assert conv.to_simplified(conv.to_traditional("数据库迁移方案")) == "数据库迁移方案"


def test_simplified_is_only_a_mapping_not_a_model():
    """繁体不走模型：同一句简体的输出只有一个「字面对应」，不涉及语序/用词改写。"""
    from ipinyin.t2s import Converter

    conv = Converter.load("data/s2t.marshal")
    src = "今天天气不错，我们出去走走吧"
    assert conv.to_traditional(src) == "今天天氣不錯，我們出去走走吧"
    # 粤语才需要模型：我们 → 我哋，这不是字形映射能做出来的
    assert conv.to_traditional(src) != "今日天氣唔錯，我哋出去行下啦"


def test_t2s_converter_roundtrip():
    """简↔繁只是显示层映射，必须可逆：繁→简不能产出异体字（㓰 那类）。"""
    from ipinyin.t2s import Converter

    conv = Converter.load("data/s2t.marshal")
    assert len(conv) > 3000
    assert conv.to_traditional("这个问题") == "這個問題"
    assert conv.to_simplified("這個問題") == "这个问题"
    # 划/劃 曾因扩展区异体字 㓰 污染反向表，映射成「计㓰」
    assert conv.to_simplified("計劃") == "计划"
    assert conv.to_simplified(conv.to_traditional("数据库迁移方案")) == "数据库迁移方案"


def test_simplified_is_only_a_mapping_not_a_model():
    """繁体不走模型：同一句简体的输出只有一个「字面对应」，不涉及语序/用词改写。"""
    from ipinyin.t2s import Converter

    conv = Converter.load("data/s2t.marshal")
    src = "今天天气不错，我们出去走走吧"
    assert conv.to_traditional(src) == "今天天氣不錯，我們出去走走吧"
    # 粤语才需要模型：我们 → 我哋，这不是字形映射能做出来的
    assert conv.to_traditional(src) != "今日天氣唔錯，我哋出去行下啦"


def test_abbreviation_initials(lexicon):
    """简拼：rg → 如果（首字母缩写，搜狗/Rime 同一原理）。"""
    import os

    path = os.path.join(ROOT, "data", "initials.marshal")
    if not os.path.exists(path):
        import pytest

        pytest.skip("缺 data/initials.marshal，先跑 tools/build_initials.py")
    lexicon.load_initials(path)
    ime = ImeState(lexicon)
    for char in "rg":
        ime.insert(char)
    assert ime.view().words[0] == "如果"
