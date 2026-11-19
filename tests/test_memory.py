"""输入法记忆体：采集、增量整理、导出、清空。"""
from ipinyin import memory


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "home", lambda: str(tmp_path / "memory"))
    monkeypatch.setattr(memory, "_LEXICON", {})


def test_record_and_consolidate(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("meiyou", "没有", "simp", rank=0, gap_ms=1200)
    memory.record("kandao", "看到", "simp", rank=3, gap_ms=800)
    assert memory.pending() > 0

    result = memory.consolidate()
    assert result["processed"] == 2
    profile = memory._load_profile()
    assert profile["commits"] == 2
    assert profile["words"]["没有"] == 1
    # 越过首选（rank=3）要记进「首选不准」
    assert "kandao\t看到" in profile["corrections"]
    assert memory.pending() == 0, "整理完游标要跟上，不能重复统计"


def test_consolidate_is_incremental(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("a", "第一句", "simp")
    memory.consolidate()
    memory.record("b", "第二句", "simp")
    result = memory.consolidate()
    assert result["processed"] == 1, "只整理新增的那条"
    assert memory._load_profile()["commits"] == 2


def test_consolidate_without_events(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert memory.consolidate() == {"processed": 0}


def test_disabled_records_nothing(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setenv("IPINYIN_MEMORY", "0")
    memory.record("a", "不该记")
    assert memory.pending() == 0


def test_single_char_and_stopwords_are_not_high_freq(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("de", "的", "simp")
    memory.record("wo", "我", "simp")
    memory.record("gongchengshi", "工程师", "simp")
    memory.consolidate()
    words = memory._load_profile()["words"]
    assert "工程师" in words
    assert "的" not in words and "我" not in words


def test_export_markdown_has_sections(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("gongchengshi", "工程师", "simp")
    memory.record("kandao", "看到", "simp", rank=2)
    memory.consolidate()
    text = memory.export_markdown()
    assert "# 我的输入画像" in text
    assert "## 高频词" in text
    assert "## 首选不准" in text
    assert "kandao" in text


def test_wipe_removes_everything(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("a", "测试词")
    memory.consolidate()
    memory.wipe()
    assert memory.status()["commits"] == 0
    assert memory.pending() == 0


def test_ascii_and_raw_pinyin_not_counted(tmp_path, monkeypatch):
    """英文/原样上屏的拼音不算高频词——它们是噪声，不是你的用词。"""
    _isolate(tmp_path, monkeypatch)
    memory.record("ai", "ai")
    memory.record("cansyu", "cansyu ")
    memory.record("gongchengshi", "工程师")
    memory.consolidate()
    words = memory._load_profile()["words"]
    assert "ai" not in words
    assert "cansyu " not in words
    assert "工程师" in words


def test_abandon_and_input_feel(tmp_path, monkeypatch):
    """打到一半放弃是最有价值的负样本；退格/翻页/耗时都要能记下来。"""
    _isolate(tmp_path, monkeypatch)
    memory.record(None, "", kind="abandon", keys=4, backs=2, flips=1, ms=3200)
    memory.record("gongchengshi", "工程师", keys=11, backs=1, flips=0, ms=2400)
    memory.consolidate()
    profile = memory._load_profile()
    assert profile["abandons"] == 1
    assert profile["commits"] == 1
    assert profile["backs"] == 3
    assert profile["keystrokes"] == 15
    text = memory.export_markdown()
    assert "退格率" in text and "放弃率" in text


def test_focus_switch_counted(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record(None, "", kind="focus", gap_ms=1000)
    memory.record(None, "", kind="focus", gap_ms=2500)
    memory.consolidate()
    assert memory._load_profile()["focus_switches"] == 2


def test_typos_needs_both_signals(tmp_path, monkeypatch):
    """词库缺 + 模型也觉得不像话，才报。只用一条会误报（很好 不在词库，屏幕 模型不认识）。"""
    _isolate(tmp_path, monkeypatch)

    class FakeLM:
        def logp(self, text):
            return -60.0 if text == "顺连" else -6.0

    monkeypatch.setattr(memory, "_load_lm", lambda: FakeLM())
    monkeypatch.setattr(memory, "_lexicon_words", lambda: {"屏幕"})
    memory.record("shunlian", "顺连")
    memory.record("pingmu", "屏幕")
    memory.consolidate()
    rows = memory.typos()
    assert [row["word"] for row in rows] == ["顺连"]
    assert rows[0]["in_lexicon"] is False


def test_apply_fixes_writes_boost_queue(tmp_path, monkeypatch):
    """提权键用「错误写法」的拼音，词换成正确写法——下次打同样的音首选就对。"""
    import json

    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(memory, "_pinyin_keys_for",
                        lambda words: {"哪个": ["nage"], "那个": ["nage"]})
    monkeypatch.setattr(memory.config, "boost_queue_path",
                        lambda: str(tmp_path / "boost.jsonl"))
    # 显式给 weight，避免依赖「按差距算」那条路径（它要读真实词频）
    applied = memory.apply_fixes([{"wrong": "哪个", "right": "那个", "why": "同音误用"}], weight=3)
    assert applied[0]["keys"] == ["nage"]
    lines = (tmp_path / "boost.jsonl").read_text(encoding="utf-8").splitlines()
    item = json.loads(lines[0])
    assert item["key"] == "nage" and item["word"] == "那个" and item["weight"] == 3


def test_suggest_fixes_parses_llm_json(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    memory.record("nage", "哪个")
    memory.consolidate()
    monkeypatch.setattr(memory, "_ask_llm", lambda prompt, max_tokens=4000: (
        '好的：[{"wrong":"哪个","right":"那个","why":"同音误用"}]'))
    rows = memory.suggest_fixes(10)
    assert rows == [{"wrong": "哪个", "right": "那个", "why": "同音误用"}]


def test_needed_weight_beats_current_top(monkeypatch):
    """+3 翻不动历史统计（实测 131.79 vs 93.67），必须按差距算。"""
    class FakeModel:
        def preferred(self, key, limit=9):
            return ["哪个", "那个"]

        def score(self, key, word):
            return 131.79 if word == "哪个" else 93.67

    assert memory._needed_weight("nage", "那个", FakeModel()) == 43
