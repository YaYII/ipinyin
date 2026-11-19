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
