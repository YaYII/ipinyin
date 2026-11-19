"""朗读自监控：异常规则、样本脱敏、记录写入。"""
import json

from ipinyin import tts_metrics


def test_detect_padding():
    """edge 把短文本补到 1.87s：两个字 1.87s 必须报 padding。"""
    assert "padding" in tts_metrics.detect(2, "edge", 300, 1.87)


def test_detect_normal_short_word():
    assert tts_metrics.detect(2, "edge", 300, 0.63) == []


def test_detect_too_slow_and_too_fast():
    assert "too-slow" in tts_metrics.detect(10, "edge", 300, 8.0)    # 1.25 字/秒
    assert "too-fast" in tts_metrics.detect(10, "edge", 300, 0.8)    # 12.5 字/秒


def test_detect_fallbacks_and_empty():
    assert "fallback-espeak" in tts_metrics.detect(5, "espeak", 30, 1.0)
    assert "fallback-units" in tts_metrics.detect(5, "units", 30, 1.0)
    assert tts_metrics.detect(5, "edge", 30, 0) == ["empty-audio"]


def test_sample_keeps_only_two_chars():
    """样本只留两个字：能定位问题，还原不了原文。"""
    assert tts_metrics.sample("我哋聽日一齊去飲茶") == "我哋…"
    assert tts_metrics.sample("聽日") == "聽日"


def test_record_appends_and_flags(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(tts_metrics, "metrics_path", lambda: str(path))
    tts_metrics.record("聽日", "edge", 300, 1.87)
    tts_metrics.record("我哋聽日一齊去飲茶", "edge", 500, 1.93)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["flags"] == ["padding", "too-slow"]
    assert rows[1].get("flags") is None
    assert rows[0]["sample"] == "聽日"
    assert rows[0]["rate"] == round(2 / 1.87, 2)


def test_load_skips_broken_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"chars": 2}\n不是 JSON\n{"chars": 9}\n', encoding="utf-8")
    rows = tts_metrics.load(str(path))
    assert [row["chars"] for row in rows] == [2, 9]
