import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


@pytest.fixture(autouse=True)
def _isolate_metrics(tmp_path, monkeypatch):
    """测试不许写用户的朗读指标：假音频会被判成 empty-audio，污染真实报告。"""
    from ipinyin import tts_metrics

    monkeypatch.setattr(tts_metrics, "metrics_path", lambda: str(tmp_path / "metrics.jsonl"))
