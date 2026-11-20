"""常听模式的分句逻辑：底噪校准 → 检测说话 → 静音收尾 → 交给识别。"""
import io

import numpy as np

from ipinyin import asr


class FakeRec:
    def __init__(self):
        self.calls = []

    def recognize_samples(self, samples, rate):
        self.calls.append((len(samples), rate))
        return "测试文本"


def _frames(silence: int, loud: int, tail: int):
    frame = 16000 * 100 // 1000
    quiet = np.zeros(frame, dtype="<i2").tobytes()
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(frame) * 8000).astype("<i2").tobytes()
    return io.BytesIO(quiet * silence + speech * loud + quiet * tail)


def test_listener_detects_and_delivers(monkeypatch):
    got = []
    lst = asr.Listener(on_text=got.append, language="yue")
    lst._rec = FakeRec()
    lst._open = lambda: _frames(8, 20, 10)      # 0.8s 静音 + 2s 有声 + 1s 静音
    lst.run()                                    # 同步跑完（流会 EOF）
    assert got == ["测试文本"], "检测到说话后应该在静音处切一句交给识别"
    assert lst._rec.calls and lst._rec.calls[0][1] == 16000


def test_listener_ignores_too_short_speech(monkeypatch):
    got = []
    lst = asr.Listener(on_text=got.append, language="yue")
    lst._rec = FakeRec()
    lst._open = lambda: _frames(8, 2, 10)       # 只有 0.2s 有声，短于 MIN_SPEECH_MS
    lst.run()
    assert got == [], "太短的杂音不该触发识别"
