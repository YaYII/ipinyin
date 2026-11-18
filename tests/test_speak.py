"""朗读测试：不碰真声卡——用假 espeak 验证缓存、降级与模式映射。"""
import stat

from ipinyin import speak


def test_voice_follows_mode():
    assert speak.voice_for("yue") == "yue"
    assert speak.voice_for("simp") == "cmn"
    assert speak.voice_for("trad") == "cmn"
    assert speak.voice_for("不认识") == "cmn"


def test_normalize_trims_and_caps():
    assert speak.normalize("  你好\n世界 ") == "你好 世界"
    assert speak.normalize("\x07你好") == "你好"
    assert len(speak.normalize("啊" * 500)) == speak.MAX_CHARS


def test_without_espeak_is_silent(tmp_path):
    """没装 espeak-ng：不可用、念不出来，但绝不抛异常。"""
    speaker = speak.Speaker(binary="", player=["true"], cache_dir=str(tmp_path))
    assert not speaker.available
    assert speaker.speak("你好", "yue") is False
    assert speaker.wav("你好", "yue") is None


def test_wav_hits_cache(tmp_path):
    calls = tmp_path / "calls"
    fake = tmp_path / "fake-espeak"
    fake.write_text(f"#!/bin/sh\necho x >> {calls}\nprintf 'RIFFfake'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    speaker = speak.Speaker(
        binary=str(fake), player=["true"], cache_dir=str(tmp_path / "cache")
    )
    assert speaker.wav("你好", "yue") == b"RIFFfake"
    assert speaker.wav("你好", "yue") == b"RIFFfake"
    assert calls.read_text().count("x") == 1, "第二次应命中缓存，不再调 espeak"
    # 换了声线就是另一份音频，不能串味
    speaker.wav("你好", "cmn")
    assert calls.read_text().count("x") == 2
