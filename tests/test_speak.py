"""朗读测试：不碰真声卡——用假 espeak 验证缓存、降级、声线与拼音转换。"""
import stat

from ipinyin import speak


def test_voice_follows_mode():
    assert speak.voice_for("yue") == "yue"
    assert speak.voice_for("simp") == speak.PINYIN_VOICE
    assert speak.voice_for("trad") == speak.PINYIN_VOICE
    assert speak.voice_for("不认识") == speak.PINYIN_VOICE


def test_target_keeps_voice_of_mode():
    """念上屏句时声线也必须跟着模式——粤语模式拿普通话音去读「嘅」会变成拼字母。"""
    assert speak.target("yue", "嘅", None) == ("嘅", "yue")
    assert speak.target("yue", None, "嘅唔啱") == ("嘅唔啱", "yue")
    assert speak.target("simp", None, "声音") == ("声音", speak.PINYIN_VOICE)
    assert speak.target("simp", None, None) is None


def test_normalize_trims_and_caps():
    assert speak.normalize("  你好\n世界 ") == "你好 世界"
    assert speak.normalize("\x07你好") == "你好"
    assert len(speak.normalize("啊" * 500)) == speak.MAX_CHARS


def test_to_pinyin_keeps_ascii_and_drops_symbols():
    table = {"声": "shēng", "音": "yīn", "我": "wǒ", "们": "men"}
    assert speak.to_pinyin("声音", table) == "shēng yīn"
    assert speak.to_pinyin("我们", table) == "wǒ men"
    # 字母数字留着（API、v2），全角标点/emoji 丢掉（念出来只是噪音）
    assert speak.to_pinyin("v2.0 API", table) == "v2.0 API"
    assert speak.to_pinyin("声音😀", table) == "shēng yīn"


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
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFfake"
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFfake"
    assert calls.read_text().count("x") == 1, "第二次应命中缓存，不再调 espeak"
    # 换了声线就是另一份音频，不能串味
    speaker.wav("你好", speak.PINYIN_VOICE)
    assert calls.read_text().count("x") == 2
