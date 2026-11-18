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
    speaker = speak.Speaker(binary="", player=["true"], cache_dir=str(tmp_path), edge=None)
    assert not speaker.available
    assert speaker.speak("你好", "yue") is False
    assert speaker.wav("你好", "yue") is None


def test_wav_hits_cache(tmp_path):
    calls = tmp_path / "calls"
    fake = tmp_path / "fake-espeak"
    fake.write_text(f"#!/bin/sh\necho x >> {calls}\nprintf 'RIFFfake'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    speaker = speak.Speaker(
        binary=str(fake), player=["true"], cache_dir=str(tmp_path / "cache"), edge=None
    )
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFfake"
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFfake"
    assert calls.read_text().count("x") == 1, "第二次应命中缓存，不再调 espeak"
    # 换了声线就是另一份音频，不能串味
    speaker.wav("你好", speak.PINYIN_VOICE)
    assert calls.read_text().count("x") == 2


def _fake_script(path, body):
    """造一个假命令（shell 脚本），用来观察谁被调用了。"""
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_edge_voice_mapping():
    """粤语走香港声线，简繁走普通话声线——声线必须跟模式绑死。"""
    assert speak.EDGE_VOICES[speak.YUE_VOICE].startswith("zh-HK-")
    assert speak.EDGE_VOICES[speak.PINYIN_VOICE].startswith("zh-CN-")
    assert speak.EDGE_VOICES[speak.YUE_VOICE] != speak.EDGE_VOICES[speak.PINYIN_VOICE]


def test_edge_wins_over_espeak(tmp_path, monkeypatch):
    """装了 edge 就先用它；成功时不该再调 espeak。"""
    espeak_calls = tmp_path / "espeak_calls"
    espeak = _fake_script(
        tmp_path / "fake-espeak", f"echo x >> {espeak_calls}\nprintf 'RIFFespeak'\n"
    )
    edge = _fake_script(
        tmp_path / "fake-edge",
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in --write-media) shift; out="$1";; esac\n'
        '  shift\n'
        "done\n"
        'printf "MP3" > "$out"\n',
    )
    ffmpeg = _fake_script(tmp_path / "fake-ffmpeg", 'for last; do :; done\nprintf "RIFFedge" > "$last"\n')
    monkeypatch.setattr(speak.shutil, "which", lambda name: ffmpeg if name == "ffmpeg" else None)
    speaker = speak.Speaker(
        binary=espeak, player=["true"], cache_dir=str(tmp_path / "cache"), edge=[edge]
    )
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFedge"
    assert not espeak_calls.exists(), "edge 成功时不该再调 espeak"


def test_edge_failure_falls_back_to_espeak(tmp_path, monkeypatch):
    """断网/超时不能让朗读没声音：edge 失败要静默回落 espeak。"""
    espeak_calls = tmp_path / "espeak_calls"
    espeak = _fake_script(
        tmp_path / "fake-espeak", f"echo x >> {espeak_calls}\nprintf 'RIFFespeak'\n"
    )
    edge = _fake_script(tmp_path / "fake-edge", "exit 1\n")
    ffmpeg = _fake_script(tmp_path / "fake-ffmpeg", "exit 0\n")
    monkeypatch.setattr(speak.shutil, "which", lambda name: ffmpeg if name == "ffmpeg" else None)
    speaker = speak.Speaker(
        binary=espeak, player=["true"], cache_dir=str(tmp_path / "cache"), edge=[edge]
    )
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFespeak"
    assert espeak_calls.read_text().count("x") == 1


def test_edge_cache_does_not_leak_into_espeak(tmp_path, monkeypatch):
    """同一句话两种引擎的音频不能互相冒充——断网后不能播到 edge 留下的缓存。"""
    espeak = _fake_script(tmp_path / "fake-espeak", "printf 'RIFFespeak'\n")
    edge = _fake_script(
        tmp_path / "fake-edge",
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in --write-media) shift; out="$1";; esac\n'
        '  shift\n'
        "done\n"
        'printf "MP3" > "$out"\n',
    )
    ffmpeg = _fake_script(tmp_path / "fake-ffmpeg", 'for last; do :; done\nprintf "RIFFedge" > "$last"\n')
    monkeypatch.setattr(speak.shutil, "which", lambda name: ffmpeg if name == "ffmpeg" else None)
    cache = str(tmp_path / "cache")
    assert speak.Speaker(binary=espeak, player=["true"], cache_dir=cache,
                         edge=[edge]).wav("你好", speak.YUE_VOICE) == b"RIFFedge"
    assert speak.Speaker(binary=espeak, player=["true"], cache_dir=cache,
                         edge=None).wav("你好", speak.YUE_VOICE) == b"RIFFespeak"
