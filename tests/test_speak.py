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
    # 不截断：有多少念多少
    assert speak.normalize("啊" * 500) == "啊" * 500


def test_to_pinyin_keeps_ascii_and_drops_symbols():
    table = {"声": "shēng", "音": "yīn", "我": "wǒ", "们": "men"}
    assert speak.to_pinyin("声音", table) == "shēng yīn"
    assert speak.to_pinyin("我们", table) == "wǒ men"
    # 字母数字留着（API、v2），全角标点/emoji 丢掉（念出来只是噪音）
    assert speak.to_pinyin("v2.0 API", table) == "v2.0 API"
    assert speak.to_pinyin("声音😀", table) == "shēng yīn"


def test_without_espeak_is_silent(tmp_path):
    """没装 espeak-ng：不可用、念不出来，但绝不抛异常。"""
    speaker = speak.Speaker(binary="", player=["true"], cache_dir=str(tmp_path), edge=None, unit_lib=None)
    assert not speaker.available
    assert speaker.speak("你好", "yue") is False
    assert speaker.wav("你好", "yue") is None


def test_wav_hits_cache(tmp_path):
    calls = tmp_path / "calls"
    fake = tmp_path / "fake-espeak"
    fake.write_text(f"#!/bin/sh\necho x >> {calls}\nprintf 'RIFFfake'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    speaker = speak.Speaker(
        binary=str(fake), player=["true"], cache_dir=str(tmp_path / "cache"), edge=None, unit_lib=None
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
    """没有本地音库时，装了 edge 就用它；成功时不该再调 espeak。"""
    monkeypatch.setenv("IPINYIN_TTS_EDGE", "1")
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
    ffmpeg = _fake_script(tmp_path / "fake-ffmpeg", 'for last; do :; done\n[ "$last" = "-" ] && exit 0\nprintf "RIFFedge" > "$last"\n')
    monkeypatch.setattr(speak.shutil, "which", lambda name: ffmpeg if name == "ffmpeg" else None)
    speaker = speak.Speaker(
        binary=espeak, player=["true"], cache_dir=str(tmp_path / "cache"), edge=[edge], unit_lib=None
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
        binary=espeak, player=["true"], cache_dir=str(tmp_path / "cache"), edge=[edge], unit_lib=None
    )
    assert speaker.wav("你好", speak.YUE_VOICE) == b"RIFFespeak"
    assert espeak_calls.read_text().count("x") == 1


def test_edge_cache_does_not_leak_into_espeak(tmp_path, monkeypatch):
    monkeypatch.setenv("IPINYIN_TTS_EDGE", "1")
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
    ffmpeg = _fake_script(tmp_path / "fake-ffmpeg", 'for last; do :; done\n[ "$last" = "-" ] && exit 0\nprintf "RIFFedge" > "$last"\n')
    monkeypatch.setattr(speak.shutil, "which", lambda name: ffmpeg if name == "ffmpeg" else None)
    cache = str(tmp_path / "cache")
    assert speak.Speaker(binary=espeak, player=["true"], cache_dir=cache,
                         edge=[edge], unit_lib=None).wav("你好", speak.YUE_VOICE) == b"RIFFedge"
    assert speak.Speaker(binary=espeak, player=["true"], cache_dir=cache,
                         edge=None, unit_lib=None).wav("你好", speak.YUE_VOICE) == b"RIFFespeak"


def test_rapid_speak_plays_everything(tmp_path):
    """连按不丢字：每条都会播（分段流水线），不会因为积压被丢掉。"""
    import time

    plays = tmp_path / "plays"
    espeak = _fake_script(tmp_path / "fake-espeak", "sleep 0.05\nprintf 'RIFF'\n")
    player = _fake_script(tmp_path / "fake-player", f"echo x >> {plays}\n")
    speaker = speak.Speaker(
        binary=espeak, player=[player], cache_dir=str(tmp_path / "cache"), edge=None, unit_lib=None
    )
    assert speaker.speak("第一句", speak.YUE_VOICE)
    assert speaker.speak("第二句", speak.YUE_VOICE)
    time.sleep(1.5)
    assert plays.read_text().count("x") == 2, "两条都要播，一条都不能丢"


def test_stop_clears_queue(tmp_path):
    """stop() 才是打断：清队列并掐掉正在播的那条。"""
    import time

    plays = tmp_path / "plays"
    espeak = _fake_script(tmp_path / "fake-espeak", "sleep 0.3\nprintf 'RIFF'\n")
    player = _fake_script(tmp_path / "fake-player", f"echo x >> {plays}\n")
    speaker = speak.Speaker(
        binary=espeak, player=[player], cache_dir=str(tmp_path / "cache"), edge=None, unit_lib=None
    )
    speaker.speak("第一句", speak.YUE_VOICE)
    speaker.speak("第二句", speak.YUE_VOICE)
    speaker.stop()
    time.sleep(1.2)
    assert not plays.exists() or plays.read_text().count("x") <= 1


def test_edge_output_is_trimmed():
    """edge 把短文本补到 1.87s；不裁的话两个字的词听起来像两秒。"""
    from ipinyin import units

    pcm = b"\x00\x00" * 2000 + b"\x00\x10" * 100 + b"\x00\x00" * 2000
    wav = units._wav_header(len(pcm)) + pcm
    out = speak._trim_wav(wav)
    assert out[:4] == b"RIFF"
    assert len(out) < len(wav)
    assert len(out) >= 44 + 100 * 2          # 有声部分完整保留
    # 首尾必须淡入淡出，否则裁切点有 DC 台阶，播放时"嗒"一声
    import array
    pcm = array.array("h")
    pcm.frombytes(out[44:])
    assert abs(pcm[0]) < 500 and abs(pcm[-1]) < 500


def test_auto_voice_follows_tts_state_not_input_mode(monkeypatch):
    """状态选了粤语，就算正在打普通话，也要播粤语。"""
    from types import SimpleNamespace

    from ipinyin import engine, speak

    played = []
    monkeypatch.setattr(speak, "get_speaker",
                        lambda: SimpleNamespace(speak=lambda t, v: played.append((t, v)) or True))
    monkeypatch.setattr(engine.config, "tts_enabled", lambda: True)

    class FakeEngine:
        _speak_auto = engine.PinyinEngine._speak_auto
        _spoken_text = engine.PinyinEngine._spoken_text

        def __init__(self, state, mode):
            self._tts_state = state
            self._ime = SimpleNamespace(mode=mode, chao=None)

        def _refresh_tts_auto(self):
            pass

    FakeEngine("yue", "simp")._speak_auto("你好")
    assert played[-1][1] == speak.YUE_VOICE, "状态粤语必须播粤语，不能跟输入模式"

    FakeEngine("cmn", "yue")._speak_auto("你好")
    assert played[-1][1] == speak.PINYIN_VOICE, "状态国语才播国语"

    FakeEngine("off", "yue")._speak_auto("你好")
    assert len(played) == 2, "关档不播"


def test_tts_lang_prefers_state_then_mode(monkeypatch):
    """Ctrl+回车与上屏即读共用同一条规则：状态优先，关档按模式兜底。"""
    from types import SimpleNamespace

    from ipinyin import engine

    class FakeEngine:
        _tts_lang = engine.PinyinEngine._tts_lang

        def __init__(self, state, mode):
            self._tts_state = state
            self._ime = SimpleNamespace(mode=mode)

        def _refresh_tts_auto(self):
            pass

    assert FakeEngine("yue", "simp")._tts_lang() == "yue"
    assert FakeEngine("cmn", "yue")._tts_lang() == "cmn"
    assert FakeEngine("off", "yue")._tts_lang() == "yue", "关档时按打字模式"


def test_take_batch_merges_same_voice_only(tmp_path):
    """合并只发生在同声线之间；换声线就断开，不能把粤语和国语拼成一句。"""
    speaker = speak.Speaker(
        binary=None, player=["true"], cache_dir=str(tmp_path / "cache"),
        edge=None, unit_lib=None,
    )
    with speaker._cv:
        speaker._queue.extend([("甲", speak.YUE_VOICE, True, None), ("乙", speak.YUE_VOICE, True, None),
                               ("丙", speak.PINYIN_VOICE, True, None)])
    text, voice, _, _ = speaker._take_batch()
    assert text == "甲乙" and voice == speak.YUE_VOICE
    text, voice, _, _ = speaker._take_batch()
    assert text == "丙" and voice == speak.PINYIN_VOICE


def test_enter_replays_whole_sentence(monkeypatch):
    """空 buffer 按回车 = 一句说完：把整句连贯念一遍，而不是只念最后一个词。"""
    from types import SimpleNamespace

    from ipinyin import engine, speak

    played = []
    dropped = []
    fake = SimpleNamespace(
        speak=lambda t, v, prefer=None: played.append((t, prefer)) or True,
        has_cache=lambda t, v, backend="edge": False,
        warm=lambda t, v, prefer=None: None,
        pending=lambda: 3,
        stop=lambda: dropped.append(True),
    )
    monkeypatch.setattr(speak, "get_speaker", lambda: fake)
    monkeypatch.setattr(engine.config, "tts_enabled", lambda: True)

    class FakeEngine:
        _speak_sentence = engine.PinyinEngine._speak_sentence
        _tts_lang = engine.PinyinEngine._tts_lang
        _spoken_text = engine.PinyinEngine._spoken_text
        _sentence_text = engine.PinyinEngine._sentence_text

        def get_surrounding_text(self):
            raise RuntimeError("app 不支持")

        def __init__(self):
            self._sentence = "我唔知你講乜"
            self._last_spoken = ""
            self._tts_state = "yue"
            self._ime = SimpleNamespace(mode="yue", chao=None)

        def _refresh_tts_auto(self):
            pass

    eng = FakeEngine()
    eng._speak_sentence()
    assert [text for text, _ in played] == ["我唔知你講乜"]
    assert played[0][1] == "local", "在线无缓存时先用本地，避免等 3-4 秒"
    assert dropped, "整句重播前要丢掉逐词积压，否则同一句会播两遍"
    assert eng._sentence == "", "念完要清空，否则下次回车会重复"


def test_spoken_text_converts_mandarin_to_cantonese():
    """打普通话、朗读档是粤语：念之前必须先转成粤语，否则是「粤语读普通话」。"""
    from types import SimpleNamespace

    from ipinyin import engine

    class FakeEngine:
        _spoken_text = engine.PinyinEngine._spoken_text

        def __init__(self, mode):
            self._ime = SimpleNamespace(mode=mode)

    # mode != yue：正文是普通话，要转
    assert FakeEngine("simp")._spoken_text("看看", "yue") != "看看"
    # mode == yue：正文本来就是粤语，不能再转
    assert FakeEngine("yue")._spoken_text("睇下", "yue") == "睇下"
    # 念国语时永远不转
    assert FakeEngine("simp")._spoken_text("看看", "cmn") == "看看"


def test_sentence_text_prefers_surrounding_text():
    """应用里的真实文本优先：删掉的字不能再念出来。"""
    from types import SimpleNamespace

    from ipinyin import engine

    class FakeText:
        def __init__(self, s):
            self._s = s

        def get_text(self):
            return self._s

    class FakeEngine:
        _sentence_text = engine.PinyinEngine._sentence_text

        def __init__(self, surrounding, cursor):
            self._surrounding = surrounding
            self._cursor = cursor
            self._sentence = "上屏累积的旧内容"

        def get_surrounding_text(self):
            return FakeText(self._surrounding), self._cursor, self._cursor

    # 输入框里最终是「睇下啦」，之前累积的是「睇下睇下啦」——要念输入框里的
    eng = FakeEngine("前面一句。睇下啦", len("前面一句。睇下啦"))
    assert eng._sentence_text() == "睇下啦"

    # 应用不支持 surrounding text：退回累积缓冲
    class NoSurround(FakeEngine):
        def get_surrounding_text(self):
            raise RuntimeError("not supported")

    assert NoSurround("", 0)._sentence_text() == "上屏累积的旧内容"


def test_commit_stage_sends_everything_at_once():
    """中间态：选中的字先攒着，回车时一次性上屏，不是每个词各自上屏。"""
    from ipinyin import engine

    sent = []

    class FakeEngine:
        _commit_stage = engine.PinyinEngine._commit_stage

        def __init__(self):
            self._stage = "我唔知你講乜"
            self._sentence = ""
            self.hide_preedit_text = lambda: None
            self.commit_text = lambda t: sent.append(t.get_text())
            self._speak_sentence = lambda: sent.append("<speak>")

    eng = FakeEngine()
    assert eng._commit_stage() is True
    assert sent == ["我唔知你講乜", "<speak>"], "整句一次上屏，然后整句念"
    assert eng._stage == ""


def test_chunks_splits_long_sentence_at_punctuation():
    """长句分段：播一段时合成下一段，感知延迟≈第一段。"""
    from ipinyin import speak

    text = "我哋今日去行街買嘢食，順便睇下戲然後返屋企食飯，夜晚再出去散步傾下偈。"
    parts = speak.chunks(text)
    assert len(parts) >= 3
    assert "".join(parts) == text, "切段不能丢字"
    assert all(len(p) <= speak.CHUNK_CHARS + 8 for p in parts)
    # 短句不切
    assert speak.chunks("你好") == ["你好"]


def test_deleted_tail_is_not_spoken(monkeypatch):
    """用户删掉尾部的字：以应用文本为准，不能退回累积缓冲。"""
    import time
    from types import SimpleNamespace

    from ipinyin import engine

    class FakeText:
        def get_text(self):
            return "我哋今日去行街"

    class FakeEngine:
        _sentence_text = engine.PinyinEngine._sentence_text
        _last_commit_at = 0.0

        def __init__(self):
            self._sentence = "我哋今日去行街買嘢食"     # 累积里还有被删的字

        def get_surrounding_text(self):
            return FakeText(), 7, 7

    eng = FakeEngine()
    # 最后一次上屏是 5 秒前 → 不可能是「应用滞后」，只能是用户删除
    eng._last_commit_at = time.monotonic() - 5.0
    assert eng._sentence_text() == "我哋今日去行街"
    assert eng._sentence == "我哋今日去行街", "缓冲要跟着同步，下次不再带旧字"


def test_empty_enter_does_not_play(monkeypatch):
    """空回车（两次回车之间没有新内容）不该播放。"""
    from types import SimpleNamespace

    from ipinyin import engine, speak

    played = []
    fake = SimpleNamespace(speak=lambda t, v: played.append(t) or True,
                           pending=lambda: 0, stop=lambda: None)
    monkeypatch.setattr(speak, "get_speaker", lambda: fake)
    monkeypatch.setattr(engine.config, "tts_enabled", lambda: True)

    class FakeEngine:
        _speak_sentence = engine.PinyinEngine._speak_sentence
        _tts_lang = engine.PinyinEngine._tts_lang
        _spoken_text = engine.PinyinEngine._spoken_text

        def __init__(self):
            self._sentence = ""          # 没有新内容
            self._last_spoken = ""
            self._tts_state = "yue"
            self._ime = SimpleNamespace(mode="yue", chao=None)

        def _sentence_text(self):
            return "整句已经有内容"       # 输入框里有旧内容，但不该念

        def _refresh_tts_auto(self):
            pass

    FakeEngine()._speak_sentence()
    assert played == [], "空回车不能播放"
