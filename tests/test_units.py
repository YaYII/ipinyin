"""词级音库：切词、拼接、覆盖不全时回落（不碰真声卡、不调 ffmpeg）。"""
import json
import stat

from ipinyin import speak, units


def make_lib(tmp_path, monkeypatch, words):
    root = tmp_path / "units"
    root.mkdir()
    index = {word: f"{i}.opus" for i, word in enumerate(words)}
    (root / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    lib = units.UnitLibrary(str(root), ffmpeg="/bin/true")
    monkeypatch.setattr(lib, "_decode", lambda name: b"\x00\x01" * 100)
    return lib


def test_segment_prefers_longest_word(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, monkeypatch, ["我哋", "聽日", "去", "飲茶"])
    assert lib.segment("我哋聽日去飲茶") == ["我哋", "聽日", "去", "飲茶"]


def test_segment_falls_back_to_single_char(tmp_path, monkeypatch):
    """词表没有「我哋」时，退到单字也能拼（单字听感用户已确认可接受）。"""
    lib = make_lib(tmp_path, monkeypatch, ["我", "哋", "聽日"])
    assert lib.segment("我哋聽日") == ["我", "哋", "聽日"]


def test_missing_unit_returns_none(tmp_path, monkeypatch):
    """有一个字不在库里就整句回落，不能念半句。"""
    lib = make_lib(tmp_path, monkeypatch, ["我"])
    assert lib.segment("我愛你") is None
    assert lib.render("我愛你") is None


def test_render_builds_valid_wav(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, monkeypatch, ["你好", "世界"])
    data = lib.render("你好世界")
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    gap = b"\x00\x00" * int(units.SR * units.GAP_MS / 1000)
    # 2 段 PCM（各 200 字节）+ 段间与句尾各一个间隙
    assert len(data) == 44 + 400 + 2 * len(gap)


def test_empty_index_is_unavailable(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    lib = units.UnitLibrary(str(root), ffmpeg="/bin/true")
    assert not lib.available
    assert lib.render("你好") is None


def test_speak_prefers_units_for_yue_only(tmp_path, monkeypatch):
    """音库单元是粤语男声：粤语用它，普通话必须走别的后端。"""
    lib = make_lib(tmp_path, monkeypatch, ["你好"])
    espeak = tmp_path / "fake-espeak"
    espeak.write_text("#!/bin/sh\nprintf 'RIFFespeak'\n")
    espeak.chmod(espeak.stat().st_mode | stat.S_IEXEC)
    speaker = speak.Speaker(
        binary=str(espeak), player=["true"], cache_dir=str(tmp_path / "cache"),
        edge=None, unit_lib=lib,
    )
    assert speaker.wav("你好", speak.YUE_VOICE)[:4] == b"RIFF"
    assert speaker.wav("你好", speak.YUE_VOICE) != b"RIFFespeak"
    assert speaker.wav("你好", speak.PINYIN_VOICE) == b"RIFFespeak"


def test_trim_removes_padding_silence():
    """edge 把短文本补到 1.87s；不解静音，每个词后面都会停 1.3 秒。"""
    pcm = b"\x00\x00" * 500 + b"\x00\x10" * 200 + b"\x00\x00" * 500
    out = units._trim(pcm)
    assert 0 < len(out) < len(pcm)
    assert len(out) >= 200 * 2          # 有声部分必须完整保留
