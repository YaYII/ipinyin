"""ScreenCast：游标选屏与游标文件（不碰真 portal）。"""
from ipinyin import screencast


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(screencast, "home", lambda: str(tmp_path / "vision"))


def test_cursor_roundtrip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    screencast.write_cursor(100, 200, 10, 20)
    assert screencast.read_cursor() == (100, 200)


def test_pick_stream_by_cursor(tmp_path, monkeypatch):
    """左屏 (0,0) 2560x1440，右屏 (2560,0) 3840x2160。"""
    _isolate(tmp_path, monkeypatch)
    streams = [
        {"node": 1, "x": 0, "y": 0, "w": 2560, "h": 1440},
        {"node": 2, "x": 2560, "y": 0, "w": 3840, "h": 2160},
    ]
    screencast.write_cursor(100, 100)
    assert screencast.pick_stream(streams)["node"] == 1
    screencast.write_cursor(3000, 500)
    assert screencast.pick_stream(streams)["node"] == 2


def test_pick_stream_without_cursor_falls_back(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    streams = [{"node": 9, "x": 0, "y": 0, "w": 100, "h": 100}]
    assert screencast.pick_stream(streams)["node"] == 9
    assert screencast.pick_stream([]) is None


def test_cursor_outside_all_streams_falls_back(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    streams = [{"node": 1, "x": 0, "y": 0, "w": 100, "h": 100}]
    screencast.write_cursor(5000, 5000)
    assert screencast.pick_stream(streams)["node"] == 1
