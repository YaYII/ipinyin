import os

"""屏幕感知：扫描新截图、调用可插拔模型命令、落上下文。"""
from ipinyin import vision


def _isolate(tmp_path, monkeypatch):
    shots = tmp_path / "shots"
    shots.mkdir(exist_ok=True)
    monkeypatch.setenv("IPINYIN_VISION_DIRS", str(shots))
    monkeypatch.setattr(vision, "home", lambda: str(tmp_path / "vision"))
    return shots


def test_scan_detects_new_only_once(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    (shots / "a.png").write_bytes(b"x")
    (shots / "b.txt").write_text("不是图片")
    first = vision.scan()
    assert first == [str(shots / "a.png")]
    assert vision.scan() == [], "同一张截图不能重复处理"


def test_scan_picks_up_later_files(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    (shots / "a.png").write_bytes(b"x")
    vision.scan()
    (shots / "b.jpg").write_bytes(b"y")
    assert vision.scan() == [str(shots / "b.jpg")]


def test_describe_uses_command_template(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    image = shots / "a.png"
    image.write_bytes(b"x")
    monkeypatch.setenv("IPINYIN_VISION_CMD", "echo 画面里有终端 {image}")
    text = vision.describe(str(image))
    assert "画面里有终端" in text
    assert "a.png" in text


def test_describe_without_command_returns_none(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    image = shots / "a.png"
    image.write_bytes(b"x")
    monkeypatch.delenv("IPINYIN_VISION_CMD", raising=False)
    assert vision.describe(str(image)) is None


def test_describe_handles_failure(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    image = shots / "a.png"
    image.write_bytes(b"x")
    monkeypatch.setenv("IPINYIN_VISION_CMD", "exit 3")
    assert vision.describe(str(image)) is None


def test_capture_context_logs(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    (shots / "a.png").write_bytes(b"x")
    monkeypatch.setenv("IPINYIN_VISION_CMD", "echo 在写代码")
    rows = vision.capture_context()
    assert len(rows) == 1
    assert rows[0]["desc"] == "在写代码"
    assert vision.recent(5)[0]["desc"] == "在写代码"
    assert vision.capture_context() == [], "已处理过的截图不再重复"


def test_capture_copies_frame_and_cleanup_removes_it(tmp_path, monkeypatch):
    """原件是你的，永不删；只删自己复制过来的帧。"""
    import os
    import time as _time

    shots = _isolate(tmp_path, monkeypatch)
    image = shots / "a.png"
    image.write_bytes(b"x")
    monkeypatch.setenv("IPINYIN_VISION_CMD", "echo 看到终端")

    rows = vision.capture_context()
    frame = rows[0]["image"]
    assert frame.startswith(vision.frames_dir())
    assert os.path.exists(frame)
    assert image.exists(), "原始截图必须保留"

    # 把帧的修改时间改成两天前，再清理
    old = _time.time() - 2 * 86400
    os.utime(frame, (old, old))
    result = vision.cleanup()
    assert result["removed"] == 1
    assert not os.path.exists(frame)
    assert image.exists()


def test_cleanup_keeps_recent_frames(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    (shots / "a.png").write_bytes(b"x")
    monkeypatch.setenv("IPINYIN_VISION_CMD", "echo ok")
    rows = vision.capture_context()
    result = vision.cleanup()
    assert result["removed"] == 0
    assert os.path.exists(rows[0]["image"])


def test_cleanup_never_touches_original_dirs(tmp_path, monkeypatch):
    shots = _isolate(tmp_path, monkeypatch)
    original = shots / "keep.png"
    original.write_bytes(b"x")
    vision.cleanup()
    assert original.exists(), "原图目录不在清理范围"
