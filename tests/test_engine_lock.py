"""单实例锁：不允许两个引擎同时注册（否则 IBus 把按键送进半死的那个）。"""
import os

from ipinyin import engine


def test_singleton_writes_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.config, "cache_home", lambda: str(tmp_path))
    handle = engine._acquire_singleton()
    assert handle is not None
    assert (tmp_path / "engine.lock").read_text().strip() == str(os.getpid())
