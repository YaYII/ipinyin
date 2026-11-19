"""切档键必须在任何状态下都生效——英文档按 Tab 要能回到中文。

回归的是这个真实 bug：TOGGLE_TRAD 以前写在 elif 链里（在 `if not composing:
return False` 之后），英文档 buffer 永远为空，于是 Tab 被吞掉，
切到英文就再也回不来了。
"""
import types

import pytest

pytest.importorskip("gi")
pytest.importorskip("ipinyin.engine")

from ipinyin import engine as E  # noqa: E402
from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402

TAB = 0xFF09
SHIFT_RELEASE = (0xFFE1, 1 << 30)


@pytest.fixture()
def harness(monkeypatch):
    ime = ImeState(Lexicon.load("data/lexicon.tsv"))
    ime.mode = "en"
    monkeypatch.setattr(E, "_write_mode", lambda mode: None)
    stub = types.SimpleNamespace(
        ready=True,
        _ime=ime,
        _shift_down=False,
        _shift_clean=True,
        _last_cn_mode="simp",
        _plain=False,
        _sync_now=lambda: None,
        _flash_mode_hint=lambda: None,
        _push_mode_hint=lambda *args, **kwargs: None,
    )
    stub.ime = ime
    return stub, ime


def test_tab_leaves_english_mode(harness):
    stub, ime = harness
    assert ime.mode == "en"
    assert E.PinyinEngine._handle(stub, TAB, 0) is True   # 被输入法吃掉
    assert ime.mode == "simp"                             # 真的切回来了


def test_shift_forces_chinese_from_english(harness):
    stub, ime = harness
    # 轻按 = 按下 + 释放，且中间没按别的键
    stub._shift_down = True
    stub._shift_clean = True
    keyval, state = SHIFT_RELEASE
    E.PinyinEngine._handle(stub, keyval, state)
    assert ime.mode == "simp"
