"""按键映射测试：确认「什么键会被输入法吃掉、什么键会还给应用」。"""
import pytest

from ipinyin import keys


@pytest.mark.parametrize(
    "keyval, expected",
    [
        (0x6E, keys.Action.LETTER),   # n
        (0x4E, keys.Action.LETTER),   # N（Shift+n）
        (0x31, keys.Action.DIGIT),    # 1
        (0x20, keys.Action.COMMIT),   # space
        (0xFF0D, keys.Action.COMMIT_RAW),  # Return
        (0xFF08, keys.Action.BACKSPACE),
        (0xFF1B, keys.Action.CLEAR),
        (0xFF55, keys.Action.PAGE_PREV),
        (0x2D, keys.Action.PAGE_PREV),
        (0xFF56, keys.Action.PAGE_NEXT),
        (0x3D, keys.Action.PAGE_NEXT),
        (0x5B, keys.Action.PAGE_PREV),      # [
        (0x5D, keys.Action.PAGE_NEXT),      # ]
        (0x7B, keys.Action.PAGE_PREV),      # {
        (0x7D, keys.Action.PAGE_NEXT),      # }
        (0xFF52, keys.Action.PREV),
        (0xFF54, keys.Action.NEXT),
        (0x2C, keys.Action.PASSTHROUGH),  # 逗号：输入中先上屏再交给应用
        (0x2E, keys.Action.PASSTHROUGH),  # 句号
        (0x27, keys.Action.LETTER),       # 分隔符
    ],
)
def test_translate(keyval, expected):
    action, _ = keys.translate(keyval, 0)
    assert action is expected


def test_modifier_keys_are_ignored():
    for mask in (keys.CONTROL_MASK, keys.MOD1_MASK, keys.SUPER_MASK):
        assert keys.translate(0x6E, mask)[0] is keys.Action.IGNORE


def test_release_events_are_ignored():
    assert keys.translate(0x6E, keys.RELEASE_MASK)[0] is keys.Action.IGNORE
