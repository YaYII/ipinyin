"""按键映射测试：确认「什么键会被输入法吃掉、什么键会还给应用」。"""
import pytest

from ipinyin import keys


@pytest.mark.parametrize(
    "keyval, expected",
    [
        (0x6E, keys.Action.LETTER),   # n
        (0x4E, keys.Action.PASSTHROUGH),  # N（Shift+n）：大写原样交给应用
        (0x31, keys.Action.DIGIT),    # 1
        (0x20, keys.Action.COMMIT),   # space
        (0xFF0D, keys.Action.COMMIT_RAW),  # Return
        (0xFF08, keys.Action.BACKSPACE),
        (0xFF1B, keys.Action.CLEAR),
        (0xFF55, keys.Action.PAGE_PREV),
        (0x2D, keys.Action.PAGE_PREV),
        (0xFF56, keys.Action.PAGE_NEXT),
        (0xFFC6, keys.Action.TOGGLE_TTS),   # F9：上屏即读 开/关
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


def test_uppercase_goes_to_the_application():
    """Shift+字母 / Caps Lock 出来的大写必须原样交给应用——打 API/SQL 要能出大写。"""
    action, value = keys.translate(0x41, keys.SHIFT_MASK)   # Shift+A
    assert action is keys.Action.PASSTHROUGH
    assert value == "A"
    # 小写仍是拼音
    assert keys.translate(0x61, 0)[0] is keys.Action.LETTER


def test_f10_is_ai_toggle():
    """F10 与 F9 对称：F9 管朗读，F10 管 AI 屏幕感知。"""
    assert keys.translate(0xFFC7, 0)[0] is keys.Action.TOGGLE_AI
    assert keys.translate(0xFFC6, 0)[0] is keys.Action.TOGGLE_TTS
