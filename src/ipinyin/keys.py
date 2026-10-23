"""IBus 按键 → 输入法动作。纯函数，方便单测。"""
from __future__ import annotations

from enum import Enum, auto

# IBus/X11 键值常量
KEY_BACKSPACE = 0xFF08
KEY_RETURN = 0xFF0D
KEY_KP_ENTER = 0xFF8D
KEY_ESCAPE = 0xFF1B
KEY_PAGE_UP = 0xFF55
KEY_PAGE_DOWN = 0xFF56
KEY_UP = 0xFF52
KEY_DOWN = 0xFF54
KEY_LEFT = 0xFF51
KEY_RIGHT = 0xFF53
KEY_SPACE = 0x20
KEY_TAB = 0xFF09
KEY_APOSTROPHE = 0x27
KEY_MINUS = 0x2D
KEY_EQUAL = 0x3D
KEY_BRACKET_LEFT = 0x5B    # [
KEY_BRACKET_RIGHT = 0x5D   # ]
KEY_BRACE_LEFT = 0x7B      # {
KEY_BRACE_RIGHT = 0x7D     # }

RELEASE_MASK = 1 << 30
SHIFT_MASK = 1 << 0
CONTROL_MASK = 1 << 2
MOD1_MASK = 1 << 3
SUPER_MASK = 1 << 26


class Action(Enum):
    IGNORE = auto()
    LETTER = auto()
    DIGIT = auto()
    BACKSPACE = auto()
    COMMIT = auto()      # 空格：提交首选
    COMMIT_RAW = auto()  # 回车：提交原始拼音
    CLEAR = auto()       # Esc：清空
    PAGE_PREV = auto()
    PAGE_NEXT = auto()
    PREV = auto()       # 上下：移候选
    NEXT = auto()
    SEG_PREV = auto()   # 左右：切拼音段（你/nihao/nihaoma）
    SEG_NEXT = auto()
    TOGGLE_TRAD = auto()
    PASSTHROUGH = auto()  # 输入中遇到标点：先提交再交给应用


def translate(keyval: int, state: int) -> tuple[Action, str]:
    if state & RELEASE_MASK:
        return Action.IGNORE, ""
    if state & (CONTROL_MASK | MOD1_MASK | SUPER_MASK):
        return Action.IGNORE, ""

    # 字母 a-z / A-Z
    if 0x61 <= keyval <= 0x7A:
        return Action.LETTER, chr(keyval)
    if 0x41 <= keyval <= 0x5A:
        return Action.LETTER, chr(keyval).lower()
    # 数字 1-9（0 留给标点/空格场景）
    if 0x31 <= keyval <= 0x39:
        return Action.DIGIT, chr(keyval)
    if keyval == KEY_APOSTROPHE:
        return Action.LETTER, "'"

    if keyval == KEY_SPACE:
        return Action.COMMIT, ""
    if keyval in (KEY_RETURN, KEY_KP_ENTER):
        return Action.COMMIT_RAW, ""
    if keyval == KEY_BACKSPACE:
        return Action.BACKSPACE, ""
    if keyval == KEY_TAB:
        return Action.TOGGLE_TRAD, ""
    if keyval == KEY_ESCAPE:
        return Action.CLEAR, ""

    if keyval in (KEY_PAGE_UP, KEY_MINUS, KEY_BRACKET_LEFT, KEY_BRACE_LEFT):
        return Action.PAGE_PREV, ""
    if keyval in (KEY_PAGE_DOWN, KEY_EQUAL, KEY_BRACKET_RIGHT, KEY_BRACE_RIGHT):
        return Action.PAGE_NEXT, ""
    if keyval == KEY_UP:
        return Action.PREV, ""
    if keyval == KEY_DOWN:
        return Action.NEXT, ""
    if keyval == KEY_LEFT:
        return Action.SEG_PREV, ""
    if keyval == KEY_RIGHT:
        return Action.SEG_NEXT, ""

    # 可见 ASCII 标点/符号：输入中先提交候选，再让应用自己处理
    if 0x21 <= keyval <= 0x7E:
        return Action.PASSTHROUGH, chr(keyval)
    return Action.IGNORE, ""
