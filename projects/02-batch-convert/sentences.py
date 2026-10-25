"""分句：按标点断句，并把句长压到转换单位上限以内。

批量转换的单位上限默认 7 字（= 守卫严格路径阈值，见 convert.py）：
每个单位都要过「有实证才改」的校验，所以上限必须卡在它以下。
1. 终端标点（。！？…）处断句，收尾的引号/括号挂回前一句；
2. 仍超过 max_len 的，在次级标点（，、；：）处切，切点尽量靠后（前段越长，
   单段上下文越足，段数也最少）；连次级标点都没有就硬切；
3. 硬切点若落在词库词或英文/数字串中间，向左滑到边界（词库可选传入；
   滑动下限是 max_len 的四分之一，比这更长的串只能拦腰切，防退化）；
4. 不变量：「"".join(分句结果) == 原文」，输出按原样拼回即还原，测试盯死这条。
"""
from __future__ import annotations

# 神经输入 40 token 减 SOS/EOS 剩 38 字：这是物理上限，转换管线的默认单位
# 上限（7 字）由守卫阈值决定，见 convert.DEFAULT_UNIT_LEN。
DEFAULT_MAX_LEN = 36

# 终端标点：断句点。ASCII 句点只在前后都不是数字时才算（3.14 不切）。
TERMINAL = "。！？!?…"
# 次级标点：只在超长句内部找切点。ASCII 逗号在数字之间时跳过（1,000 不切）。
SECONDARY = "，、；：,;:"
# 收尾字符：终端标点后紧跟的引号/括号属于前一句。
CLOSERS = "」』）〉】\"'"


def _cut_points(text: str) -> list[int]:
    """终端标点之后的下标列表（可在此处把文本断开）。"""
    cuts: list[int] = []
    i, size = 0, len(text)
    while i < size:
        ch = text[i]
        if ch in TERMINAL:
            if ch == "." and text[i - 1:i].isdigit() and text[i + 1:i + 2].isdigit():
                i += 1
                continue
            j = i + 1
            while j < size and text[j] in CLOSERS:
                j += 1
            cuts.append(j)
            i = j
        else:
            i += 1
    return cuts


def _is_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _mid_ascii_run(seg: str, k: int) -> bool:
    """切点 k 是否落在英文/数字串中间。"""
    return k < len(seg) and _is_word_char(seg[k - 1]) and _is_word_char(seg[k])


_MAX_WORD = 8


def _mid_word(seg: str, k: int, words: set[str] | None) -> bool:
    """切点 k 是否落在词库词中间（词起点在 k 之前、终点在 k 之后）。"""
    if not words:
        return False
    for m in range(1, min(k, _MAX_WORD - 1) + 1):
        for n in range(1, _MAX_WORD - m + 1):
            if k + n > len(seg):
                break
            if seg[k - m:k + n] in words:
                return True
    return False


def _hard_cut(seg: str, max_len: int, words: set[str] | None = None) -> int:
    """硬切位置：从 max_len 起向左找第一个不切伤边界的点。

    三类边界都要护：英文/数字串、词库词；滑动下限是 max_len 的四分之一，
    比这更长的串只能拦腰切（否则会退化成逐字符切分）。
    """
    top = min(max_len, len(seg))
    floor = max(1, top // 4)
    for k in range(top, floor - 1, -1):
        if not _mid_ascii_run(seg, k) and not _mid_word(seg, k, words):
            return k
    return floor


def _split_too_long(seg: str, max_len: int, words: set[str] | None = None) -> list[str]:
    if len(seg) <= max_len:
        return [seg]
    best = -1
    for i, ch in enumerate(seg[:max_len]):
        if ch in SECONDARY:
            if ch in ",，" and text_digit_around(seg, i):
                continue
            best = i
    if best >= 0:
        head, rest = seg[:best + 1], seg[best + 1:]
    else:
        k = _hard_cut(seg, max_len, words)
        head, rest = seg[:k], seg[k:]
    return [head] + _split_too_long(rest, max_len, words)


def text_digit_around(seg: str, i: int) -> bool:
    """位置 i 的字符是否夹在两个数字之间。"""
    return seg[i - 1:i].isdigit() and seg[i + 1:i + 2].isdigit()


def split_sentences(text: str, max_len: int = DEFAULT_MAX_LEN,
                    words: set[str] | None = None) -> list[str]:
    """把一行文本切成转换单位，每段 ≤ max_len，拼接结果与原文完全一致。

    words：普通话词库（词的集合），硬切时用来避开词中间；None 表示不护界。
    """
    pieces: list[str] = []
    prev = 0
    for c in _cut_points(text):
        pieces.append(text[prev:c])
        prev = c
    pieces.append(text[prev:])
    out: list[str] = []
    for p in pieces:
        if p:
            out.extend(_split_too_long(p, max_len, words))
    return out
