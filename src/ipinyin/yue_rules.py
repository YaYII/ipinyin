"""普通话 → 粤语 的语序规则（词表换不出来的那部分）。

粤语常把状语、宾语后置：
  我先走        → 我走先
  给你一本书    → 畀本书你
  他比你高      → 他高过你
  正在吃饭      → 喺度食饭
规则在「词表替换之前」作用在普通话语序上，之后再做词级映射。
"""
from __future__ import annotations

import re

PRONOUNS = "我|你|他|她|我们|你们|他们|大家"

# (正则, 替换)。顺序从具体到宽泛，先匹配的先生效。
RULES: list[tuple[re.Pattern[str], str]] = [
    # 给 + 人 + 数量词 + 东西  →  畀 + 东西 + 人
    (re.compile(rf"给(?P<who>{PRONOUNS})(?P<num>[一二三四五六七八九十两几]?)(?P<mw>本|个|张|件|只|条|份|杯|碗|瓶|把|支|盒)(?P<obj>[^，。！？；]{{0,6}})"),
     r"畀\g<num>\g<mw>\g<obj>\g<who>"),
    # 给 + 东西 + 人  →  畀 + 东西 + 人
    (re.compile(rf"给(?P<obj>[^，。！？；]{{1,10}}?)(?P<who>{PRONOUNS})"),
     r"畀\g<obj>\g<who>"),
    # 先 + 动词  →  动词 + 先（主语后）
    (re.compile(rf"(?P<subj>{PRONOUNS})先(?P<verb>[\u4e00-\u9fff]{{1,4}})"),
     r"\g<subj>\g<verb>先"),
    # 比 + 人 + 形容词  →  形容词 + 过 + 人
    (re.compile(rf"(?P<a>[\u4e00-\u9fff]{{1,4}})比(?P<b>{PRONOUNS})(?P<adj>高|矮|大|小|多|少|快|慢|好|早|晚|贵|便宜|靓)"),
     r"\g<a>\g<adj>过\g<b>"),
    # 在 + 动词  →  喺度 + 动词（我在吃饭 → 我喺度食饭）
    # 词表先把「在」换成「喺」，所以这里两种写法都要认
    (re.compile(rf"(?P<subj>{PRONOUNS})[在喺](?P<v>吃|喝|看|说|想|做|写|读|玩|睡|洗|买|卖|坐|等|找|给|拿|听|唱|跳|笑|哭|学|教|打|开|关|用|穿|脱|戴|修|搬|煮|炒|食|飲|睇|講|諗|搵|攞|畀|行|企|瞓|沖|買|賣)"),
     r"\g<subj>喺度\g<v>"),
    # 了吗 → 未（你吃饭了吗 → 你食饭未）；必须在「了」之前
    (re.compile(r"了吗(?=[，。！？；]|$)"), "未"),
    # 句末「了」→ 喇（他回家了 → 佢返屋企喇）
    (re.compile(r"了(?=[，。！？；]|$)"), "喇"),
    # 正在 + 动词  →  喺度 + 动词
    (re.compile(r"正在(?P<v>[\u4e00-\u9fff]{1,4})"), r"喺度\g<v>"),
    # 太 + 形容词 + 了  →  太 + 形容词 + 喇
    (re.compile(r"太(?P<x>[\u4e00-\u9fff]{1,3})了"), r"太\g<x>喇"),
    # 给 + 人 + 打电话/发消息  →  打电话/发消息 + 畀 + 人
    (re.compile(rf"给(?P<who>{PRONOUNS})(?P<v>打电话|打電话|发消息|发信息|传话)"), r"\g<v>畀\g<who>"),
    # 一下  →  下（睇一下 → 睇下）
    (re.compile(r"一下(?![儿])"), "下"),
    # 才 + 动词  →  先 + 动词
    (re.compile(r"才(?P<v>[\u4e00-\u9fff]{1,3})"), r"先\g<v>"),
]


def apply(text: str) -> str:
    """按顺序套用规则；只做语序调整，不做用字替换。"""
    out = text
    for pattern, replacement in RULES:
        out = pattern.sub(replacement, out, count=1)
    return out
