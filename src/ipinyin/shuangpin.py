"""小鹤双拼（flypy）：每个音节两键，映射表与 Rime 的 double_pinyin_flypy 一致。

声母：zh→v、ch→i、sh→u，其余同全拼；零声母用 a/o/e 自身当声母。
韵母：iu→q ei→w uan→r ve/üe→t un→y uo→o ie→p ong/iong→s ing/uai→k
      ai→d en→f eng→g iang/uang→l ang→h ian→m an→j ou→z ia/ua→x iao→n
      ao→c ui→v in→b er→r
"""
from __future__ import annotations

INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
            "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")
INITIAL_KEY = {"zh": "v", "ch": "i", "sh": "u"}

FINALS = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "d", "ei": "w", "ui": "v", "ao": "c", "ou": "z", "iu": "q",
    "ie": "p", "ve": "t", "ue": "t", "er": "r",
    "an": "j", "en": "f", "in": "b", "un": "y", "vn": "y",
    "ang": "h", "eng": "g", "ing": "k", "ong": "s", "iong": "s",
    "ia": "x", "ua": "x", "uo": "o", "uai": "k",
    "ian": "m", "uan": "r", "van": "r",
    "iang": "l", "uang": "l", "iao": "n",
}

# 反向表：两键码 → 音节（同码多音节时，按此顺序取；全拼链路会再按词频挑词）
_DECODE: dict[str, list[str]] = {}


def _build_decode() -> None:
    if _DECODE:
        return
    for final, key in FINALS.items():
        for initial in INITIALS:
            syllable = initial + final
            code = INITIAL_KEY.get(initial, initial) + key
            _DECODE.setdefault(code, []).append(syllable)
    # 零声母也接受"原样输入"（爱 ai、安 an 都能打）
    for final in FINALS:
        if final[0] in "aoe" and 1 <= len(final) <= 3:
            _DECODE.setdefault(final, []).append(final)
    # 零声母：a/o/e + 韵母
    for final, key in FINALS.items():
        for zero in ("a", "o", "e"):
            if not final.startswith(zero):
                continue
            code = zero + key
            _DECODE.setdefault(code, []).append(final)


def split_syllable(syllable: str) -> tuple[str, str]:
    for initial in INITIALS:
        if syllable.startswith(initial) and len(syllable) > len(initial):
            return initial, syllable[len(initial):]
    return "", syllable


def encode(syllable: str) -> str | None:
    initial, final = split_syllable(syllable)
    if not initial:
        if not syllable or syllable[0] not in "aoe":
            return None
        if syllable in FINALS:                  # 安 an → aj、爱 ai → ai
            return syllable[0] + FINALS[syllable]
        if len(syllable) == 1:                  # 啊 a → aa
            return syllable + syllable
        initial, final = syllable[0], syllable[1:]
        key = FINALS.get(final)
        return initial + key if key else None
    key = FINALS.get(final)
    if key is None:
        return None
    return INITIAL_KEY.get(initial, initial) + key


def decode(code: str) -> list[str]:
    _build_decode()
    return _DECODE.get(code, [])


def to_full(text: str) -> str:
    """把双拼串还原成全拼（两键一音节）；解不出的部分原样保留。"""
    out: list[str] = []
    i = 0
    while i + 1 < len(text):
        candidates = decode(text[i:i + 2])
        if not candidates:
            out.append(text[i])
            i += 1
            continue
        out.append(candidates[0])
        i += 2
    if i < len(text):
        out.append(text[i])
    return "".join(out)
