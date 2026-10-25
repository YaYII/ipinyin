#!/usr/bin/env python3
"""粤拼（Jyutping）音素清单：从 data/yue.marshal 提取声母/韵母/声调集合，
并给出「粤拼符号 → Piper 风格音素（IPA）」的占位映射。

拆分规则（确定性的，无歧义）：
  一个粤拼音节 = 声母(initial) + 韵母(final) + 声调(tone 数字 1-6)
  1. 去掉末尾声调数字得到 core。
  2. core == "m" 或 "ng"（整个音节就一个鼻音）→ 成音节鼻音：无声母，韵母 = core。
  3. 否则按 INITIALS 从长到短匹配前缀（gw/kw/ng 两个字母优先于 b/d/g 等）：
     前缀是声母，余下部分是韵母。Jyutping 的韵母不以任何声母字母开头
     （成音节 m/ng 已在规则 2 排除），所以贪心匹配无歧义。
     匹配不到声母 → 零声母，整个 core 就是韵母（aa、jyu、oeng 这类）。
  4. 韵母必须落在 FINAL_SET 里；越界视为坏音节（正常数据不会出现，防御用）。

韵母拼写的拆法要点（对 aa/a 这类易混对）：
  aa 系（长 a）：aa aai aau aam aan aang aap aat aak —— 都是 aa 开头
  a  系（短 ɐ）：a ai au am an ang ap at ak     —— 韵腹直接跟韵尾/介音
  -ng 结尾是鼻韵尾（韵母的一部分），与成音节鼻音 ng5（规则 2）区分：
     ngo5 = ng + o（声母 ng），ong6 = 空 + ong（韵尾 ng）
  入声韵尾 -p/-t/-k 是「唯闭音」，IPA 用 p̚ t̚ k̚（上标拼音字母表示无除阻）。

声调：Jyutping 六调（本表数据就是 1-6，没有把入声单列 7-9 的高平/高中/低入变体）。
Piper 占位方案：声调作为独立 token（T1..T6）附在音节后面，训练时要写进 phoneme_id_map；
等真的开始训声学模型（下一步，不在本轮），可以换成 Chao 声调字母或直接进嵌入。

用法：
  python3 projects/03-tts-g2p/phonemes.py                 # 打印清单摘要
  python3 projects/03-tts-g2p/phonemes.py --write          # 生成 phoneme_inventory.json
"""
from __future__ import annotations

import argparse
import json
import marshal
import os
import re
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YUE_MARSHAL = os.path.join(ROOT, "data", "yue.marshal")

# ---- 声母（19 个 + 零声母）----------------------------------------------
INITIALS: list[str] = [
    "gw", "kw", "ng",                      # 双字母优先匹配
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "w", "z", "c", "s", "j",
]

# ---- 韵母（59 个，全部实测来自 data/yue.marshal）--------------------------
FINAL_SET: frozenset[str] = frozenset({
    # aa 系（长 aː）
    "aa", "aai", "aau", "aam", "aan", "aang", "aap", "aat", "aak",
    # a 系（短 ɐ）
    "a", "ai", "au", "am", "an", "ang", "ap", "at", "ak",
    # e 系
    "e", "ei", "eu", "em", "eng", "ep", "et", "ek",
    # i 系
    "i", "iu", "im", "in", "ing", "ip", "it", "ik",
    # o 系
    "o", "oi", "ou", "on", "ong", "ot", "ok",
    # u 系
    "u", "ui", "un", "ung", "ut", "uk",
    # oe / eoi 系
    "oe", "oeng", "oek", "oet", "eoi", "eon", "eot",
    # yu 系
    "yu", "yun", "yut",
    # 成音节鼻音
    "m", "ng",
})

TONES: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

SYLLABLE_RE = re.compile(r"^([a-z]+)([1-6])$")

# ---- 粤拼 → IPA 映射（Piper 风格占位方案）--------------------------------
IPA_INITIAL: dict[str, str] = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f",
    "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "ng": "ŋ", "h": "h",
    "gw": "kʷ", "kw": "kʷʰ", "w": "w",
    "z": "ts", "c": "tsʰ", "s": "s", "j": "j",
    "": "",                                   # 零声母
}

IPA_FINAL: dict[str, str] = {
    "aa": "aː", "aai": "aːi", "aau": "aːu", "aam": "aːm", "aan": "aːn",
    "aang": "aːŋ", "aap": "aːp̚", "aat": "aːt̚", "aak": "aːk̚",
    "a": "ɐ", "ai": "ɐi", "au": "ɐu", "am": "ɐm", "an": "ɐn",
    "ang": "ɐŋ", "ap": "ɐp̚", "at": "ɐt̚", "ak": "ɐk̚",
    "e": "ɛː", "ei": "ei", "eu": "ɛːu", "em": "ɛːm", "eng": "ɛːŋ",
    "ep": "ɛːp̚", "et": "ɛt̚", "ek": "ɛːk̚",
    "i": "iː", "iu": "iːu", "im": "iːm", "in": "iːn", "ing": "ɪŋ",
    "ip": "iːp̚", "it": "iːt̚", "ik": "ɪk̚",
    "o": "ɔː", "oi": "ɔːy", "ou": "ou", "on": "ɔːn", "ong": "ɔːŋ",
    "ot": "ɔːt̚", "ok": "ɔːk̚",
    "u": "uː", "ui": "uːy", "un": "uːn", "ung": "ʊŋ", "ut": "uːt̚",
    "uk": "ʊk̚",
    "oe": "œː", "oeng": "œːŋ", "oek": "œːk̚", "oet": "ɵt̚",
    "eoi": "ɵy", "eon": "ɵn", "eot": "ɵt̚",
    "yu": "yː", "yun": "yːn", "yut": "yːt̚",
    "m": "m̩", "ng": "ŋ̩",
}

# Chao 五度标调字母（占位；训练时通常只需要 T1..T6 这个离散标记）
TONE_IPA: dict[str, str] = {
    "1": "˥",        # 55 高平（旧派 53 高降变体不区分，TODO）
    "2": "˨˥",       # 25 高升
    "3": "˧",        # 33 中平
    "4": "˨˩",       # 21 低降
    "5": "˩˧",       # 13 低升
    "6": "˨",        # 22 低平
}


def split_syllable(syllable: str) -> tuple[str, str, str] | None:
    """把一个粤拼音节拆成 (声母, 韵母, 声调)。坏音节返回 None。

    例：aa1→("", "aa", "1")  ngo5→("ng", "o", "5")
        ng5→("", "ng", "5")  gwong2→("gw", "ong", "2")
    """
    match = SYLLABLE_RE.match(syllable)
    if not match:
        return None
    core, tone = match.group(1), match.group(2)
    if core in ("m", "ng"):                   # 成音节鼻音：整个音节就是一个鼻音
        return ("", core, tone)
    for initial in INITIALS:
        if core.startswith(initial):
            final = core[len(initial):]
            if final in FINAL_SET:
                return (initial, final, tone)
            return None                       # 声母对但韵母非法（防御）
    if core in FINAL_SET:                     # 零声母
        return ("", core, tone)
    return None


def syllable_to_tokens(syllable: str) -> list[str] | None:
    """一个粤拼音节 → Piper 风格音素 token 序列（声母 IPA + 韵母 IPA + T声调）。

    例：ngo5 → ["ŋ", "o", "T5"]；aa1 → ["aː", "T1"]
    声母为空（零声母）时不发空 token。坏音节返回 None（调用方自行报告缺字）。
    """
    parts = split_syllable(syllable)
    if parts is None:
        return None
    initial, final, tone = parts
    tokens: list[str] = []
    ipa = IPA_INITIAL[initial]
    if ipa:
        tokens.append(ipa)
    tokens.append(IPA_FINAL[final])
    tokens.append("T" + tone)
    return tokens


def _iter_entries(data: dict):
    yield from data.get("chars", {}).items()
    yield from data.get("words", {}).items()


def build_inventory(path: str = YUE_MARSHAL) -> dict:
    """扫全表统计实际出现的声母/韵母/声调，并核对面上的集合。"""
    raw = open(path, "rb").read()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass                                  # 兼容未压缩的旧文件
    data = marshal.loads(raw)

    initials_seen: set[str] = set()
    finals_seen: set[str] = set()
    tones_seen: set[str] = set()
    syllables = 0
    bad: list[str] = []
    for _key, reading in _iter_entries(data):
        for syl in reading.split():
            parts = split_syllable(syl)
            if parts is None:
                bad.append(syl)
                continue
            syllables += 1
            initials_seen.add(parts[0])
            finals_seen.add(parts[1])
            tones_seen.add(parts[2])

    initials_seen.discard("")                 # 零声母不进清单
    return {
        "syllables_scanned": syllables,
        "bad_syllables": sorted(set(bad)),
        "initials": sorted(initials_seen),
        "finals": sorted(finals_seen),
        "tones": sorted(tones_seen),
        "ipa_initial": {k: v for k, v in IPA_INITIAL.items() if k},
        "ipa_final": dict(IPA_FINAL),
        "tone_ipa": dict(TONE_IPA),
        "finals_split_rules": (
            "syllable = initial + final + tone-digit; "
            "core m/ng alone = syllabic nasal (no initial); "
            "longest-prefix match over initials (gw/kw/ng before b/g/...); "
            "aa-series are long aː, a-series are short ɐ; "
            "-ng ends nasal coda (part of final), vs ng5 syllabic; "
            "-p/-t/-k are unreleased stops (IPA p̚ t̚ k̚)"
        ),
        "piper_placeholder": (
            "one syllable -> [initial_ipa?, final_ipa, T<tone>]; "
            "add every token below to phoneme_id_map when training"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="粤拼音素清单")
    ap.add_argument("--write", action="store_true", help="写出 phoneme_inventory.json")
    args = ap.parse_args()

    inv = build_inventory()
    print(f"扫描音节     : {inv['syllables_scanned']}")
    print(f"坏音节       : {inv['bad_syllables'] or '无'}")
    print(f"声母         : {len(inv['initials'])} 个 -> {' '.join(inv['initials'])}")
    print(f"韵母         : {len(inv['finals'])} 个")
    print(f"声调         : {len(inv['tones'])} 个 -> {' '.join(inv['tones'])}")
    print(f"韵母全集校验 : FINAL_SET {len(FINAL_SET)} 项，实测 {len(inv['finals'])} 项，"
          f"差集 {sorted(FINAL_SET - set(inv['finals'])) or '无'}")
    if args.write:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "phoneme_inventory.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(inv, fh, ensure_ascii=False, indent=2)
        print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
