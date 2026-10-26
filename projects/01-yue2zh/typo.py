#!/usr/bin/env python3
"""同音错字增强：用 data/yue.marshal 的字级粤拼表构造「同音异形字」替换。

动机（项目① README 已知局限 + 社区实践 AlienKevin/bart-canto-mando-bing-300K-typo）：
粤语网络语料里大量同音别字（「细路」写成「係路」），端到端模型没见过错字分布
就会被「忠实」译错。训练时对粤语（src）侧按字符级概率注入同音字，让模型学到
「同音错字也映射到同一个普通话词」。

分组口径：去掉粤拼尾部声调数字，声母+韵母相同即同音（真实错字常跨声调混用）。
纯函数、不依赖 torch；data/yue.marshal 的加载方式对齐 src/ipinyin/yue.py::load_dict
（zlib 压缩的 marshal，兼容未压缩旧文件）。
"""
from __future__ import annotations

import marshal
import random
import re
import zlib

_TONE_RE = re.compile(r"\d+$")


def is_han(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def load_yue_chars(path: str) -> dict[str, str]:
    """读 yue.marshal 的字级粤拼表 {字: 粤拼}；失败返回空表（增强自动失效）。"""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass                     # 兼容未压缩的旧文件
        data = marshal.loads(raw)
    except (OSError, ValueError, EOFError):
        return {}
    return data.get("chars", {}) if isinstance(data, dict) else {}


def strip_tone(jyutping: str) -> str:
    """去尾部声调数字：'lei5' → 'lei'，'ngo4' → 'ngo'。"""
    return _TONE_RE.sub("", jyutping)


def build_homophones(chars: dict[str, str],
                     allowed: set[str] | None = None) -> dict[str, tuple[str, ...]]:
    """字 → 同音异形字元组。

    allowed：候选字白名单（训练时传 src 词表，避免注入未登录字产生 UNK 噪声）。
    只要有同音字就保留条目；调用方按需再用。
    """
    groups: dict[str, list[str]] = {}
    for ch, jp in chars.items():
        syl = strip_tone(jp)
        if syl:
            groups.setdefault(syl, []).append(ch)
    out: dict[str, tuple[str, ...]] = {}
    for ch, jp in chars.items():
        if allowed is not None and ch not in allowed:
            continue          # 键字也须在白名单内，否则会把 UNK 字「修好」，注入方向就反了
        syl = strip_tone(jp)
        if not syl:
            continue
        cands = [c for c in groups[syl]
                 if c != ch and (allowed is None or c in allowed)]
        if cands:
            out[ch] = tuple(cands)
    return out


def inject_typos(text: str, homophones: dict[str, tuple[str, ...]],
                 rng: random.Random, p: float = 0.12) -> str:
    """每个汉字独立以概率 p 换成随机同音字；非汉字 / 无同音字 / p<=0 时原样。"""
    if p <= 0 or not homophones:
        return text
    out: list[str] = []
    for ch in text:
        cands = homophones.get(ch)
        if cands and is_han(ch) and rng.random() < p:
            out.append(cands[rng.randrange(len(cands))])
        else:
            out.append(ch)
    return "".join(out)
