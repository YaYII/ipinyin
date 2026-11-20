"""朗读自监控：把每次朗读的耗时/语速/回落原因记成 JSONL，供 tools/tts_doctor.py 复盘。

只记摘要、不记原文——朗读内容比翻译更敏感。`sample` 只留前两个字加长度，
足够看出「哪类文本出问题」，不足以还原整句。

异常规则都是这次踩过的坑，写成判定而不是靠人记：

| 规则 | 含义 | 来源 |
| --- | --- | --- |
| `padding` | ≤3 个字却超过 1.5s | edge 把短文本补到 1.87s |
| `too-slow` | 语速 < 2 字/秒 | 同上，或单元拼接间隙过大 |
| `too-fast` | 语速 > 9 字/秒 | 合成吞字/截断 |
| `slow-synth` | 单次合成 > 8s | 网络抖动，用户会觉得「按了没反应」 |
| `fallback-espeak` | 回落到机器人音 | edge/音库都不可用 |
| `fallback-units` | 回落到词库拼接 | edge 不可用（离线） |
| `empty-audio` | 合成出空音频 | 引擎或转码失败 |
"""
from __future__ import annotations

import json
import logging
import os
import time

from . import config

LOG = logging.getLogger("ipinyin")

MAX_BYTES = 1_000_000      # 超过就轮转成 .1，只留最近一轮
SAMPLE_CHARS = 2           # 样本只留前两个字
RATE_MIN = 2.0             # 字/秒：低于这个说明补静音没裁干净
RATE_MAX = 9.0             # 高于这个说明合成吞字/截断
SHORT_TEXT = 3             # ≤3 个字算短文本
SHORT_MAX_SECONDS = 1.5    # 短文本超过这个长度 = 补静音没裁
SLOW_MS = 8000             # 单次合成超过 8 秒记异常


def metrics_path() -> str:
    return os.path.join(config.cache_home(), "tts-metrics.jsonl")


def sample(text: str) -> str:
    """只留前两个字：能看出是词还是句子，还原不了原文。"""
    return text[:SAMPLE_CHARS] + ("…" if len(text) > SAMPLE_CHARS else "")


def detect(chars: int, backend: str, synth_ms: int, audio_s: float) -> list[str]:
    """异常判定：只返回规则名，绝不抛。"""
    if audio_s <= 0:
        return ["empty-audio"]
    flags: list[str] = []
    rate = chars / audio_s if chars else 0
    if chars and chars <= SHORT_TEXT and audio_s > SHORT_MAX_SECONDS:
        flags.append("padding")
    if chars and rate < RATE_MIN:
        flags.append("too-slow")
    elif chars and rate > RATE_MAX:
        flags.append("too-fast")
    if synth_ms > SLOW_MS:
        flags.append("slow-synth")
    if backend == "espeak":
        flags.append("fallback-espeak")
    # units 现在是主力音库，不再标记为异常（以前每句都刷一条 WARNING）
    return flags


def record(text: str, backend: str, synth_ms: int, audio_s: float) -> None:
    """记一条朗读摘要；有异常时同时写一条 WARNING 到引擎日志。"""
    chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chars": chars,
        "backend": backend,
        "synth_ms": int(synth_ms),
        "audio_ms": int(audio_s * 1000),
        "rate": round(chars / audio_s, 2) if audio_s > 0 else 0.0,
        "sample": sample(text),
    }
    flags = detect(chars, backend, synth_ms, audio_s)
    if flags:
        entry["flags"] = flags
        LOG.warning("朗读异常 %s：%s", ",".join(flags), entry)
    _append(entry)


def _append(entry: dict) -> None:
    path = metrics_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > MAX_BYTES:
            os.replace(path, path + ".1")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load(path: str | None = None) -> list[dict]:
    """读回记录（坏的行走跳过，不抛）。"""
    rows: list[dict] = []
    try:
        with open(path or metrics_path(), encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows
