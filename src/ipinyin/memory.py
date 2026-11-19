"""输入法记忆体：把每次上屏记成个人语料，空闲时聚合成画像。

采集 → `memory/events.jsonl`（一行一次上屏：时间/拼音/上屏词/模式/候选排名/间隔）
整理 → 空闲 60 秒后由后台线程批量聚合，产出 `profile.json` 与 `corrections.tsv`
导出 → `bin/ipinyin-memory export` 出 Markdown，给别的 AI 或笔记软件读

为什么不实时整理：整理要读全量统计，放在按键路径上会卡输入；空闲时做，代价为零。
密码框由 IBus 的 purpose 判定，不进入采集（密码不是语言数据，混进来只会污染词频）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from . import config

LOG = logging.getLogger("ipinyin")

MAX_EVENT_BYTES = 5_000_000   # 超过就轮转成 .1，只留最近一轮
MAX_WORDS = 500               # 画像里最多保留的词条数
MIN_WORD_LEN = 2              # 单字不进高频词（噪声太大）
STOPWORDS = set(
    "的了是我你他她它们这那就很都也和有在个上到说要去会着没看好自己知道可以什么因为所以"
    "但是而且如果还是已经现在应该需要问题时候一个没有这个那个我们你们他们"
)

_LOCK = threading.Lock()
_LEXICON: dict[str, int] | None = None


def home() -> str:
    return os.path.join(config.data_home(), "memory")


def events_path() -> str:
    return os.path.join(home(), "events.jsonl")


def profile_path() -> str:
    return os.path.join(home(), "profile.json")


def cursor_path() -> str:
    return os.path.join(home(), "cursor.json")


def corrections_path() -> str:
    return os.path.join(home(), "corrections.tsv")


def enabled() -> bool:
    """IPINYIN_MEMORY=0 可关掉采集（默认开：个人使用）。"""
    return os.environ.get("IPINYIN_MEMORY", "1") != "0"


# ---------- 采集 ----------
def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def record(key: str | None, text: str, mode: str = "simp", rank: int = 0,
           gap_ms: int = 0) -> None:
    """记一次上屏。任何异常都不许冒泡到输入路径。"""
    if not enabled() or not text:
        return
    entry = {
        "ts": round(time.time(), 3),
        "key": key or "",
        "text": text,
        "mode": mode,
        "rank": int(rank),
        "gap_ms": int(gap_ms),
    }
    try:
        with _LOCK:
            path = events_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and os.path.getsize(path) > MAX_EVENT_BYTES:
                os.replace(path, path + ".1")
                _write_cursor(0)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def pending() -> int:
    """还没被整理的字节数。"""
    try:
        size = os.path.getsize(events_path())
        return max(size - _read_cursor(), 0)
    except OSError:
        return 0


def _read_cursor() -> int:
    try:
        with open(cursor_path(), encoding="utf-8") as handle:
            return int(json.load(handle).get("offset", 0))
    except (OSError, ValueError):
        return 0


def _write_cursor(offset: int) -> None:
    try:
        os.makedirs(home(), exist_ok=True)
        with open(cursor_path(), "w", encoding="utf-8") as handle:
            json.dump({"offset": offset}, handle)
    except OSError:
        pass


# ---------- 整理 ----------
def _read_new() -> list[dict]:
    path = events_path()
    start = _read_cursor()
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            raw = handle.read()
    except OSError:
        return []
    events: list[dict] = []
    for line in raw.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    _write_cursor(start + len(raw))
    return events


def _load_profile() -> dict:
    try:
        with open(profile_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"commits": 0, "words": {}, "keys": {}, "modes": {}, "hours": {},
            "gaps": [], "corrections": {}}


def _save_profile(profile: dict) -> None:
    try:
        os.makedirs(home(), exist_ok=True)
        tmp = profile_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=0, sort_keys=True)
        os.replace(tmp, profile_path())
    except OSError:
        pass


def consolidate() -> dict:
    """把未整理的上屏事件并进画像；返回本次处理条数。"""
    with _LOCK:
        events = _read_new()
        if not events:
            return {"processed": 0}
        profile = _load_profile()
        words = profile.setdefault("words", {})
        keys = profile.setdefault("keys", {})
        modes = profile.setdefault("modes", {})
        hours = profile.setdefault("hours", {})
        corrections = profile.setdefault("corrections", {})
        gaps = profile.setdefault("gaps", [])

        for event in events:
            text = str(event.get("text") or "")
            profile["commits"] = int(profile.get("commits", 0)) + 1
            modes[str(event.get("mode") or "?")] = modes.get(str(event.get("mode") or "?"), 0) + 1
            key = str(event.get("key") or "")
            if key:
                keys[key] = keys.get(key, 0) + 1
            if len(text) >= MIN_WORD_LEN and text not in STOPWORDS and _has_cjk(text):
                words[text] = words.get(text, 0) + 1
            hour = time.strftime("%H", time.localtime(float(event.get("ts") or 0)))
            hours[hour] = hours.get(hour, 0) + 1
            gap = int(event.get("gap_ms") or 0)
            if 0 < gap < 60_000:
                gaps.append(gap)
            rank = int(event.get("rank") or 0)
            if rank >= 1 and key:
                # 首选不是你想要的：这条最能说明输入法哪里不准
                ckey = f"{key}\t{text}"
                corrections[ckey] = corrections.get(ckey, 0) + 1

        # 只留最大的若干条，避免文件无限长
        profile["words"] = dict(sorted(words.items(), key=lambda kv: -kv[1])[:MAX_WORDS])
        profile["keys"] = dict(sorted(keys.items(), key=lambda kv: -kv[1])[:MAX_WORDS])
        profile["corrections"] = dict(
            sorted(corrections.items(), key=lambda kv: -kv[1])[:MAX_WORDS]
        )
        profile["gaps"] = gaps[-2000:]
        profile["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_profile(profile)
        _write_corrections(profile["corrections"])
        return {"processed": len(events), "commits": profile["commits"]}


def _write_corrections(corrections: dict[str, int]) -> None:
    try:
        os.makedirs(home(), exist_ok=True)
        with open(corrections_path(), "w", encoding="utf-8") as handle:
            handle.write("# 拼音\t你选的词\t次数（首选不是你想要的）\n")
            for ckey, count in sorted(corrections.items(), key=lambda kv: -kv[1]):
                handle.write(f"{ckey}\t{count}\n")
    except OSError:
        pass


# ---------- 导出 ----------
def _lexicon_freq() -> dict[str, int]:
    """通用词频，用来挑「领域词」（在你的输入里常见、在通用语料里罕见）。"""
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    freq: dict[str, int] = {}
    try:
        with open(os.path.join(os.path.dirname(config.lexicon_path()), "lexicon.tsv"),
                  encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[2].isdigit():
                    freq[parts[1]] = max(freq.get(parts[1], 0), int(parts[2]))
    except OSError:
        pass
    _LEXICON = freq
    return freq


def export_markdown(top: int = 50) -> str:
    profile = _load_profile()
    words = profile.get("words", {})
    corrections = profile.get("corrections", {})
    gaps = profile.get("gaps", [])
    general = _lexicon_freq()
    common_top = {word for word, _ in sorted(general.items(), key=lambda kv: -kv[1])[:3000]}

    lines = ["# 我的输入画像", ""]
    lines.append(f"- 上屏次数：{profile.get('commits', 0)}")
    lines.append(f"- 最近整理：{profile.get('updated', '（还没整理过）')}")
    if gaps:
        gaps_sorted = sorted(gaps)
        lines.append(f"- 输入节奏：中位间隔 {gaps_sorted[len(gaps_sorted) // 2]}ms")
    lines.append("")

    lines.append(f"## 高频词 Top {top}")
    lines.append("")
    lines.append("| 词 | 次数 |")
    lines.append("| --- | --- |")
    for word, count in list(words.items())[:top]:
        lines.append(f"| {word} | {count} |")
    lines.append("")

    domain = [(w, c) for w, c in words.items() if c >= 3 and w not in common_top][:top]
    if domain:
        lines.append("## 领域词（你的输入里常见，通用语料里罕见）")
        lines.append("")
        lines.append("| 词 | 次数 |")
        lines.append("| --- | --- |")
        for word, count in domain:
            lines.append(f"| {word} | {count} |")
        lines.append("")

    if corrections:
        lines.append("## 首选不准（你越过了默认候选）")
        lines.append("")
        lines.append("| 拼音 | 你选的词 | 次数 |")
        lines.append("| --- | --- | --- |")
        for ckey, count in list(corrections.items())[:top]:
            key, _, word = ckey.partition("\t")
            lines.append(f"| {key} | {word} | {count} |")
        lines.append("")
    return "\n".join(lines)


def wipe() -> None:
    """删掉所有采集与整理结果。"""
    import shutil

    shutil.rmtree(home(), ignore_errors=True)


def status() -> dict:
    profile = _load_profile()
    return {
        "enabled": enabled(),
        "home": home(),
        "commits": profile.get("commits", 0),
        "words": len(profile.get("words", {})),
        "corrections": len(profile.get("corrections", {})),
        "pending_bytes": pending(),
        "updated": profile.get("updated", ""),
    }
