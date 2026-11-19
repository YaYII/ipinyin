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
           gap_ms: int = 0, keys: int = 0, backs: int = 0, flips: int = 0,
           ms: int = 0, kind: str = "commit") -> None:
    """记一次输入事件（上屏或放弃）。任何异常都不许冒泡到输入路径。

    除上屏本身，还记这次输入的「手感」：敲了几个键、退了几次、翻了几页、
    从第一个键到上屏用了多久——这些才是判断「输入法懂不懂你」的依据。
    """
    if not enabled() or (not text and kind == "commit"):
        return
    entry = {
        "ts": round(time.time(), 3),
        "kind": kind,
        "key": key or "",
        "text": text,
        "mode": mode,
        "rank": int(rank),
        "gap_ms": int(gap_ms),
        "keys": int(keys),
        "backs": int(backs),
        "flips": int(flips),
        "ms": int(ms),
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
    return {"commits": 0, "abandons": 0, "focus_switches": 0, "backs": 0, "keystrokes": 0,
            "words": {}, "keys": {}, "modes": {}, "hours": {},
            "gaps": [], "corrections": {}, "decisions": []}


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
        decisions = profile.setdefault("decisions", [])

        for event in events:
            text = str(event.get("text") or "")
            if event.get("kind") == "abandon":
                profile["abandons"] = int(profile.get("abandons", 0)) + 1
            elif event.get("kind") == "focus":
                profile["focus_switches"] = int(profile.get("focus_switches", 0)) + 1
            else:
                profile["commits"] = int(profile.get("commits", 0)) + 1
            profile["backs"] = int(profile.get("backs", 0)) + int(event.get("backs") or 0)
            profile["keystrokes"] = int(profile.get("keystrokes", 0)) + int(event.get("keys") or 0)
            if int(event.get("ms") or 0) > 0:
                decisions.append(int(event["ms"]))
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
        profile["decisions"] = decisions[-2000:]
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
    if profile.get("commits"):
        backs = profile.get("backs", 0)
        keys = profile.get("keystrokes", 0) or 1
        lines.append(f"- 退格率：{backs / keys:.1%}（越高说明越常打错/改主意）")
        lines.append(f"- 放弃率：{profile.get('abandons', 0)} 次打到一半按 Esc")
        lines.append(f"- 焦点切换：{profile.get('focus_switches', 0)} 次（多任务切换频率）")
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


def _all_events() -> list[dict]:
    rows: list[dict] = []
    for path in (events_path(), events_path() + ".1"):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            continue
    return rows


def _lexicon_words() -> set[str]:
    """词库里的词。只在 CLI 里读一次，不进按键路径。"""
    words: set[str] = set()
    try:
        path = config.lexicon_path()
    except FileNotFoundError:
        return words
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    words.add(parts[1])
    except OSError:
        return words
    return words


TYPO_THRESHOLD = -8.5     # 平均 logp 低于它 = 语言模型也觉得不像话


def _load_lm():
    try:
        from .lm import CharLM

        return CharLM.load(os.path.join(config.data_home(), "data", "lm_char.marshal"))
    except Exception:
        return None


def typos(limit: int = 20, threshold: float = TYPO_THRESHOLD) -> list[dict]:
    """可疑上屏：语言模型打分异常低的词。

    只报「模型也觉得不像话」的，不自动改。判定依据是字级语言模型的
    平均 logp（校准过：好词 -4~-8，错字/乱码 -9 以下），词库只做辅助。
    """
    lm = _load_lm()
    if lm is None:
        return []
    words = _lexicon_words()
    hits: dict[str, dict] = {}
    for event in _all_events():
        if event.get("kind") not in (None, "commit"):
            continue
        text = str(event.get("text") or "")
        if len(text) < 2 or not _has_cjk(text):
            continue
        score = lm.logp(text) / len(text)
        # 双条件：词库不收 + 模型也觉得不像话。只用一条都会误报
        # （词库缺常用词如「很好」，模型又不认识「屏幕/线程」这类技术词）
        if score >= threshold or text in words:
            continue
        row = hits.setdefault(text, {"word": text, "count": 0, "score": round(score, 2),
                                     "in_lexicon": text in words})
        row["count"] += 1
    ranked = sorted(hits.values(), key=lambda row: (row["score"], -row["count"]))
    return ranked[:limit]


def proofread(limit: int = 120) -> str:
    """把最近的上屏文本交给大模型挑错别字（用 DSH 同一个 opencode-go 通道）。"""
    import subprocess

    texts: list[str] = []
    for event in _all_events()[-2000:]:
        text = str(event.get("text") or "")
        if event.get("kind") in (None, "commit") and len(text) >= 2 and _has_cjk(text):
            texts.append(text)
    texts = texts[-limit:]
    if not texts:
        return ""
    try:
        from . import vision

        key = vision._opencode_key()
        if not key:
            return ""
        body = {
            "model": vision.OPENCODE_MODEL,
            "max_tokens": 6000,   # 这个模型先吐推理，给小了 content 会是空
            "messages": [{"role": "user", "content": (
                "下面是某人输入法上屏的文本片段（不完整、可能只是词）。"
                "逐条找出明显的错别字、同音误用、别字。"
                "只列确实有问题的，格式：`原文 → 建议（一句话理由）`。"
                "没有问题就只回答「没有发现」。不要解释过程。\n\n"
                + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))) }],
        }
        done = subprocess.run(
            ["curl", "-sS", "--max-time", "90", "-A", "ipinyin-proofread/1.0",
             vision.OPENCODE_BASE + "/chat/completions",
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-H", f"x-opencode-session: {vision.OPENCODE_SESSION}",
             "--data-binary", "@-"],
            input=json.dumps(body), capture_output=True, text=True, timeout=120, check=False)
        if done.returncode != 0:
            return ""
        payload = json.loads(done.stdout)
        return (payload["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return ""


def _ask_llm(prompt: str, max_tokens: int = 4000) -> str:
    """走 opencode-go 通道问一次；失败返回空串。"""
    import subprocess

    try:
        from . import vision

        key = vision._opencode_key()
        if not key:
            return ""
        body = {"model": vision.OPENCODE_MODEL, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        done = subprocess.run(
            ["curl", "-sS", "--max-time", "120", "-A", "ipinyin-fix/1.0",
             vision.OPENCODE_BASE + "/chat/completions",
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-H", f"x-opencode-session: {vision.OPENCODE_SESSION}",
             "--data-binary", "@-"],
            input=json.dumps(body), capture_output=True, text=True, timeout=150, check=False)
        if done.returncode != 0:
            return ""
        payload = json.loads(done.stdout)
        return (payload["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return ""


def suggest_fixes(limit: int = 40) -> list[dict]:
    """让大模型从最近上屏里找错别字，返回 [{"wrong","right","why"}]。"""
    import re

    texts = []
    for event in _all_events()[-2000:]:
        text = str(event.get("text") or "")
        if event.get("kind") in (None, "commit") and len(text) >= 2 and _has_cjk(text):
            texts.append(text)
    texts = texts[-limit:]
    if not texts:
        return []
    # 分批：一次给太多，模型的推理会把 token 吃光、content 变空
    found: list[dict] = []
    for start in range(0, len(texts), 40):
        found.extend(_suggest_batch(texts[start:start + 40]))
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in found:
        tag = (row["wrong"], row["right"])
        if tag not in seen:
            seen.add(tag)
            unique.append(row)
    return unique


def _suggest_batch(texts: list[str]) -> list[dict]:
    import re

    prompt = (
        "下面是某人输入法上屏的文本片段。找出其中明显的错别字、同音误用。"
        "只输出 JSON 数组，元素形如 {\"wrong\":\"哪个\",\"right\":\"那个\",\"why\":\"同音误用\"}。"
        "没有错就输出 []。不要输出任何解释文字。\n\n"
        + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts)))
    raw = _ask_llm(prompt)
    if not raw:
        return []
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        rows = json.loads(match.group(0))
    except ValueError:
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        wrong, right = str(row.get("wrong") or "").strip(), str(row.get("right") or "").strip()
        if wrong and right and wrong != right and _has_cjk(right):
            out.append({"wrong": wrong, "right": right, "why": str(row.get("why") or "")})
    return out


def _pinyin_keys_for(words: set[str]) -> dict[str, list[str]]:
    """反查：词 → 它出现过的拼音键（扫一遍词库，CLI 用）。"""
    found: dict[str, list[str]] = {word: [] for word in words}
    try:
        path = config.lexicon_path()
    except FileNotFoundError:
        return found
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1] in found and parts[0] not in found[parts[1]]:
                    found[parts[1]].append(parts[0])
    except OSError:
        pass
    return found


def apply_fixes(pairs: list[dict], weight: int = 3) -> list[dict]:
    """把「正确写法」写进提权队列，由引擎应用到个人词频。

    键用**错误写法**的拼音（你敲的就是这个音），词换成正确写法——
    这样下次打同样的音，首选就是对的。
    """
    if not pairs:
        return []
    keys_map = _pinyin_keys_for({pair["wrong"] for pair in pairs} | {pair["right"] for pair in pairs})
    path = config.boost_queue_path()
    applied = []
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for pair in pairs:
                wrong, right = pair["wrong"], pair["right"]
                keys = keys_map.get(wrong) or keys_map.get(right) or []
                for key in keys[:3]:
                    handle.write(json.dumps({"key": key, "word": right, "weight": weight,
                                             "why": pair.get("why", "")}, ensure_ascii=False) + "\n")
                applied.append({"wrong": wrong, "right": right, "keys": keys[:3]})
    except OSError:
        return []
    return applied


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
