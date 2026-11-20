#!/usr/bin/env python3
"""预生成词级音库：按词频选词 → edge-tts 合成 → Opus 单元 + index.json。

单元只生成一次，之后朗读完全离线（见 src/ipinyin/units.py）。
文件名是词哈希，某个词读错时删掉对应 .opus 重跑即可，不必重建全库。

用法：
    python3 tools/build_tts_units.py                    # 默认 3000 词 + 800 字
    python3 tools/build_tts_units.py --limit 5000 --workers 6
    python3 tools/build_tts_units.py --out /tmp/units   # 先小规模试
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin import config, t2s  # noqa: E402

VOICE = "zh-HK-WanLungNeural"     # 粤语男声：声调准，且不需要情感
OPUS_BITRATE = "24k"
DEFAULT_OUT = os.path.expanduser("~/.local/share/ipinyin/tts-units")


def load_freq() -> dict[str, int]:
    """data/lexicon.tsv：拼音键 \\t 词 \\t 频次（简体）。"""
    freq: dict[str, int] = {}
    path = os.path.join(ROOT, "data", "lexicon.tsv")
    if not os.path.exists(path):
        return freq
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                n = int(parts[2])
            except ValueError:
                continue
            word = parts[1]
            if n > freq.get(word, 0):
                freq[word] = n
    return freq


def load_yue() -> tuple[dict[str, str], dict[str, str]]:
    import zlib
    import marshal

    raw = open(os.path.join(ROOT, "data", "yue.marshal"), "rb").read()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    data = marshal.loads(raw)
    return data.get("chars", {}) or {}, data.get("words", {}) or {}


def corpus_freq(words: dict[str, str], chars: dict[str, str]):
    """用粤语语料统计词频/字频——比通用词频准得多。

    vendor/yue_pairs.json 是 16.9 万条「普通话-粤语」句对，粤语侧就是现成语料。
    实测：按通用词频选 3000 词只能覆盖日常句子 11%，按语料词频能到 78%，
    再配 1500 个高频字兜底 → 100%。
    """
    import collections

    path = os.path.join(ROOT, "vendor", "yue_pairs.json")
    word_cnt: collections.Counter = collections.Counter()
    char_cnt: collections.Counter = collections.Counter()
    if not os.path.exists(path):
        return word_cnt, char_cnt
    import json as _json

    try:
        with open(path, encoding="utf-8") as fh:
            pairs = _json.load(fh)
    except (OSError, ValueError):
        return word_cnt, char_cnt
    maxlen = max((len(w) for w in words), default=1)
    for pair in pairs:
        text = pair.get("yue") or ""
        i = 0
        while i < len(text):
            ch = text[i]
            if "\u4e00" <= ch <= "\u9fff":
                char_cnt[ch] += 1
            for n in range(min(maxlen, len(text) - i), 0, -1):
                piece = text[i:i + n]
                if piece in words:
                    word_cnt[piece] += 1
                    i += n
                    break
            else:
                i += 1
    return word_cnt, char_cnt


def pick_units(limit: int, char_limit: int) -> list[str]:
    """语料词频选词 + 高频字兜底；语料缺失时退回通用词频。"""
    chars, words = load_yue()
    word_cnt, char_cnt = corpus_freq(words, chars)

    if not word_cnt:                       # 没语料：退回通用词频（覆盖会差）
        conv = t2s.Converter.load(config.s2t_path())
        for word, n in load_freq().items():
            trad = conv.to_traditional(word)
            if trad in words:
                word_cnt[trad] = max(word_cnt[trad], n)
            if len(word) == 1 and trad in chars:
                char_cnt[trad] = max(char_cnt[trad], n)

    picked = [w for w, _ in word_cnt.most_common(limit)]
    picked.extend(ch for ch, _ in char_cnt.most_common(char_limit))

    seen: set[str] = set()
    out: list[str] = []
    for unit in picked:
        if unit not in seen:
            seen.add(unit)
            out.append(unit)
    return out


def unit_name(word: str) -> str:
    return hashlib.sha1(word.encode("utf-8")).hexdigest()[:16] + ".opus"


async def build(units: list[str], out: str, workers: int) -> dict[str, str]:
    lib = os.path.expanduser("~/.local/share/ipinyin/lib")
    if os.path.isdir(lib):
        sys.path.insert(0, lib)
    import edge_tts

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("需要 ffmpeg（sudo apt install ffmpeg）")
    os.makedirs(out, exist_ok=True)

    index: dict[str, str] = {}
    index_path = os.path.join(out, "index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, ValueError):
            index = {}

    # 已有文件的也要补进索引：上一次跑到一半被杀，文件在、索引没写，就成了孤儿
    for w in units:
        if os.path.exists(os.path.join(out, unit_name(w))):
            index[w] = unit_name(w)
    todo = [w for w in units if not os.path.exists(os.path.join(out, unit_name(w)))]
    print(f"目标 {len(units)} 个单元，已有 {len(units) - len(todo)}，待合成 {len(todo)}")
    sem = asyncio.Semaphore(workers)
    counter = {"ok": 0, "fail": 0}
    t0 = time.time()

    async def one(word: str) -> None:
        name = unit_name(word)
        target = os.path.join(out, name)
        async with sem:
            mp3 = target + ".mp3"
            try:
                await edge_tts.Communicate(word, VOICE).save(mp3)
                done = subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", mp3, "-c:a", "libopus",
                     "-b:a", OPUS_BITRATE, "-ar", "24000", "-ac", "1", target],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                )
                if done.returncode != 0 or not os.path.exists(target):
                    raise RuntimeError(f"ffmpeg 退出码 {done.returncode}")
                counter["ok"] += 1
                index[word] = name
            except Exception as exc:  # 单个词失败不该拖垮整批
                counter["fail"] += 1
                print(f"  跳过 {word}: {exc}")
            finally:
                if os.path.exists(mp3):
                    os.remove(mp3)
            if (counter["ok"] + counter["fail"]) % 100 == 0:
                print(f"  进度 {counter['ok']} 成功 / {counter['fail']} 失败，"
                      f"{time.time() - t0:.0f}s")

    await asyncio.gather(*(one(w) for w in todo))

    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=0, sort_keys=True)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in index.values())
    print(f"完成：{len(index)} 个单元，{size / 1e6:.1f}MB，耗时 {time.time() - t0:.0f}s → {out}")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description="预生成词级音库")
    ap.add_argument("--limit", type=int, default=3000, help="词条上限（默认 3000）")
    ap.add_argument("--chars", type=int, default=1500, help="单字兜底上限（默认 1500）")
    ap.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"输出目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--dry-run", action="store_true", help="只打印选词，不合成")
    args = ap.parse_args()

    units = pick_units(args.limit, args.chars)
    print(f"选出 {len(units)} 个单元（词 {args.limit} 上限 / 字 {args.chars} 上限）")
    print("  前 20:", " ".join(units[:20]))
    if args.dry_run:
        return 0
    asyncio.run(build(units, os.path.expanduser(args.out), args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
