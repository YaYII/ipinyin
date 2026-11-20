#!/usr/bin/env python3
"""语音播放往返质检：合成 → 识别 → 比对，报出读错的地方。

用法：python3 tools/tts_roundtrip.py [--limit N] [--save 报告.md]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/usr/lib/ipinyin")

from ipinyin import asr, config, speak, t2s, units  # noqa: E402

PUNCT = "。！？，、；：,.!?;: \n"

CASES: list[tuple[str, list[str]]] = [
    ("单字", ["我", "你", "佢", "食", "行"]),
    ("双字词", ["唔該", "今日", "天氣", "香港", "多謝"]),
    ("变调多音", ["阿爸", "銀行", "長大", "重要", "校長"]),
    ("重复字", ["你你你你", "謝謝", "嘅嘅"]),
    ("标点停顿", ["我哋去行街,買嘢食。", "你食咗飯未?"]),
    ("英数混排", ["iPhone 幾多錢", "2026 年"]),
    ("长句", ["我哋今日去行街買嘢食順便睇下戲然後返屋企食飯夜晚再出去散步傾下偈"]),
    ("生僻词", ["魑魅魍魉"]),
    ("普转粤", ["我不知道", "这个多少钱", "他刚才说什么"]),
]


def normalize(text: str, conv) -> str:
    text = text.strip(PUNCT).lower()
    try:
        text = conv.to_simplified(text)
    except Exception:
        pass
    return "".join(ch for ch in text if ch not in PUNCT)


DIGITS = str.maketrans({"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"})


def reading(text: str, yue, conv=None) -> str:
    """按粤拼比对：返/翻 同音、钱/剪 异音。

    粤拼字典是繁体：简体文本必须先转繁体，否则查不到读音、音节数直接少几个。
    """
    if yue is None:
        return ""
    if conv is not None:
        try:
            text = conv.to_traditional(text)
        except Exception:
            pass
    try:
        return " ".join((yue.jyutping(text) or "").split())
    except Exception:
        return ""


def digits_to_chinese(text: str) -> str:
    return text.translate(DIGITS)


def compare(text: str, heard: str, conv, yue) -> str:
    """返回 ok / homophone / mismatch。"""
    plain, got = normalize(text, conv), normalize(heard, conv)
    if plain == got:
        return "ok"
    # 数字读成汉字（2026 → 二零二六）是正确行为，不算错
    if digits_to_chinese(plain) == digits_to_chinese(got):
        return "ok"
    a, b = reading(text, yue, conv), reading(heard, yue, conv)
    if a and a == b:
        return "homophone"      # 同音不同字，发音其实是对的
    return "mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS 往返质检")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 条")
    ap.add_argument("--save", default="", help="把报告写到文件")
    args = ap.parse_args()

    conv = t2s.Converter.load(config.s2t_path())
    rec = asr.Recognizer("yue")
    speaker = speak.Speaker(binary=speak._find_espeak(), player=["true"],
                            cache_dir=config.tts_cache_dir(), edge="auto", unit_lib="auto")
    yue = None
    try:
        from ipinyin import yue as yue_mod

        yue = yue_mod.YueConverter.load_dict(config.yue_dict_path(),
                                             yue_mod.YueConverter.load(config.yue_path())._map)
    except Exception:
        yue = None

    rows = []
    total = ok = 0
    homophones = 0
    started = time.time()
    for title, items in CASES:
        for raw in items:
            if args.limit and total >= args.limit:
                break
            text = raw
            if title == "普转粤" and yue is not None:
                text = yue.convert(raw) or raw
            got = speaker.wav(text, speak.YUE_VOICE)
            if not got:
                rows.append((title, text, "（合成失败）", "mismatch"))
                total += 1
                continue
            path = os.path.join("/tmp", "roundtrip.wav")
            with open(path, "wb") as fh:
                fh.write(got)
            heard = rec.recognize_file(path)
            verdict = compare(text, heard, conv, yue)
            ok += verdict == "ok"
            homophones += verdict == "homophone"
            total += 1
            rows.append((title, text, heard, verdict))
        if args.limit and total >= args.limit:
            break

    lines = [f"# TTS 往返质检报告", "",
             f"用例 {total}，完全一致 {ok}（{ok / total * 100:.0f}%），"
             f"同音 {homophones}，真错 {total - ok - homophones}，"
             f"耗时 {time.time() - started:.0f}s", "",
             "| 场景 | 原文 | 识别结果 | 判定 |", "|---|---|---|---|"]
    mark = {"ok": "✅", "homophone": "🟡 同音", "mismatch": "❌"}
    for title, text, heard, verdict in rows:
        lines.append(f"| {title} | {text} | {heard} | {mark.get(verdict, verdict)} |")
    report = "\n".join(lines)
    print(report)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\n报告已写入 {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
