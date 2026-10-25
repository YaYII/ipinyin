#!/usr/bin/env python3
"""覆盖率报告：样例普通话文本走完 G2P 管线后，统计能注音的字/词占比。

对 TTS 来说「有读法的字占比」是硬指标 —— 缺读法的字在合成语音里要么跳过
要么静音，直接毁掉整句。本脚本给出三个口径：

  字级覆盖   有粤拼的汉字 / 全部汉字（粤语书面文本上统计）
  词级命中   以「词表整词」注音的 token / 全部汉字 token（词级注音比字级准，
             例：呢個 ni1 go3 是整词读法，逐字拼可能得出错音）
  音节产出   产出的粤拼音节数 / 汉字数（词表条目音节数≠字数时暴露出来）

缺读法的字按出现次数排序列出 —— 补表就是按这个清单补。

用法：
  python3 projects/03-tts-g2p/coverage.py                     # 用自带 samples.txt
  python3 projects/03-tts-g2p/coverage.py --file 你的文本.txt
  python3 projects/03-tts-g2p/coverage.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from g2p import G2P  # noqa: E402

DEFAULT_SAMPLES = os.path.join(HERE, "samples.txt")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def analyze(g2p: G2P, lines: list[str]) -> dict:
    per_sentence: list[dict] = []
    missing_counter: Counter[str] = Counter()
    totals = {"hanzi": 0, "read": 0, "word_tokens": 0,
              "char_tokens": 0, "missing_tokens": 0, "syllables": 0}

    for line in lines:
        res = g2p(line)
        hanzi = sum(len(t.surface) for t in res.tokens if t.kind in ("word", "char", "missing"))
        read = sum(len(t.surface) for t in res.tokens if t.kind in ("word", "char"))
        word_tok = sum(1 for t in res.tokens if t.kind == "word")
        char_tok = sum(1 for t in res.tokens if t.kind == "char")
        miss_tok = sum(1 for t in res.tokens if t.kind == "missing")
        missing_counter.update(res.missing)
        totals["hanzi"] += hanzi
        totals["read"] += read
        totals["word_tokens"] += word_tok
        totals["char_tokens"] += char_tok
        totals["missing_tokens"] += miss_tok
        totals["syllables"] += len(res.jyutping)
        per_sentence.append({
            "raw": res.raw,
            "yue": res.yue,
            "hanzi": hanzi,
            "read": read,
            "missing": res.missing,
        })

    hanzi = totals["hanzi"] or 1
    cjk_tokens = totals["word_tokens"] + totals["char_tokens"] + totals["missing_tokens"] or 1
    return {
        "sentences": len(lines),
        "char_coverage": totals["read"] / hanzi,
        "char_missing": totals["hanzi"] - totals["read"],
        "word_hit_rate": totals["word_tokens"] / cjk_tokens,
        "word_tokens": totals["word_tokens"],
        "char_tokens": totals["char_tokens"],
        "missing_tokens": totals["missing_tokens"],
        "syllables": totals["syllables"],
        "syllables_per_hanzi": totals["syllables"] / hanzi,
        "missing_chars": missing_counter.most_common(),
        "sentences_detail": per_sentence,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="G2P 覆盖率报告")
    ap.add_argument("--file", default=DEFAULT_SAMPLES, help="样例文本（每行一句）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--nn", action="store_true", help="启用神经转换")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not lines:
        print("没有可处理的句子", file=sys.stderr)
        return 1

    report = analyze(G2P(neural=args.nn), lines)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"样例文件     : {args.file}")
    print(f"句子数       : {report['sentences']}")
    print(f"字级覆盖率   : {report['char_coverage']:.1%}"
          f"（缺读法 {report['char_missing']} 字）")
    print(f"词级命中率   : {report['word_hit_rate']:.1%}"
          f"（词 {report['word_tokens']} / 字 {report['char_tokens']}"
          f" / 缺 {report['missing_tokens']}）")
    print(f"音节产出     : {report['syllables']} 个"
          f"（每汉字 {report['syllables_per_hanzi']:.2f} 个，应≈1.00）")
    if report["missing_chars"]:
        print("缺读法的字（按出现次数）:")
        for ch, cnt in report["missing_chars"]:
            print(f"  {ch}  x{cnt}")
    else:
        print("缺读法的字   : 无")
    return 0


if __name__ == "__main__":
    sys.exit(main())
