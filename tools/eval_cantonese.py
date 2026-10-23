#!/usr/bin/env python3
"""粤语转换质量基准：拿真实的「普通话-粤语」句对来打分。

数据：HuggingFace `indiejoseph/yue-zh-translation`（CC BY 4.0，16.9 万句对）。
用法：
  python3 tools/eval_cantonese.py [pairs.json] [-v]
  # pairs.json 形如 [{"zh": "...", "yue": "..."}, ...]
  # 没有文件时会用 datasets-server 抓 1000 句（需要联网）
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.t2s import Converter   # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402

DATA = os.path.join(os.path.expanduser("~"), ".local/share/ipinyin/data")
API = ("https://datasets-server.huggingface.co/rows?dataset=indiejoseph%2Fyue-zh-translation"
       "&config=default&split=train&offset={}&length=100")


def fetch(limit: int) -> list[dict]:
    rows: list[dict] = []
    for offset in range(0, limit, 100):
        try:
            with urllib.request.urlopen(API.format(offset), timeout=60) as fh:
                payload = json.load(fh)
        except Exception as exc:                      # 网络/限流
            print(f"抓取中断（{exc}），已取 {len(rows)} 条", file=sys.stderr)
            break
        rows.extend(item["row"]["translation"] for item in payload.get("rows", []))
    return rows


def main() -> int:
    verbose = "-v" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        pairs = json.load(open(args[0], encoding="utf-8"))
    else:
        pairs = fetch(1000)

    converter = YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))

    exact = 0
    overlap = 0.0
    total = 0
    ours_tokens: Counter[str] = Counter()
    ref_tokens: Counter[str] = Counter()
    samples: list[tuple[str, str, str]] = []

    for item in pairs:
        zh, ref = item.get("zh", ""), item.get("yue", "")
        if not zh or not ref:
            continue
        out = t2s.to_traditional(converter.convert(zh))
        total += 1
        exact += out == ref
        overlap += sum(1 for a, b in zip(out, ref) if a == b) / max(len(ref), 1)
        for n in (2, 3):
            for i in range(len(out) - n + 1):
                ours_tokens[out[i:i + n]] += 1
            for i in range(len(ref) - n + 1):
                ref_tokens[ref[i:i + n]] += 1
        if len(samples) < 10 and out != ref:
            samples.append((zh, out, ref))

    if not total:
        print("没有可用句对")
        return 1

    print(f"句对 {total} 条")
    print(f"  完全一致   {exact:5d}  {exact/total:6.1%}")
    print(f"  字符重合   {overlap/total:6.1%}")

    gaps = [(tok, cnt) for tok, cnt in ref_tokens.most_common(3000)
            if ours_tokens[tok] == 0 and cnt >= 30]
    if gaps:
        print("\n参考里高频、我们从不产出（说明缺映射）：")
        for tok, cnt in gaps[:15]:
            print(f"  {tok} ×{cnt}")

    if verbose:
        print("\n错例：")
        for zh, out, ref in samples:
            print(f"  普 {zh}\n  我 {out}\n  参 {ref}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
