#!/usr/bin/env python3
"""合并多源粤语平行语料，供反向模型（yue→zh）扩量训练。

来源（2026-10-26 实测）：
  vendor/yue_pairs.json                169,949 对（= indiejoseph/yue-zh-translation，
                                       已核实逐对相同，作底料）
  vendor/yue_extra/msgchi_train.json   raptorkwok/cantonese-chinese-parallel-corpus-base
                                       （msgchi 论文底料；zh 侧多为繁体，须归一简体）
  vendor/yue_extra/botisan_train.json  botisan-ai/cantonese-mandarin-translations（社区）

处理：zh 侧统一简体（Converter.to_simplified，语言模型/训练口径都是简中）、
清洗（空串/超长/非中文）、按 (zh, yue) 精确对去重、统计各来源贡献。

用法：python3 fetch_extra_corpus.py [--out data/merged_pairs.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.t2s import Converter  # noqa: E402

SOURCES = [
    ("repo", os.path.join(ROOT, "vendor", "yue_pairs.json")),
    ("msgchi", os.path.join(ROOT, "vendor", "yue_extra", "msgchi_train.json")),
    ("botisan", os.path.join(ROOT, "vendor", "yue_extra", "botisan_train.json")),
]


def is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def load_source(path: str) -> list[dict]:
    """两种格式：list[{zh,yue}]（repo）或 JSONL {'translation':{zh,yue}}（HF 下载）。"""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        head = fh.read(256)
        fh.seek(0)
        if head.lstrip().startswith("["):
            records = json.load(fh)
        else:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line).get("translation", {}))
                except ValueError:
                    continue
    out = []
    for r in records:
        zh = (r.get("zh") or "").strip()
        yue = (r.get("yue") or "").strip()
        if zh and yue and is_cjk(zh) and is_cjk(yue) \
                and len(zh) <= 38 and len(yue) <= 38:
            out.append({"zh": zh, "yue": yue})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "merged_pairs.json"))
    args = ap.parse_args()

    conv = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    stats: dict[str, dict] = {}

    for name, path in SOURCES:
        if not os.path.exists(path):
            stats[name] = {"file": "missing"}
            continue
        rows = load_source(path)
        kept = 0
        for r in rows:
            zh = conv.to_simplified(r["zh"])       # zh 侧统一简体（msgchi 是繁体）
            yue = r["yue"]
            key = (zh, yue)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"zh": zh, "yue": yue})
            kept += 1
        stats[name] = {"loaded": len(rows), "new_after_dedup": kept}
        print(f"{name}: 载入 {len(rows)}，去重后新增 {kept}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    print(f"合并总计 {len(merged)} 对 → {args.out}")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
