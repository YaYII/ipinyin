#!/usr/bin/env python3
"""朗读体检：读 tts-metrics.jsonl，出语速/耗时分布、异常清单，并顺手检查当前环境。

用法：
    python3 tools/tts_doctor.py              # 最近一轮记录 + 环境体检
    python3 tools/tts_doctor.py --tail 50    # 多列几条异常
    python3 tools/tts_doctor.py --all        # 连上一轮（.1）一起统计

异常规则见 src/ipinyin/tts_metrics.py：padding / too-slow / too-fast /
slow-synth / fallback-* / empty-audio。
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin import config, speak, tts_metrics  # noqa: E402


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def check_environment() -> None:
    """当前环境：输出设备、音库、后端可用性——「现在按下去会不会出声」。"""
    print("\n== 环境体检 ==")
    try:
        out = subprocess.run(["wpctl", "status"], capture_output=True, text=True,
                             timeout=5).stdout
        sink = next((line.strip() for line in out.splitlines() if line.strip().startswith("│  *")
                     or line.strip().startswith("*")), "")
        print(f"  输出设备: {sink or '（未找到）'}")
    except (OSError, subprocess.SubprocessError):
        print("  输出设备: wpctl 不可用")
    speaker = speak.Speaker()
    print(f"  音库单元: {len(speaker._units) if speaker._units else 0}")
    print(f"  edge 可用: {bool(speaker._edge)}    espeak 可用: {bool(speaker._binary)}")
    print(f"  指标文件: {tts_metrics.metrics_path()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="朗读体检")
    ap.add_argument("--tail", type=int, default=20, help="列出最近 N 条异常（默认 20）")
    ap.add_argument("--all", action="store_true", help="连上一轮记录（.1）一起统计")
    ap.add_argument("--no-env", action="store_true", help="只看记录，不查环境")
    args = ap.parse_args()

    path = tts_metrics.metrics_path()
    paths = [path]
    if args.all and os.path.exists(path + ".1"):
        paths.insert(0, path + ".1")
    rows: list[dict] = []
    for one in paths:
        rows.extend(tts_metrics.load(one))

    if not rows:
        print(f"还没有朗读记录（{path}）。按一次 Ctrl+回车 再来看。")
        if not args.no_env:
            check_environment()
        return 0

    print(f"== 朗读记录 {len(rows)} 条（{rows[0].get('ts')} ~ {rows[-1].get('ts')}）==")
    backends = collections.Counter(row.get("backend", "?") for row in rows)
    print("  后端分布:", ", ".join(f"{k}={v}" for k, v in backends.most_common()))

    rates = [row["rate"] for row in rows if row.get("rate")]
    if rates:
        print(f"  语速 字/秒: p10={_pct(rates, 0.1):.1f}  p50={statistics.median(rates):.1f}"
              f"  p90={_pct(rates, 0.9):.1f}  min={min(rates):.1f}  max={max(rates):.1f}")

    synth = [row["synth_ms"] for row in rows if row.get("synth_ms") is not None]
    if synth:
        print(f"  合成耗时: p50={statistics.median(synth):.0f}ms  p95={_pct(synth, 0.95):.0f}ms"
              f"  max={max(synth)}ms")

    flagged = [row for row in rows if row.get("flags")]
    print(f"  异常 {len(flagged)} 条（{len(flagged) / len(rows):.0%}）")
    counter = collections.Counter(flag for row in flagged for flag in row["flags"])
    for name, count in counter.most_common():
        print(f"    {name}: {count}")

    if flagged:
        print(f"\n== 最近 {min(args.tail, len(flagged))} 条异常 ==")
        for row in flagged[-args.tail:]:
            print(f"  {row.get('ts')}  {row.get('backend')}  {row.get('chars')}字 "
                  f"{row.get('audio_ms')}ms  {row.get('rate')}字/秒  "
                  f"{','.join(row['flags'])}  {row.get('sample')}")

    if not args.no_env:
        check_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
