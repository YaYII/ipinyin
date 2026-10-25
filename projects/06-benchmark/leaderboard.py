#!/usr/bin/env python3
"""汇总 results/ 下所有结果 JSON 生成 LEADERBOARD.md。

行 = 一次评测（系统 × suite × 模式）。主指标：
  yue-lengths  → weighted.weighted_neural（无神经列退回 weighted_rule）
  yue-guard    → guarded 的按题数微平均（从 per_item 或桶值算）
  zh-sentence  → aggregate.top1

纪律：quick 与 full 分表；quick 表整表标注「冒烟，不具排名资格」
（METHODOLOGY §6：≤50 题/桶只够抓 >10pp 的事故，不够排名）。

用法：python3 leaderboard.py [--results results] [--out LEADERBOARD.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from bench_common import BENCH_DIR

sys.path.insert(0, BENCH_DIR)

MAIN_METRIC = {
    "yue-lengths": ("加权", lambda r: (r.get("weighted") or {}).get("weighted_neural")
                    if (r.get("weighted") or {}).get("weighted_neural") is not None
                    else (r.get("weighted") or {}).get("weighted_rule")),
    "yue-guard": ("guarded 微平均", None),   # 专用计算（见 _guard_micro）
    "zh-sentence": ("Top1", lambda r: (r.get("aggregate") or {}).get("top1")),
}


def _guard_micro(res: dict) -> float | None:
    per = res.get("per_item")
    if per:
        ok = sum(1 for it in per if it.get("guarded_ok"))
        return ok / len(per) if per else None
    buckets = res.get("buckets") or {}
    num = den = 0
    for b in buckets.values():
        if isinstance(b, dict) and b.get("guarded") is not None:
            num += b["guarded"] * b["n"]
            den += b["n"]
    return num / den if den else None


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def _rows(results_dir: str) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            doc = json.load(open(os.path.join(results_dir, fn), encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(doc, dict) or doc.get("status") != "ok" \
                or doc.get("suite") not in MAIN_METRIC:
            continue
        res = doc.get("results") or {}
        metric_label, fn_metric = MAIN_METRIC[doc["suite"]]
        val = _guard_micro(res) if doc["suite"] == "yue-guard" else fn_metric(res)
        n_total = sum(b.get("n", 0) for b in (res.get("buckets") or {}).values()
                      if isinstance(b, dict))
        tables.setdefault(doc["suite"], []).append({
            "system": doc.get("system") or "未知系统",
            "version": doc.get("suite_version"),
            "quick": doc.get("quick"),
            "n": n_total,
            "metric_label": metric_label,
            "value": val,
            "date": (doc.get("generated_at") or "")[:10],
            "file": fn,
        })
    for rows in tables.values():
        rows.sort(key=lambda r: (r["value"] is None, -(r["value"] or 0)))
    return tables


def render(tables: dict[str, list[dict]]) -> str:
    lines = ["# LEADERBOARD", "",
             "由 `leaderboard.py` 从 results/ 自动汇总；重新生成：",
             "`python3 leaderboard.py`。报数纪律见 METHODOLOGY §6 —— "
             "**quick 表不具排名资格**，只用于冒烟与回归。",
             ""]
    for quick_mode in (False, True):
        section = [] 
        for suite in ("yue-lengths", "yue-guard", "zh-sentence"):
            rows = [r for r in tables.get(suite, []) if r["quick"] == quick_mode]
            if not rows:
                continue
            section.append(f"## {suite}（{'--quick 冒烟' if quick_mode else '全量'}）")
            if quick_mode:
                section.append("> ⚠️ 冒烟数据（每桶 ≤50 题），**不具排名资格**。")
            section.append("")
            label = rows[0]["metric_label"]
            section.append("| 系统 | 版本 | n | " + label + " | 日期 | 文件 |")
            section.append("|---|---|---|---|---|---|")
            for r in rows:
                section.append(
                    f"| {r['system']} | {r['version']} | {r['n']} | "
                    f"{_fmt(r['value'])} | {r['date']} | `{r['file']}` |")
            section.append("")
        if section:
            lines.extend(section)
    if not any(tables.values()):
        lines.append("（results/ 下暂无 ok 状态的结果）")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(BENCH_DIR, "results"))
    ap.add_argument("--out", default=os.path.join(BENCH_DIR, "LEADERBOARD.md"))
    args = ap.parse_args(argv)
    tables = _rows(args.results)
    md = render(tables)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    n_rows = sum(len(v) for v in tables.values())
    print(f"[ok] {n_rows} 行 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
