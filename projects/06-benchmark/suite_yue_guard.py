#!/usr/bin/env python3
"""Suite ②：粤语守卫决策评测（yue-guard）。

口径与 experiments/exp_yue_guard.py 完全一致（复用其题面桶与决策分类）：
  - 桶：2字 / 4字成语 / 5-7字 / 5-7字古诗 / 8-15字
  - 三个系统：规则、神经（裸输出）、神经+守卫（无实证改动否掉→回退规则修复）
  - 每桶报：三系统准确率、守卫否决数、救回 / 弄丢（分开报）、守卫延迟中位数
  - 附守卫决策样例（input / neural / guarded / expected / outcome）

依赖 torch + 神经模型 + 守卫数据（data/yue_guard.eval.marshal，训练侧切分，
守卫没见过任何考题）。缺任一 → 状态 skipped（输出 {"status":"skipped"}）。
"""
from __future__ import annotations

import time
from collections import OrderedDict

from bench_common import (DEFAULT_GUARD, DEFAULT_MODEL, DEFAULT_PAIRS, file_ok,
                          load_yue_nn, median, repo, r4, stub_nn)

SUITE = "yue-guard"
SUITE_VERSION = "1.0.0"

EVAL_BUCKETS = ("2字", "4字成语", "5-7字", "5-7字古诗", "8-15字")
QUICK_CAP = 50

from suite_yue_lengths import build_buckets  # noqa: E402  题面构造与 suite ① 同源


def run(params: dict) -> tuple[str, dict | None, str | None]:
    quick = params["quick"]
    cap = QUICK_CAP if quick else 10 ** 9
    limit = params.get("limit") or (suite_limit(quick))
    pairs_path = params.get("pairs") or DEFAULT_PAIRS
    guard_path = params.get("guard") or DEFAULT_GUARD

    if not file_ok(pairs_path):
        return "skipped", None, f"语料不存在：{pairs_path}（先跑 `make corpus`）"
    if not file_ok(guard_path):
        return "skipped", None, (f"守卫数据不存在：{guard_path}（评测口径数据不入库，"
                                 f"先跑 `python3 tools/build_yue_guard.py`，默认 eval 切分）")

    from eval_ime_lengths import load_pairs
    from ipinyin.t2s import Converter
    from ipinyin.yue import YueConverter
    from ipinyin.yue_guard import YueGuard

    guard = YueGuard.load(guard_path)
    if guard is None:
        return "skipped", None, f"守卫数据加载失败：{guard_path}"

    mode = params.get("nn") or "auto"
    nn_fn, neural_note = None, None
    if params.get("_nn_fn") is not None:
        nn_fn = params["_nn_fn"]
    elif mode == "stub":
        nn_fn = stub_nn()
        neural_note = "stub（原样返回，仅验证管线）"
    elif mode == "auto":
        nn_fn, reason = load_yue_nn(params.get("model") or DEFAULT_MODEL,
                                    params.get("device") or "cpu")
        if nn_fn is None:
            return "skipped", None, reason

    test = load_pairs(pairs_path, limit)
    conv = YueConverter.load_dict(repo("data", "yue.marshal"),
                                  YueConverter.load(repo("data", "yue.tsv"))._map)
    t2s = Converter.load(repo("data", "s2t.marshal"))
    guard.ctx_max_len = int(params.get("ctx_max") or 7)   # 与 exp_yue_guard 默认一致

    buckets = build_buckets(test, t2s)
    buckets["8-15字"] = [(p["zh"], p["yue"]) for p in test
                         if 8 <= len(p["zh"]) <= 15][:300]

    show = int(params.get("show") or 5)
    out_buckets: "OrderedDict[str, dict]" = OrderedDict()
    examples: "OrderedDict[str, list[dict]]" = OrderedDict()
    empty: list[str] = []
    tot = {"rejected": 0, "saved": 0, "lost": 0}
    for name in EVAL_BUCKETS:
        items = (buckets.get(name) or [])[:cap]
        if not items:
            empty.append(name)
            continue
        r_ok = n_ok = g_ok = rejected = saved = lost = 0
        lat: list[float] = []
        ex: list[dict] = []
        for zh, ref in items:
            rule_out = conv.convert(zh)
            rule_right = t2s.to_traditional(rule_out) == ref
            if rule_right:
                r_ok += 1
            nn_out = nn_fn(zh)
            nn_right = t2s.to_traditional(nn_out) == ref
            if nn_right:
                n_ok += 1
            t0 = time.perf_counter()
            final = guard.check(zh, nn_out, repair=conv.convert)
            lat.append((time.perf_counter() - t0) * 1000)
            repaired = final != nn_out
            if t2s.to_traditional(final) == ref:
                g_ok += 1
            if not repaired:
                continue
            rejected += 1
            outcome = None
            if rule_right and not nn_right:
                saved += 1
                outcome = "saved"
            elif nn_right and not rule_right:
                lost += 1
                outcome = "lost"
            elif t2s.to_traditional(final) == ref:
                outcome = "fixed_right"
            else:
                outcome = "both_wrong"
            if len(ex) < show:
                ex.append({"input": zh, "neural": nn_out, "guarded": final,
                           "expected": ref, "outcome": outcome})
        n = len(items)
        out_buckets[name] = {
            "n": n,
            "copy_baseline": None,               # 守卫口径不评照抄列（规则列已承担对照）
            "rule": r4(r_ok / n),
            "neural": r4(n_ok / n),
            "guarded": r4(g_ok / n),
            "rejected": rejected,
            "accepted": n - rejected,
            "saved": saved,
            "lost": lost,
            "guarded_median_latency_ms": r4(median(lat)) if lat else None,
            "weight": None,
        }
        if ex:
            examples[name] = ex
        tot["rejected"] += rejected
        tot["saved"] += saved
        tot["lost"] += lost

    results = {
        "buckets": out_buckets,
        "aggregate": dict(tot),
        "examples": examples,
        "empty_buckets": empty,
        "neural_note": neural_note,
    }
    return "ok", results, None


def suite_limit(quick: bool) -> int:
    from suite_yue_lengths import FULL_LIMIT, QUICK_LIMIT
    return QUICK_LIMIT if quick else FULL_LIMIT
