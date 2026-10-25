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

from bench_common import (DEFAULT_GUARD, DEFAULT_MODEL, DEFAULT_PAIRS,
                          accept_forms, file_ok, fingerprint, load_variant_map,
                          load_yue_nn, median, repo, r4, stub_nn,
                          question_hash, wilson)

SUITE = "yue-guard"
SUITE_VERSION = "1.3.0"

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
    nn_from_ckpt = mode == "auto" and params.get("_nn_fn") is None

    test = load_pairs(pairs_path, limit)
    conv = YueConverter.load_dict(repo("data", "yue.marshal"),
                                  YueConverter.load(repo("data", "yue.tsv"))._map)
    t2s = Converter.load(repo("data", "s2t.marshal"))
    guard.ctx_max_len = int(params.get("ctx_max") or 7)   # 与 exp_yue_guard 默认一致

    buckets = build_buckets(test, t2s)
    buckets["8-15字"] = [(p["zh"], p["yue"]) for p in test
                         if 8 <= len(p["zh"]) <= 15][:300]

    show = int(params.get("show") or 5)
    variants = load_variant_map()
    out_buckets: "OrderedDict[str, dict]" = OrderedDict()
    examples: "OrderedDict[str, list[dict]]" = OrderedDict()
    empty: list[str] = []
    per_item: list[dict] = []
    questions: list[dict] = []
    inputs: list[dict] = []
    for p in (pairs_path, guard_path,
              params.get("model") or DEFAULT_MODEL if nn_from_ckpt else None,
              repo("data", "yue.marshal"), repo("data", "yue.tsv"),
              repo("data", "s2t.marshal")):
        fp = fingerprint(p)
        if fp:
            inputs.append(fp)
    tot = {"rejected": 0, "saved": 0, "lost": 0}
    for name in EVAL_BUCKETS:
        items = (buckets.get(name) or [])[:cap]
        if not items:
            empty.append(name)
            continue
        r_ok = n_ok = g_ok = rejected = saved = lost = 0
        lat: list[float] = []
        ex: list[dict] = []
        multi = name in ("2字", "3字")           # v1.3.0：词桶多答案（合法变体）
        for i, (zh, ref) in enumerate(items):
            questions.append({"id": f"{name}:{i}", "bucket": name, "input": zh})
            accept = accept_forms(zh, ref, variants, shell=t2s) if multi \
                else [t2s.to_traditional(ref)]
            rule_out = conv.convert(zh)
            rule_right = t2s.to_traditional(rule_out) in accept
            if rule_right:
                r_ok += 1
            nn_out = nn_fn(zh)
            nn_right = t2s.to_traditional(nn_out) in accept
            if nn_right:
                n_ok += 1
            t0 = time.perf_counter()
            final = guard.check(zh, nn_out, repair=conv.convert)
            lat.append((time.perf_counter() - t0) * 1000)
            repaired = final != nn_out
            guarded_right = t2s.to_traditional(final) in accept
            if guarded_right:
                g_ok += 1
            outcome = "pass"                       # 守卫放行且未改
            if repaired:
                rejected += 1
                if rule_right and not nn_right:
                    saved += 1
                    outcome = "saved"
                elif nn_right and not rule_right:
                    lost += 1
                    outcome = "lost"
                elif guarded_right:
                    outcome = "fixed_right"
                else:
                    outcome = "both_wrong"
                if len(ex) < show:
                    ex.append({"input": zh, "neural": nn_out, "guarded": final,
                               "expected": ref, "outcome": outcome})
            per_item.append({"id": f"{name}:{i}", "bucket": name, "input": zh,
                             "expected": ref, "accept": accept,
                             "rule_out": rule_out,
                             "rule_ok": rule_right, "neural_out": nn_out,
                             "nn_ok": nn_right, "guarded_out": final,
                             "guarded_ok": guarded_right, "outcome": outcome,
                             "latency_ms": r4(lat[-1])})
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
            "wilson": {"rule": wilson(r_ok, n), "neural": wilson(n_ok, n),
                       "guarded": wilson(g_ok, n)},
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
        "inputs": inputs,
        "question_set_hash": question_hash(questions),
        "per_item": per_item,
    }
    return "ok", results, None


def suite_limit(quick: bool) -> int:
    from suite_yue_lengths import FULL_LIMIT, QUICK_LIMIT
    return QUICK_LIMIT if quick else FULL_LIMIT


def question_sets(params: dict):
    """题面/答案构造（export/score 复用，与 run() 同源，含 8-15字 桶）。

    守卫数据只影响判分（guarded 列），不影响题面，构造题面不需要它。
    返回 (questions, answers, conv, t2s)；v1.3.0 起词桶 answers 带变体 accept。
    """
    from eval_ime_lengths import load_pairs
    from ipinyin.t2s import Converter
    from ipinyin.yue import YueConverter

    quick = params["quick"]
    cap = QUICK_CAP if quick else 10 ** 9
    limit = params.get("limit") or (suite_limit(quick))
    pairs_path = params.get("pairs") or DEFAULT_PAIRS

    test = load_pairs(pairs_path, limit)
    conv = YueConverter.load_dict(repo("data", "yue.marshal"),
                                  YueConverter.load(repo("data", "yue.tsv"))._map)
    t2s = Converter.load(repo("data", "s2t.marshal"))
    buckets = build_buckets(test, t2s)
    buckets["8-15字"] = [(p["zh"], p["yue"]) for p in test
                         if 8 <= len(p["zh"]) <= 15][:300]
    variants = load_variant_map()

    questions: list[dict] = []
    answers: list[dict] = []
    for name in EVAL_BUCKETS:
        multi = name in ("2字", "3字")
        for i, (zh, ref) in enumerate((buckets.get(name) or [])[:cap]):
            questions.append({"id": f"{name}:{i}", "bucket": name, "input": zh})
            answers.append({"id": f"{name}:{i}", "bucket": name, "expected": ref,
                            "accept": accept_forms(zh, ref, variants, shell=t2s)
                            if multi else [t2s.to_traditional(ref)]})
    return questions, answers, conv, t2s
