#!/usr/bin/env python3
"""Suite ①：粤语按「真实打字长度」分桶评测（yue-lengths）。

口径与 tools/eval_ime_lengths.py 完全一致（直接复用它的题面构造与权重）：
  - 桶：2字 / 3字（词级挖矿题）/ 4字成语 / 5-7字古诗（「照抄不改」对照题）/ 5-7字（语料句）
  - 每桶报：照抄不改基线、规则转换、（可选）神经转换、神经延迟中位数
  - 汇总：按真实打字长度分布加权（2字 0.40 / 3字 0.20 / 4字成语 0.25 / 5-7字 0.15）

无模型 / 无 torch 时自动退化为「照抄 + 规则」两列，状态仍为 ok；
题料缺失（vendor/yue_pairs.json 不存在）时返回 skipped。
"""
from __future__ import annotations

import time
from collections import OrderedDict

from bench_common import (DEFAULT_MODEL, DEFAULT_PAIRS, file_ok, load_yue_nn,
                          median, repo, r4, stub_nn)

SUITE = "yue-lengths"
SUITE_VERSION = "1.0.0"

# 评测哪些桶（与 tools/eval_ime_lengths.py 的打印顺序一致；挖矿出的「4字」桶
# 被「4字成语」对照桶覆盖，不单独报——见 METHODOLOGY.md）
EVAL_BUCKETS = ("2字", "3字", "4字成语", "5-7字", "5-7字古诗")

QUICK_CAP = 50          # --quick 每桶题数上限
QUICK_LIMIT = 100000    # --quick 扫描的语料句对数（取前 1/20 当 test，保证挖矿桶非空）
FULL_LIMIT = 150000     # 全量口径，与 tools/eval_ime_lengths.py 默认一致


def build_buckets(test_pairs: list[dict], t2s) -> "OrderedDict[str, list[tuple[str, str]]]":
    """题面构造：与 tools/eval_ime_lengths.py 逐行同构。"""
    from eval_ime_lengths import build_word_test

    buckets = build_word_test(test_pairs)          # 2字 / 3字 / 4字（挖矿，剔除规则表已覆盖的 key）
    buckets["5-7字"] = [(p["zh"], p["yue"]) for p in test_pairs
                        if 5 <= len(p["zh"]) <= 7][:300]
    idioms = [ln.strip() for ln in open(_idioms_path(), encoding="utf-8")
              if ln.strip() and not ln.startswith("#")]
    poems = [ln.strip() for ln in open(_poems_path(), encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    # 对照桶：书面粤语里成语/古诗写法与普通话一致，正确输出 = 原文（繁体外壳）
    buckets["4字成语"] = [(w, t2s.to_traditional(w)) for w in idioms]
    buckets["5-7字古诗"] = [(w, t2s.to_traditional(w)) for w in poems]
    return buckets


def _idioms_path() -> str:
    return repo("tools", "idioms.txt")


def _poems_path() -> str:
    return repo("tools", "poems.txt")


def run(params: dict) -> tuple[str, dict | None, str | None]:
    """返回 (status, results, reason)。status ∈ {ok, skipped, error}。"""
    quick = params["quick"]
    cap = QUICK_CAP if quick else 10 ** 9
    limit = params.get("limit") or (QUICK_LIMIT if quick else FULL_LIMIT)
    pairs_path = params.get("pairs") or DEFAULT_PAIRS

    if not file_ok(pairs_path):
        return "skipped", None, f"语料不存在：{pairs_path}（先跑 `make corpus`）"

    from eval_ime_lengths import WEIGHTS, load_pairs
    from ipinyin.t2s import Converter
    from ipinyin.yue import YueConverter

    test = load_pairs(pairs_path, limit)
    conv = YueConverter.load_dict(
        params.get("yue_dict") or repo("data", "yue.marshal"),
        YueConverter.load(repo("data", "yue.tsv"))._map)
    t2s = Converter.load(repo("data", "s2t.marshal"))

    # 神经列解析：_nn_fn（测试注入）> nn=stub > nn=auto（真模型，缺则退化为两列）> off
    nn_fn, neural_note = None, None
    mode = params.get("nn") or "auto"
    if params.get("_nn_fn") is not None:
        nn_fn = params["_nn_fn"]
    elif mode == "stub":
        nn_fn = stub_nn()
        neural_note = "stub（原样返回，仅验证管线）"
    elif mode == "auto":
        nn_fn, reason = load_yue_nn(params.get("model") or DEFAULT_MODEL,
                                    params.get("device") or "cpu")
        if nn_fn is None:
            neural_note = reason

    buckets = build_buckets(test, t2s)
    out_buckets: "OrderedDict[str, dict]" = OrderedDict()
    empty: list[str] = []
    for name in EVAL_BUCKETS:
        items = (buckets.get(name) or [])[:cap]
        if not items:
            empty.append(name)
            continue
        c_ok = r_ok = n_ok = 0
        lat: list[float] = []
        for zh, ref in items:
            if t2s.to_traditional(zh) == ref:              # 照抄不改对照列
                c_ok += 1
            if t2s.to_traditional(conv.convert(zh)) == ref:
                r_ok += 1
            if nn_fn is not None:
                t0 = time.perf_counter()
                got = t2s.to_traditional(nn_fn(zh))
                lat.append((time.perf_counter() - t0) * 1000)
                if got == ref:
                    n_ok += 1
        n = len(items)
        out_buckets[name] = {
            "n": n,
            "copy_baseline": r4(c_ok / n),
            "rule": r4(r_ok / n),
            "neural": r4(n_ok / n) if nn_fn is not None else None,
            "neural_median_latency_ms": r4(median(lat)) if lat else None,
            "weight": WEIGHTS.get(name, 0.0),
        }

    total_w = sum(b["weight"] for b in out_buckets.values())
    rule_w = sum(b["weight"] * b["rule"] for b in out_buckets.values())
    nn_w = (sum(b["weight"] * b["neural"] for b in out_buckets.values()
                if b["neural"] is not None) if nn_fn is not None else None)
    results = {
        "buckets": out_buckets,
        "weighted": {
            "weighted_rule": r4(rule_w / total_w) if total_w else None,
            "weighted_neural": r4(nn_w / total_w) if (nn_w is not None and total_w) else None,
            "weight_coverage": r4(total_w),
        },
        "empty_buckets": empty,
        "neural_note": neural_note,
    }
    return "ok", results, None
