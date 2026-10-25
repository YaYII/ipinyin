#!/usr/bin/env python3
"""结果 JSON 的轻量校验器（schemas.md 的可执行形态）。

不依赖 jsonschema 库：字段存在性、类型、取值范围手写检查。
run_all.py 写盘前自检，test_runner.py 对每份产出的 JSON 断言零错误。
字段定义与版本化规则见同目录 schemas.md；改字段必须同步改这里并升 suite_version。
"""
from __future__ import annotations

import math

SUITES = ("yue-lengths", "yue-guard", "zh-sentence")
STATUSES = ("ok", "skipped", "error")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_ratio(x) -> bool:
    return _is_num(x) and 0.0 <= x <= 1.0


def _errs_if(cond: bool, msg: str, errs: list[str]) -> None:
    if cond:
        errs.append(msg)


def _check_bucket_metrics(name: str, b: dict, errs: list[str]) -> None:
    for k in ("n", "copy_baseline", "rule"):
        _errs_if(k not in b, f"{name}: 缺字段 {k}", errs)
    _errs_if(not isinstance(b.get("n"), int) or b["n"] < 0, f"{name}: n 非法", errs)
    for k in ("copy_baseline", "rule", "neural"):
        v = b.get(k)
        _errs_if(v is not None and not _is_ratio(v), f"{name}: {k} 不是 [0,1] 比率", errs)
    lat = b.get("neural_median_latency_ms")
    _errs_if(lat is not None and (not _is_num(lat) or lat < 0),
             f"{name}: neural_median_latency_ms 非法", errs)
    w = b.get("weight")
    _errs_if(w is not None and not _is_ratio(w), f"{name}: weight 不是 [0,1]", errs)
    _check_wilson(name, b.get("wilson"), errs)


def _check_wilson(name: str, w, errs: list[str]) -> None:
    """wilson（v1.1.0 增，可选）：{指标名: {lo, hi}}，0 ≤ lo ≤ hi ≤ 1。"""
    if w is None:
        return
    _errs_if(not isinstance(w, dict), f"{name}: wilson 应为 object", errs)
    if not isinstance(w, dict):
        return
    for k, iv in w.items():
        if iv is None:
            continue
        ok = (isinstance(iv, dict) and {"lo", "hi"} <= set(iv)
              and _is_num(iv["lo"]) and _is_num(iv["hi"])
              and 0.0 <= iv["lo"] <= iv["hi"] <= 1.0)
        _errs_if(not ok, f"{name}: wilson.{k} 非法（需 0≤lo≤hi≤1）", errs)


def _check_v1_1_fields(suite: str, res: dict, errs: list[str]) -> None:
    """v1.1.0 增的可选字段：inputs 文件指纹、question_set_hash、per_item 逐题。"""
    inputs = res.get("inputs")
    if inputs is not None:
        _errs_if(not isinstance(inputs, list), f"{suite}: inputs 应为 list", errs)
        if isinstance(inputs, list):
            for i, fp in enumerate(inputs):
                ok = (isinstance(fp, dict)
                      and isinstance(fp.get("path"), str) and fp["path"]
                      and isinstance(fp.get("sha256_16"), str) and len(fp["sha256_16"]) == 16
                      and isinstance(fp.get("bytes"), int) and fp["bytes"] >= 0)
                _errs_if(not ok, f"{suite}: inputs[{i}] 指纹非法", errs)
    qsh = res.get("question_set_hash")
    _errs_if(qsh is not None and (not isinstance(qsh, str) or len(qsh) != 16),
             f"{suite}: question_set_hash 应为 16 位十六进制", errs)
    per_item = res.get("per_item")
    if per_item is not None:
        _errs_if(not isinstance(per_item, list), f"{suite}: per_item 应为 list", errs)
        if isinstance(per_item, list):
            seen: set = set()
            for i, it in enumerate(per_item):
                ok = (isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"]
                      and isinstance(it.get("bucket"), str) and it["bucket"])
                _errs_if(not ok, f"{suite}: per_item[{i}] 缺 id/bucket", errs)
                if ok:
                    _errs_if(it["id"] in seen, f"{suite}: per_item id 重复 {it['id']}", errs)
                    seen.add(it["id"])
                acc = it.get("accept") if isinstance(it, dict) else None
                if acc is not None:
                    _errs_if(not isinstance(acc, list) or not all(
                        isinstance(a, str) and a for a in acc),
                        f"{suite}: per_item[{i}].accept 应为非空字符串列表", errs)
                for k in ("copy_ok", "rule_ok", "neural_ok", "nn_ok",
                          "guarded_ok", "top1_ok", "top9_ok", "base_top1_ok"):
                    v = it.get(k) if isinstance(it, dict) else None
                    _errs_if(v is not None and not isinstance(v, bool),
                             f"{suite}: per_item[{i}].{k} 应为布尔或 null", errs)


def _check_yue_lengths(res: dict, errs: list[str]) -> None:
    buckets = res.get("buckets")
    _errs_if(not isinstance(buckets, dict), "yue-lengths: buckets 缺失", errs)
    if not isinstance(buckets, dict):
        return
    for name, b in buckets.items():
        if isinstance(b, dict):
            _check_bucket_metrics(f"yue-lengths.{name}", b, errs)
    weighted = res.get("weighted")
    _errs_if(not isinstance(weighted, dict), "yue-lengths: weighted 缺失", errs)
    if isinstance(weighted, dict):
        for k in ("weighted_rule", "weighted_neural", "weight_coverage"):
            v = weighted.get(k)
            _errs_if(v is not None and not _is_ratio(v),
                     f"yue-lengths.weighted.{k} 不是 [0,1]", errs)


def _check_yue_guard(res: dict, errs: list[str]) -> None:
    buckets = res.get("buckets")
    _errs_if(not isinstance(buckets, dict), "yue-guard: buckets 缺失", errs)
    if isinstance(buckets, dict):
        for name, b in buckets.items():
            if not isinstance(b, dict):
                continue
            _check_bucket_metrics(f"yue-guard.{name}", b, errs)
            for k in ("guarded",):               # 守卫后准确率，[0,1] 比率
                v = b.get(k)
                _errs_if(v is not None and not _is_ratio(v),
                         f"yue-guard.{name}: {k} 不是 [0,1] 比率", errs)
            for k in ("rejected", "saved", "lost", "accepted"):
                v = b.get(k)
                _errs_if(not isinstance(v, int) or v < 0,
                         f"yue-guard.{name}: {k} 应为非负整数", errs)
    agg = res.get("aggregate")
    _errs_if(not isinstance(agg, dict), "yue-guard: aggregate 缺失", errs)
    if isinstance(agg, dict):
        for k in ("rejected", "saved", "lost"):
            _errs_if(not isinstance(agg.get(k), int), f"yue-guard.aggregate.{k} 非法", errs)
    ex = res.get("examples")
    _errs_if(ex is not None and not isinstance(ex, dict), "yue-guard: examples 应为 dict", errs)


def _check_zh_sentence(res: dict, errs: list[str]) -> None:
    buckets = res.get("buckets")
    _errs_if(not isinstance(buckets, dict), "zh-sentence: buckets 缺失", errs)
    if isinstance(buckets, dict):
        for name, b in buckets.items():
            if not isinstance(b, dict):
                continue
            for k in ("n", "invalid"):
                _errs_if(not isinstance(b.get(k), int) or b[k] < 0,
                         f"zh-sentence.{name}: {k} 应为非负整数", errs)
            for k in ("top1", "top9"):
                v = b.get(k)
                _errs_if(v is not None and not _is_ratio(v),
                         f"zh-sentence.{name}: {k} 不是 [0,1]", errs)
            keys = b.get("avg_keystrokes")
            _errs_if(keys is not None and (not _is_num(keys) or keys < 0),
                     f"zh-sentence.{name}: avg_keystrokes 非法", errs)
    agg = res.get("aggregate")
    _errs_if(not isinstance(agg, dict), "zh-sentence: aggregate 缺失", errs)
    rerank = res.get("rerank")
    _errs_if(rerank is not None and not isinstance(rerank, dict),
             "zh-sentence: rerank 应为 dict 或 null", errs)
    if isinstance(rerank, dict):
        for k in ("top1_no_lm", "top1_lm"):
            v = rerank.get(k)
            _errs_if(v is not None and not _is_ratio(v), f"zh-sentence.rerank.{k} 非法", errs)
        for k in ("saved", "broken"):
            _errs_if(not isinstance(rerank.get(k), int), f"zh-sentence.rerank.{k} 应为整数", errs)


_RESULT_CHECKERS = {
    "yue-lengths": _check_yue_lengths,
    "yue-guard": _check_yue_guard,
    "zh-sentence": _check_zh_sentence,
}


def validate(doc: dict) -> list[str]:
    """校验一份基准结果 JSON，返回错误列表（空列表 = 合法）。"""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["顶层必须是 JSON object"]
    for k in ("suite", "suite_version", "status", "quick", "generated_at",
              "duration_s", "params", "environment", "results"):
        _errs_if(k not in doc, f"缺信封字段 {k}", errs)
    if errs:
        return errs
    _errs_if(doc["suite"] not in SUITES, f"suite 必须是 {SUITES} 之一", errs)
    _errs_if(not isinstance(doc["suite_version"], str) or not doc["suite_version"],
             "suite_version 必须是非空字符串（semver）", errs)
    _errs_if(doc["status"] not in STATUSES, f"status 必须是 {STATUSES} 之一", errs)
    _errs_if(not isinstance(doc["quick"], bool), "quick 必须是布尔", errs)
    _errs_if(not isinstance(doc["generated_at"], str), "generated_at 必须是 ISO 时间字符串", errs)
    _errs_if(not _is_num(doc["duration_s"]) or doc["duration_s"] < 0,
             "duration_s 必须是非负数", errs)
    _errs_if(not isinstance(doc["params"], dict), "params 必须是 object", errs)
    _errs_if(not isinstance(doc["environment"], dict), "environment 必须是 object", errs)

    status, suite, results = doc["status"], doc["suite"], doc["results"]
    if status == "skipped":
        _errs_if(results is not None, "skipped 时 results 必须为 null", errs)
        _errs_if(not isinstance(doc.get("reason"), str) or not doc["reason"],
                 "skipped 时必须有 reason", errs)
        return errs
    _errs_if(not isinstance(results, dict), f"{suite}: results 必须是 object", errs)
    if isinstance(results, dict) and suite in _RESULT_CHECKERS:
        _RESULT_CHECKERS[suite](results, errs)
        _check_v1_1_fields(suite, results, errs)
    system = doc.get("system")
    _errs_if(system is not None and (not isinstance(system, str) or not system.strip()),
             "system（可选）必须是非空字符串", errs)
    return errs
