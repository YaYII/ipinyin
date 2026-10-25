#!/usr/bin/env python3
"""基准套件自身的 pytest：

  1. JSON schema 校验器：合法信封零错误、坏结构能抓出来；
  2. --quick 管线可跑通：yue-lengths / zh-sentence / yue-guard（无 torch 依赖的
     路径用桩模型注入，真实大数据缺失时逐测试跳过，不挂 CI）；
  3. graceful skip：yue-guard 缺模型时输出 {"status":"skipped"}；
  4. 套件注册表与 schema、文档三方一致（版本化纪律）。

运行： python3 test_runner.py -v   （或 $PY -m pytest test_runner.py -v）
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys

import pytest

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)

import bench_common  # noqa: E402
import schema_check  # noqa: E402
from schema_check import validate  # noqa: E402

PY = sys.executable


def have(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


# ---------------------------------------------------------------- 基础信封

def _envelope(suite: str, results, status: str = "ok", **over) -> dict:
    doc = {
        "suite": suite,
        "suite_version": "1.0.0",
        "spec": "projects/06-benchmark/schemas.md (envelope v1)",
        "status": status,
        "quick": True,
        "generated_at": "2026-10-25T12:00:00+08:00",
        "duration_s": 1.0,
        "params": {"quick": True},
        "environment": {"python": "3.13", "platform": "test", "torch": None},
        "reason": None,
        "results": results,
    }
    doc.update(over)
    return doc


def _yue_lengths_results() -> dict:
    return {
        "buckets": {
            "2字": {"n": 50, "copy_baseline": 0.0, "rule": 0.6, "neural": 0.86,
                    "neural_median_latency_ms": 12.5, "weight": 0.4},
            "4字成语": {"n": 40, "copy_baseline": 1.0, "rule": 0.975, "neural": None,
                       "neural_median_latency_ms": None, "weight": 0.25},
        },
        "weighted": {"weighted_rule": 0.6713, "weighted_neural": 0.7617,
                     "weight_coverage": 0.65},
        "empty_buckets": ["3字"],
        "neural_note": None,
    }


def _yue_guard_results() -> dict:
    return {
        "buckets": {
            "2字": {"n": 50, "copy_baseline": None, "rule": 0.6, "neural": 0.86,
                    "guarded": 0.9, "rejected": 3, "accepted": 47, "saved": 2,
                    "lost": 0, "guarded_median_latency_ms": 0.02, "weight": None},
        },
        "aggregate": {"rejected": 3, "saved": 2, "lost": 0},
        "examples": {"2字": [{"input": "咁", "neural": "咁", "guarded": "咁",
                             "expected": "咁", "outcome": "saved"}]},
        "empty_buckets": [],
        "neural_note": None,
    }


def _zh_results() -> dict:
    return {
        "buckets": {
            "2-4字": {"n": 23, "invalid": 27, "top1": 0.957, "top9": 1.0,
                      "avg_keystrokes": 10.9, "fixed": 1, "broken": 0},
        },
        "aggregate": {"n": 23, "top1": 0.957, "top9": 1.0},
        "rerank": {"lm": "data/lm_char.marshal", "lm_weight": 0.4,
                   "top1_no_lm": 0.913, "top1_lm": 0.957, "saved": 1,
                   "broken": 0, "net": 1},
        "empty_buckets": [],
        "questions_scanned": 20000,
    }


# ---------------------------------------------------------------- schema 校验

def test_schema_accepts_all_suites():
    for suite, res in (("yue-lengths", _yue_lengths_results()),
                       ("yue-guard", _yue_guard_results()),
                       ("zh-sentence", _zh_results())):
        assert validate(_envelope(suite, res)) == [], suite


def test_schema_accepts_skipped():
    doc = _envelope("yue-guard", None, status="skipped", reason="模型文件不存在")
    assert validate(doc) == []


def test_schema_rejects_missing_envelope_fields():
    doc = _envelope("yue-lengths", _yue_lengths_results())
    del doc["suite_version"]
    errs = validate(doc)
    assert any("suite_version" in e for e in errs)


def test_schema_rejects_bad_status_and_unknown_suite():
    assert validate(_envelope("yue-lengths", _yue_lengths_results(), status="fine"))
    assert validate(_envelope("nope", {}))


def test_schema_rejects_out_of_range_ratios():
    res = _yue_lengths_results()
    res["buckets"]["2字"]["rule"] = 1.5
    assert validate(_envelope("yue-lengths", res))


def test_schema_rejects_nan_and_negative_counts():
    res = _yue_lengths_results()
    res["buckets"]["2字"]["neural"] = float("nan")
    errs = validate(_envelope("yue-lengths", res))
    assert any("neural" in e for e in errs)
    res2 = _yue_guard_results()
    res2["buckets"]["2字"]["saved"] = -1
    assert validate(_envelope("yue-guard", res2))


def test_schema_rejects_skipped_without_reason():
    doc = _envelope("yue-guard", None, status="skipped")
    assert any("reason" in e for e in validate(doc))


# ---------------------------------------------------------------- --quick 管线

def _base_params(**over) -> dict:
    p = {"quick": True, "out_dir": os.path.join(BENCH_DIR, "results"),
         "pairs": bench_common.DEFAULT_PAIRS, "model": bench_common.DEFAULT_MODEL,
         "guard": bench_common.DEFAULT_GUARD, "lm": "", "lm_weight": 0.4,
         "limit": None, "n": None, "ctx_max": 7, "show": 3, "nn": "stub",
         "device": "cpu"}
    p.update(over)
    return p


def test_quick_yue_lengths_stub(tmp_path):
    if not have(bench_common.DEFAULT_PAIRS):
        pytest.skip("vendor/yue_pairs.json 不存在（先跑 make corpus）")
    import suite_yue_lengths as s
    params = _base_params(out_dir=str(tmp_path))
    status, results, reason = s.run(params)
    assert status == "ok" and reason is None
    doc = _envelope(s.SUITE, results)
    assert validate(doc) == [], validate(doc)
    for name, b in results["buckets"].items():
        assert b["n"] <= 50, f"--quick 每桶应 ≤50：{name}={b['n']}"
        assert 0.0 <= b["rule"] <= 1.0 and 0.0 <= b["copy_baseline"] <= 1.0
    # 对照桶的「照抄不改」必须接近满分，否则题面本身坏了
    assert results["buckets"]["4字成语"]["copy_baseline"] >= 0.95


def test_quick_yue_lengths_real_model(tmp_path):
    if not (have(bench_common.DEFAULT_PAIRS) and have(bench_common.DEFAULT_MODEL)):
        pytest.skip("语料或模型不存在")
    pytest.importorskip("torch")
    import suite_yue_lengths as s
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path), nn="auto"))
    assert status == "ok"
    neural = [b["neural"] for b in results["buckets"].values()]
    assert all(v is not None for v in neural)
    assert all(b["neural_median_latency_ms"] is not None
               for b in results["buckets"].values())


def test_quick_yue_guard_stub(tmp_path):
    if not have(bench_common.DEFAULT_PAIRS):
        pytest.skip("vendor/yue_pairs.json 不存在")
    if not have(bench_common.DEFAULT_GUARD):
        pytest.skip("data/yue_guard.eval.marshal 不存在（评测口径数据只在本机）")
    import suite_yue_guard as s

    # 破坏性桩：把成语里的字乱改（无实证的替换），守卫必须否掉并救回
    chaotic = lambda t: t.replace("守", "收").replace("画", "划")  # noqa: E731
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path), _nn_fn=chaotic))
    assert status == "ok"
    doc = _envelope(s.SUITE, results)
    assert validate(doc) == [], validate(doc)
    b = results["buckets"]["4字成语"]
    assert b["rejected"] > 0 and b["saved"] > 0      # 无实证改动被否决 → 回退规则救回
    assert b["guarded"] >= b["neural"]               # 守卫不应比裸神经更差
    # 恒等桩：输出没改动，守卫必须全部放行（不制造假否决）
    status2, results2, _ = s.run(_base_params(out_dir=str(tmp_path), _nn_fn=lambda t: t))
    assert status2 == "ok"
    assert all(bk["rejected"] == 0 for bk in results2["buckets"].values())


def test_yue_guard_graceful_skip_when_model_missing(tmp_path):
    import suite_yue_guard as s
    status, results, reason = s.run(_base_params(
        out_dir=str(tmp_path), nn="auto", model="/nonexistent/yue_model.pt"))
    assert status == "skipped" and results is None and reason
    doc = _envelope(s.SUITE, None, status="skipped", reason=reason or "")
    assert validate(doc) == []


def test_quick_zh_sentence(tmp_path):
    if not (have(bench_common.DEFAULT_PAIRS) and have(bench_common.DEFAULT_LEXICON)):
        pytest.skip("语料或词库不存在")
    import suite_zh_sentence as s
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path)))
    assert status == "ok"
    doc = _envelope(s.SUITE, results)
    assert validate(doc) == [], validate(doc)
    assert results["rerank"] is None                  # 没传 lm 就没有重排对照
    for b in results["buckets"].values():
        assert b["n"] <= 50 and b["invalid"] >= 0     # 自洽过滤的剔除数透明可查
        assert 0.0 <= b["top1"] <= b["top9"] + 1e-9   # Top1 ≤ 前9


def test_quick_zh_sentence_with_lm_and_stub_candidates(tmp_path):
    if not (have(bench_common.DEFAULT_PAIRS) and have(bench_common.DEFAULT_LEXICON)):
        pytest.skip("语料或词库不存在")
    if not have(bench_common.DEFAULT_LM):
        pytest.skip("data/lm_char.marshal 不存在（生成物不入库）")
    import suite_zh_sentence as s
    # 桩候选器：永远把输入原文当第一候选 —— 验证 candidate_fn 注入接口
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path),
                                            lm=bench_common.DEFAULT_LM,
                                            candidate_fn=lambda key: [key]))
    assert status == "ok"
    assert results["rerank"] is not None
    assert results["rerank"]["saved"] + results["rerank"]["broken"] >= 0


def test_run_all_cli_end_to_end(tmp_path):
    if not have(bench_common.DEFAULT_PAIRS):
        pytest.skip("vendor/yue_pairs.json 不存在")
    out = str(tmp_path)
    proc = subprocess.run(
        [PY, os.path.join(BENCH_DIR, "run_all.py"),
         "--suite", "zh-sentence", "--quick", "--out", out],
        capture_output=True, text=True, cwd=BENCH_DIR)
    assert proc.returncode == 0, proc.stderr
    path = os.path.join(out, "zh-sentence__quick.json")
    assert os.path.exists(path)
    doc = json.load(open(path, encoding="utf-8"))
    assert validate(doc) == [], validate(doc)


def test_committed_results_match_schema():
    results_dir = os.path.join(BENCH_DIR, "results")
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")] \
        if os.path.isdir(results_dir) else []
    if not files:
        pytest.skip("results/ 还没有产出 JSON")
    for f in files:
        doc = json.load(open(os.path.join(results_dir, f), encoding="utf-8"))
        assert validate(doc) == [], f


# ---------------------------------------------------------------- 版本化纪律

def test_registry_schema_and_versions_consistent():
    import run_all
    assert set(run_all.SUITE_MODULES) == set(schema_check.SUITES)
    semver = re.compile(r"^\d+\.\d+\.\d+$")
    for name in schema_check.SUITES:
        mod = __import__(run_all.SUITE_MODULES[name])
        assert semver.match(mod.SUITE_VERSION), f"{name}: suite_version 必须 semver"
        assert mod.SUITE == name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
