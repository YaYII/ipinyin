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

import argparse
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
        major, minor, _ = (int(x) for x in mod.SUITE_VERSION.split("."))
        assert (major, minor) >= (1, 1), \
            f"{name}: v1.1.0 字段（per_item/wilson/inputs）要求版本 ≥ 1.1.0"
    import suite_yue_guard, suite_yue_lengths  # noqa: F401
    for mod in (suite_yue_lengths, suite_yue_guard):
        assert mod.SUITE_VERSION >= "1.3.0", \
            f"{mod.SUITE}: 词桶合法变体判分（accept 字段）要求版本 ≥ 1.3.0"


# ---------------------------------------------------------------- v1.1.0 新字段

from bench_common import fingerprint, question_hash, wilson  # noqa: E402


def test_wilson_known_values():
    iv = wilson(6, 10)
    assert 0.28 < iv["lo"] < 0.34 and 0.80 < iv["hi"] < 0.86
    full = wilson(10, 10)                       # 全对也有下界，不许报 0~1
    assert 0.70 < full["lo"] < 0.75 and full["hi"] == 1.0
    zero = wilson(0, 10)
    assert zero["lo"] == 0.0 and 0.2 < zero["hi"] < 0.32
    assert wilson(0, 0) is None                 # 空桶无区间
    for k, n in ((3, 50), (47, 50), (0, 1), (1, 1)):
        iv = wilson(k, n)
        assert 0.0 <= iv["lo"] <= iv["hi"] <= 1.0


def test_question_hash_stable_and_sensitive():
    a = [{"id": "2字:0", "bucket": "2字", "input": "很不"},
         {"id": "2字:1", "bucket": "2字", "input": "得他"}]
    b = [{"id": "2字:0", "bucket": "2字", "input": "很可"},
         {"id": "2字:1", "bucket": "2字", "input": "得他"}]
    assert question_hash(a) == question_hash([dict(x) for x in a])   # 字段序无关
    assert question_hash(a) != question_hash(list(reversed(a)))      # 题序变 → hash 变
    assert question_hash(a) != question_hash(b)                      # 题面变 → hash 变
    assert len(question_hash(a)) == 16


def test_v110_fields_validate_and_reject():
    res = _yue_lengths_results()
    res["buckets"]["2字"]["wilson"] = {"rule": {"lo": 0.7, "hi": 0.1}}   # lo > hi 非法
    errs = validate(_envelope("yue-lengths", res, suite_version="1.1.0"))
    assert any("wilson" in e for e in errs)

    res = _yue_lengths_results()
    res["inputs"] = [{"path": "vendor/yue_pairs.json", "sha256_16": "0123456789abcdef",
                      "bytes": 123}]
    res["question_set_hash"] = "0123456789abcdef"
    res["per_item"] = [{"id": "2字:0", "bucket": "2字", "rule_ok": True,
                        "neural_ok": None}]
    assert validate(_envelope("yue-lengths", res, suite_version="1.1.0")) == []

    res["inputs"][0]["sha256_16"] = "short"
    assert any("inputs" in e for e in validate(_envelope("yue-lengths", res)))
    res["inputs"][0]["sha256_16"] = "0123456789abcdef"
    res["per_item"].append({"id": "2字:0", "bucket": "2字"})      # id 重复
    assert any("重复" in e for e in validate(_envelope("yue-lengths", res)))

    doc = _envelope("yue-lengths", _yue_lengths_results(), system="外部模型")
    assert validate(doc) == []                                    # system 可选字段
    doc = _envelope("yue-lengths", _yue_lengths_results(), system=" ")
    assert validate(doc)                                          # 空白 system 拒绝


def test_v100_docs_still_validate():
    """向后兼容：不带 v1.1.0 新字段的旧结果文件必须仍过校验。"""
    assert validate(_envelope("yue-lengths", _yue_lengths_results())) == []
    assert validate(_envelope("yue-guard", _yue_guard_results())) == []
    assert validate(_envelope("zh-sentence", _zh_results())) == []


# ---------------------------------------------------------------- compare.py

import compare as compare_mod  # noqa: E402


def _cmp_doc(system: str, outcomes: dict[str, bool]) -> dict:
    return {
        "suite": "yue-lengths", "suite_version": "1.1.0", "system": system,
        "status": "ok", "quick": True,
        "results": {
            "question_set_hash": "0123456789abcdef",
            "per_item": [{"id": pid, "bucket": "2字", "rule_ok": ok,
                          "neural_ok": ok} for pid, ok in outcomes.items()],
        },
    }


def test_mcnemar_exact_math():
    assert compare_mod.mcnemar_exact(0, 0) == 1.0
    assert abs(compare_mod.mcnemar_exact(5, 0) - 0.0625) < 1e-9
    assert abs(compare_mod.mcnemar_exact(10, 0) - 2 * 0.5 ** 10) < 1e-9
    assert compare_mod.mcnemar_exact(3, 3) == 1.0            # 对称 → 无差异
    assert compare_mod.mcnemar_exact(8, 0) < compare_mod.mcnemar_exact(5, 0)


def test_compare_counts_and_guards(tmp_path):
    a = _cmp_doc("A", {"2字:0": True, "2字:1": True, "2字:2": False,
                       "2字:3": False, "2字:4": True})
    b_items = {"2字:0": True, "2字:1": False, "2字:2": False,
               "2字:3": True, "2字:4": True}
    b = _cmp_doc("B", b_items)
    rep = compare_mod.compare(a, b)
    row = rep["rows"]["2字"]
    assert row["a_only"] == 1 and row["b_only"] == 1     # :1 A对B错、:3 B对A错
    assert row["n"] == 5 and row["mcnemar_p"] == 1.0
    assert row["verdict"] == "在噪声里"

    for bad, frag in ((dict(b, results={"question_set_hash": "ffffffffffffffff"}),
                       "question_set_hash"),
                      (dict(b, quick=False), "quick"),
                      ({"suite": "zh-sentence"}, "suite")):
        try:
            compare_mod.compare(a, bad)
            raise AssertionError(f"应拒绝：{frag}")
        except SystemExit as exc:
            assert frag in str(exc)


# ---------------------------------------------------------------- 导出 → 评分往返

def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_export_score_roundtrip_yue(tmp_path):
    """导出题面 → 用标准答案构造满分/规则提交 → 评分应复现内部两列。"""
    if not have(bench_common.DEFAULT_PAIRS):
        pytest.skip("vendor/yue_pairs.json 不存在")
    import export_questions as ex
    import score_submission as sc
    import suite_yue_lengths as s

    argv = argparse.Namespace(pairs=bench_common.DEFAULT_PAIRS, model="", guard="",
                              lexicon="", limit=None, n=None)
    questions, answers, _echo, version, ctx = ex._questions_for(
        "yue-lengths", "quick", argv)
    assert version == s.SUITE_VERSION
    exp = {a["id"]: a["expected"] for a in answers}
    _write_jsonl(tmp_path / "q.jsonl", questions)
    _write_jsonl(tmp_path / "perfect.jsonl",
                 [{"id": q["id"], "output": exp[q["id"]]} for q in questions])
    _write_jsonl(tmp_path / "rule.jsonl",
                 [{"id": q["id"], "output": ctx["conv"].convert(q["input"])}
                  for q in questions])
    manifest = {"suite": "yue-lengths", "suite_version": version, "mode": "quick",
                "question_set_hash": question_hash(questions),
                "answers_sha256_16": __import__("hashlib").sha256(
                    json.dumps(answers, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest()[:16]}
    sc._verify(manifest, questions, answers)                 # 防调包校验通过
    acc = {a["id"]: a.get("accept") for a in answers}
    perfect = sc._score_lengths(questions, exp, acc,
                                {r["id"]: r for r in sc._read_jsonl(tmp_path / "perfect.jsonl")},
                                0, argv, ctx)
    rule = sc._score_lengths(questions, exp, acc,
                             {r["id"]: r for r in sc._read_jsonl(tmp_path / "rule.jsonl")},
                             0, argv, ctx)
    assert perfect["weighted"]["weighted_neural"] == 1.0     # 标准答案 → 满分
    assert perfect["buckets"]["4字成语"]["copy_baseline"] >= 0.95
    # 内部评测跑一遍，提交评分的 neural 列应与内部 rule 列逐桶一致
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path), nn="off"))
    assert status == "ok"
    for name, b in results["buckets"].items():
        assert rule["buckets"][name]["neural"] == b["rule"], name


def test_score_rejects_foreign_id_and_missing_questions(tmp_path):
    """评分端两道闸：题面/清单缺失必须拒绝；未知 id 必须拒绝（防调包）。"""
    rows = [{"id": "不存在:0", "output": "x"}]
    _write_jsonl(tmp_path / "bad.jsonl", rows)
    import subprocess
    proc = subprocess.run(
        [PY, os.path.join(BENCH_DIR, "score_submission.py"), "--suite", "yue-lengths",
         "--manifest", str(tmp_path / "nope.manifest.json"),
         "--questions", str(tmp_path / "nope.questions.jsonl"),
         "--predictions", str(tmp_path / "bad.jsonl"), "--system", "X"],
        capture_output=True, text=True, cwd=BENCH_DIR)
    assert proc.returncode != 0                              # 无题面/清单 → 拒绝
    assert "题面/清单缺失" in (proc.stderr + proc.stdout)


# ---------------------------------------------------------------- leaderboard

def test_accept_forms_variants_and_schema():
    """v1.3.0：词桶判分接受训练侧实证的合法变体。"""
    from bench_common import accept_forms
    class _Shell:
        @staticmethod
        def to_traditional(s):
            return s.replace("么", "麼")
    variants = {"什么": {"乜嘢", "咩"}}
    forms = accept_forms("什么", "乜嘢", variants, shell=_Shell)
    assert "乜嘢" in forms and "咩" in forms       # 标准答案 + 实证变体都在
    forms2 = accept_forms("什么", "什麼", {"什么": {"乜嘢"}}, shell=_Shell)
    assert forms2 == ["乜嘢", "什麼"]              # shell 后排序稳定
    assert accept_forms("不知道", "唔知道", {}, shell=_Shell) == ["唔知道"]

    res = _yue_lengths_results()
    res["per_item"] = res.get("per_item") or []
    res["per_item"].append({"id": "2字:0", "bucket": "2字", "expected": "乜嘢",
                            "accept": ["什麼", "乜嘢"], "neural_ok": True})
    assert validate(_envelope("yue-lengths", res, suite_version="1.3.0")) == []
    res["per_item"][0]["accept"] = ["", 3]          # 非法变体列表
    assert any("accept" in e for e in validate(_envelope("yue-lengths", res)))


def test_quick_word_bucket_accepts_variants(tmp_path):
    """端到端：词桶 per_item 的 accept 必须含标准答案的繁体形。"""
    if not have(bench_common.DEFAULT_PAIRS):
        pytest.skip("vendor/yue_pairs.json 不存在")
    import suite_yue_lengths as s
    status, results, _ = s.run(_base_params(out_dir=str(tmp_path), nn="off"))
    assert status == "ok"
    words = [it for it in results["per_item"] if it["bucket"] in ("2字", "3字")]
    if not words:
        pytest.skip("词桶为空（语料太小）")
    for it in words:
        assert it["accept"] and all(a for a in it["accept"])


def test_leaderboard_render(tmp_path):
    import leaderboard as lb
    docs = [
        _envelope("yue-lengths", {**_yue_lengths_results(),
                                  "weighted": {"weighted_rule": 0.67,
                                               "weighted_neural": 0.76,
                                               "weight_coverage": 0.8}},
                  system="甲", suite_version="1.1.0"),
        _envelope("yue-lengths", {**_yue_lengths_results(),
                                  "weighted": {"weighted_rule": 0.90,
                                               "weighted_neural": None,
                                               "weight_coverage": 0.8}},
                  system="乙", suite_version="1.1.0"),
        _envelope("yue-guard", None, status="skipped", reason="无模型"),
    ]
    rdir = tmp_path / "results"
    rdir.mkdir()
    for i, d in enumerate(docs):
        (rdir / f"doc{i}.json").write_text(
            json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tables = lb._rows(str(rdir))
    assert len(tables["yue-lengths"]) == 2
    # 乙的 weighted_neural 为 None → 回退 weighted_rule 0.90 > 甲 0.76，乙排第一
    assert tables["yue-lengths"][0]["system"] == "乙"
    assert tables["yue-lengths"][0]["value"] == 0.90
    md = lb.render(tables)
    assert "不具排名资格" in md and "甲" in md and "乙" in md
    assert "skipped" not in md                                # skipped 不进榜


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
