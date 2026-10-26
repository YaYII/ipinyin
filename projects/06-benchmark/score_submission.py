#!/usr/bin/env python3
"""评分外部提交：predictions.jsonl → 标准结果信封（与内部评测同 schema）。

流程（C-Eval 模式的仓库侧）：
  1. 按题面清单的参数在本仓库重建标准答案（答案从不随包分发）；
  2. 核对 manifest 与题面文件：question_set_hash / answers_sha256_16 防调包；
  3. 逐题判分（复用各 suite 的口径：yue = 繁体外壳后比对；zh = 首选/前9）；
     yue-guard 还会把我们的守卫套在外部输出上，给出 guarded / saved / lost；
  4. 产出与内部评测完全同构的信封 JSON，system 字段为提交方系统名，
     可直接进 leaderboard.py 汇总。

predictions.jsonl 格式（每行一题，必须覆盖题面全部 id，不得有未知 id）：
  yue 套件：      {"id": "4字成语:3", "output": "亡羊補牢"}
  zh-sentence：   {"id": "5-7字:2", "candidates": ["今日天氣好", "今日"]}

用法：
  python3 score_submission.py --suite yue-lengths --mode quick \
      --system MyModel --predictions mypreds.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict

from bench_common import (BENCH_DIR, DEFAULT_GUARD, DEFAULT_LEXICON,
                          DEFAULT_PAIRS, env_info, file_ok, fingerprint,
                          now_iso, question_hash, r4, wilson)
from export_questions import _questions_for
from schema_check import validate

sys.path.insert(0, BENCH_DIR)

SHOW_EXAMPLES = 5


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{ln} 不是合法 JSON：{exc}")
    return rows


def _verify(manifest: dict, questions: list[dict], answers: list[dict]) -> None:
    errs = []
    if question_hash(questions) != manifest.get("question_set_hash"):
        errs.append("题面文件与 manifest 的 question_set_hash 不一致（题面被改过？）")
    canon = json.dumps(answers, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
    if hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16] != manifest.get("answers_sha256_16"):
        errs.append("重建答案与 manifest 的 answers_sha256_16 不一致（语料或参数不同）")
    if errs:
        raise SystemExit("；".join(errs))


def _bucket(name: str, n: int, **kw) -> dict:
    return {"n": n, **kw}


def _score_lengths(questions: list[dict], exp: dict, acc: dict, pred: dict,
                   missing: int, args, ctx: dict) -> dict:
    """yue-lengths：外部输出扮演 neural 列；照抄/规则两列本地重算作对照。

    exp: id→标准答案；acc: id→可接受（繁体化）答案集合（v1.3.0 词桶变体）。
    """
    from eval_ime_lengths import WEIGHTS
    t2s, conv = ctx["t2s"], ctx["conv"]

    acc_sets = {i: set(acc.get(i) or [exp[i]]) for i in exp}
    ok_sets: "OrderedDict[str, dict]" = OrderedDict()
    for q in questions:
        b = ok_sets.setdefault(q["bucket"], {"n": 0, "c": 0, "r": 0, "k": 0})
        b["n"] += 1
        accepted = acc_sets[q["id"]]
        if t2s.to_traditional(q["input"]) in accepted:
            b["c"] += 1
        if t2s.to_traditional(conv.convert(q["input"])) in accepted:
            b["r"] += 1
        row = pred.get(q["id"])
        out = row.get("output") if row else None
        if out is not None and t2s.to_traditional(out) in accepted:
            b["k"] += 1

    buckets, w_rule, w_nn, w_tot = OrderedDict(), 0.0, 0.0, 0.0
    for name, b in ok_sets.items():
        buckets[name] = _bucket(
            name, b["n"],
            copy_baseline=r4(b["c"] / b["n"]), rule=r4(b["r"] / b["n"]),
            neural=r4(b["k"] / b["n"]), neural_median_latency_ms=None,
            weight=WEIGHTS.get(name, 0.0),
            wilson={"copy_baseline": wilson(b["c"], b["n"]),
                    "rule": wilson(b["r"], b["n"]),
                    "neural": wilson(b["k"], b["n"])})
        w_rule += b["r"] / b["n"] * buckets[name]["weight"]
        w_nn += b["k"] / b["n"] * buckets[name]["weight"]
        w_tot += buckets[name]["weight"]
    per_item = []
    for q in questions:
        accepted = acc_sets[q["id"]]
        out = (pred.get(q["id"]) or {}).get("output")
        per_item.append({"id": q["id"], "bucket": q["bucket"], "input": q["input"],
                         "expected": exp[q["id"]],
                         "neural_out": out,
                         "neural_ok": (None if out is None
                                       else t2s.to_traditional(out) in accepted)})
    return {
        "buckets": buckets,
        "weighted": {
            "weighted_rule": r4(w_rule / w_tot) if w_tot else None,
            "weighted_neural": r4(w_nn / w_tot) if w_tot else None,
            "weight_coverage": r4(w_tot),
        },
        "empty_buckets": [],
        "neural_note": f"{missing} 题未作答，按错计" if missing else None,
        "per_item": per_item,
    }


def _score_guard(questions: list[dict], exp: dict, acc: dict, pred: dict,
                 missing: int, args, ctx: dict) -> dict:
    """yue-guard：外部输出扮演神经裸输出，我们的守卫照常上（guarded 列）。"""
    from ipinyin.yue_guard import YueGuard
    t2s, conv = ctx["t2s"], ctx["conv"]
    if not file_ok(args.guard):
        raise SystemExit(f"yue-guard 评分需要守卫数据：{args.guard}"
                         f"（先跑 python3 tools/build_yue_guard.py）")
    guard = YueGuard.load(args.guard)
    if guard is None:
        raise SystemExit(f"守卫数据加载失败：{args.guard}")
    guard.ctx_max_len = int(args.ctx_max)

    acc_sets = {i: set(acc.get(i) or [exp[i]]) for i in exp}
    ok_sets: "OrderedDict[str, dict]" = OrderedDict()
    per_item: list[dict] = []
    examples: "OrderedDict[str, list[dict]]" = OrderedDict()
    tot = {"rejected": 0, "saved": 0, "lost": 0}
    for q in questions:
        b = ok_sets.setdefault(q["bucket"], {"n": 0, "r": 0, "k": 0, "g": 0,
                                             "rej": 0, "sv": 0, "ls": 0})
        b["n"] += 1
        accepted = acc_sets[q["id"]]
        expected, zh = exp[q["id"]], q["input"]
        rule_out = conv.convert(zh)
        rule_ok = t2s.to_traditional(rule_out) in accepted
        if rule_ok:
            b["r"] += 1
        row = pred.get(q["id"])
        nn_out = row.get("output") if row else None
        nn_ok = nn_out is not None and t2s.to_traditional(nn_out) in accepted
        if nn_ok:
            b["k"] += 1
        final = guard.check(zh, nn_out if nn_out is not None else zh,
                            repair=conv.convert)
        g_ok = t2s.to_traditional(final) in accepted
        if g_ok:
            b["g"] += 1
        repaired = final != (nn_out if nn_out is not None else zh)
        outcome = "pass"
        if repaired:
            b["rej"] += 1
            if rule_ok and not nn_ok:
                b["sv"] += 1
                outcome = "saved"
            elif nn_ok and not rule_ok:
                b["ls"] += 1
                outcome = "lost"
            elif g_ok:
                outcome = "fixed_right"
            else:
                outcome = "both_wrong"
            if len(examples.get(q["bucket"], [])) < SHOW_EXAMPLES:
                examples.setdefault(q["bucket"], []).append(
                    {"input": zh, "neural": nn_out, "guarded": final,
                     "expected": expected, "outcome": outcome})
        per_item.append({"id": q["id"], "bucket": q["bucket"], "input": zh,
                         "expected": expected, "rule_out": rule_out,
                         "rule_ok": rule_ok, "neural_out": nn_out,
                         "nn_ok": nn_ok, "guarded_out": final,
                         "guarded_ok": g_ok, "outcome": outcome,
                         "latency_ms": None})

    buckets = OrderedDict()
    for name, b in ok_sets.items():
        buckets[name] = _bucket(
            name, b["n"], copy_baseline=None,
            rule=r4(b["r"] / b["n"]), neural=r4(b["k"] / b["n"]),
            guarded=r4(b["g"] / b["n"]), rejected=b["rej"],
            accepted=b["n"] - b["rej"], saved=b["sv"], lost=b["ls"],
            guarded_median_latency_ms=None, weight=None,
            wilson={"rule": wilson(b["r"], b["n"]),
                    "neural": wilson(b["k"], b["n"]),
                    "guarded": wilson(b["g"], b["n"])})
        tot["rejected"] += b["rej"]
        tot["saved"] += b["sv"]
        tot["lost"] += b["ls"]
    return {
        "buckets": buckets,
        "aggregate": tot,
        "examples": dict(examples),
        "empty_buckets": [],
        "neural_note": f"{missing} 题未作答，按错计" if missing else None,
        "per_item": per_item,
    }


def _score_zh(questions: list[dict], exp: dict, pred: dict, missing: int,
              args, ctx: dict) -> dict:
    """zh-sentence：candidates[0] == 原句 → Top1；原句在前 9 → 前9。"""
    acc: "OrderedDict[str, dict]" = OrderedDict()
    per_item: list[dict] = []
    for q in questions:
        b = acc.setdefault(q["bucket"], {"n": 0, "k1": 0, "k9": 0, "keys": 0})
        b["n"] += 1
        row = pred.get(q["id"])
        cands = (row.get("candidates") if row else None) or []
        s = exp[q["id"]]
        top1 = cands[:1] == [s]
        top9 = s in cands[:9]
        if top1:
            b["k1"] += 1
        if top9:
            b["k9"] += 1
        b["keys"] += len(q["key"])
        per_item.append({"id": q["id"], "bucket": q["bucket"], "key": q["key"],
                         "expected": s, "top1_ok": top1, "top9_ok": top9,
                         "keystrokes": len(q["key"]), "base_top1_ok": None})
    buckets, tot_n, tot1, tot9 = OrderedDict(), 0, 0, 0
    for name, b in acc.items():
        buckets[name] = {
            "n": b["n"], "invalid": 0,
            "top1": r4(b["k1"] / b["n"]), "top9": r4(b["k9"] / b["n"]),
            "avg_keystrokes": r4(b["keys"] / b["n"]),
            "wilson": {"top1": wilson(b["k1"], b["n"]), "top9": wilson(b["k9"], b["n"])},
        }
        tot_n += b["n"]
        tot1 += b["k1"]
        tot9 += b["k9"]
    return {
        "buckets": buckets,
        "aggregate": {"n": tot_n, "top1": r4(tot1 / tot_n) if tot_n else None,
                      "top9": r4(tot9 / tot_n) if tot_n else None},
        "rerank": None,
        "empty_buckets": [],
        "per_item": per_item,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True,
                    choices=["yue-lengths", "yue-guard", "zh-sentence"])
    ap.add_argument("--mode", default="quick", choices=["quick", "full"],
                    help="与导出题面时的模式一致（可省略，manifest 里有）")
    ap.add_argument("--manifest", default="",
                    help="题面清单（默认 questions/<suite>__<mode>.manifest.json）")
    ap.add_argument("--questions", default="",
                    help="题面文件（默认 questions/<suite>__<mode>.questions.jsonl）")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--system", required=True, help="提交方系统名（进 leaderboard）")
    ap.add_argument("--out", default="", help="结果 JSON 路径（默认 results/ 下按系统命名）")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--guard", default=DEFAULT_GUARD,
                    help="yue-guard 评分需要守卫数据（给出 guarded/saved/lost）")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON)
    ap.add_argument("--ctx-max", type=int, default=5,
                    help="守卫语境门槛（与生产/套件默认一致；v1.3 裁决值）")
    args = ap.parse_args(argv)

    stem = f"{args.suite}__{args.mode}"
    m_path = args.manifest or os.path.join(BENCH_DIR, "questions", f"{stem}.manifest.json")
    q_path = args.questions or os.path.join(BENCH_DIR, "questions", f"{stem}.questions.jsonl")
    if not file_ok(m_path) or not file_ok(q_path):
        raise SystemExit(f"题面/清单缺失：先跑 export_questions.py --suite {args.suite} "
                         f"--mode {args.mode}（{m_path}）")
    manifest = json.load(open(m_path, encoding="utf-8"))

    questions, answers, echo, suite_version, ctx = _questions_for(
        args.suite, manifest.get("mode", args.mode),
        argparse.Namespace(pairs=args.pairs, model="", guard=args.guard,
                           lexicon=args.lexicon, limit=0, n=0))
    _verify(manifest, questions, answers)

    exp = {a["id"]: a["expected"] for a in answers}
    acc = {a["id"]: a.get("accept") for a in answers}    # v1.3.0 词桶变体集
    preds = _read_jsonl(args.predictions)
    pred: dict[str, dict] = {}
    for row in preds:
        pid = row.get("id")
        if pid not in exp:
            raise SystemExit(f"predictions 含未知 id：{pid!r}（题面调包？）")
        pred[pid] = row
    missing = sum(1 for pid in exp if pid not in pred)
    if missing:
        print(f"警告：{missing} 题未作答，按错计。")

    t0 = time.perf_counter()
    if args.suite == "zh-sentence":
        results = _score_zh(questions, exp, pred, missing, args, ctx)
    elif args.suite == "yue-guard":
        results = _score_guard(questions, exp, acc, pred, missing, args, ctx)
    else:
        results = _score_lengths(questions, exp, acc, pred, missing, args, ctx)
    duration = round(time.perf_counter() - t0, 2)

    envelope = {
        "suite": args.suite,
        "suite_version": manifest.get("suite_version", suite_version),
        "spec": "projects/06-benchmark/schemas.md (envelope v1)",
        "system": args.system,
        "status": "ok",
        "quick": manifest.get("mode", args.mode) == "quick",
        "generated_at": now_iso(),
        "duration_s": duration,
        "params": {
            "mode": "submission",
            "questions": os.path.relpath(q_path, BENCH_DIR),
            "predictions": (os.path.relpath(args.predictions, BENCH_DIR)
                            if os.path.isabs(args.predictions)
                            and args.predictions.startswith(BENCH_DIR)
                            else args.predictions),
            "missing_answers": missing,
            "manifest_created_at": manifest.get("created_at"),
        },
        "environment": env_info(),
        "reason": None,
        "results": {
            **results,
            "inputs": [fp for fp in
                       (fingerprint(args.predictions), fingerprint(q_path),
                        fingerprint(m_path)) if fp],
            "question_set_hash": manifest.get("question_set_hash"),
        },
    }
    errs = validate(envelope)
    if errs:
        raise SystemExit("评分结果未过 schema 校验：" + "; ".join(errs))

    safe = re.sub(r"[^\w.-]+", "_", args.system).strip("_") or "submission"
    out = args.out or os.path.join(BENCH_DIR, "results", f"{stem}__{safe}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[ok] {args.suite} × {args.system} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
