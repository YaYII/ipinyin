#!/usr/bin/env python3
"""导出题面包：把基准题面（不含答案）打包给外部系统作答（C-Eval 模式）。

产出（写到 --out，默认 questions/）：
  <suite>__<mode>.questions.jsonl   每行一题：yue 套件 {"id","bucket","input"}；
                                    zh-sentence {"id","bucket","key"}（拼音键串）
  <suite>__<mode>.manifest.json     题面清单：suite/版本/模式/题数/分桶计数/
                                    question_set_hash/answers_sha256_16/复现参数

答案不随包分发（评分在仓库侧用 score_submission.py 重算）——但我们语料本就
开源，藏答案防不了有意过拟合，防的是题面调包与口径漂移（见 METHODOLOGY §10）。

用法：
  python3 export_questions.py --suite yue-lengths --mode quick
  python3 export_questions.py --suite all --mode full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from bench_common import BENCH_DIR, DEFAULT_GUARD, DEFAULT_LEXICON, DEFAULT_LM, \
    DEFAULT_MODEL, DEFAULT_PAIRS, now_iso, question_hash

sys.path.insert(0, BENCH_DIR)

OUT_DIR = os.path.join(BENCH_DIR, "questions")


def _questions_for(name: str, mode: str, argv) -> tuple[list[dict], list[dict], dict, str, dict]:
    """题面 + 答案 + 复现参数 + 口径版本 + 判分上下文。

    三个套件各自的 question_sets / select_questions；ctx 供 score_submission
    复用（yue：conv/t2s 转换器；zh：无需）。
    """
    params = {
        "quick": mode == "quick",
        "pairs": argv.pairs,
        "model": argv.model,
        "guard": argv.guard,
        "lexicon": argv.lexicon,
        "limit": argv.limit or None,
        "n": argv.n or None,
    }
    if name == "zh-sentence":
        import suite_zh_sentence as s
        if not os.path.exists(params["pairs"]) or not os.path.exists(params["lexicon"]):
            raise SystemExit(f"语料/词库缺失：{params['pairs']} / {params['lexicon']}")
        valid, _invalid, _scanned = s.select_questions(params)
        questions, answers = [], []
        for bucket, items in valid.items():
            for i, (key, sentence) in enumerate(items):
                questions.append({"id": f"{bucket}:{i}", "bucket": bucket, "key": key})
                answers.append({"id": f"{bucket}:{i}", "bucket": bucket,
                                "expected": sentence})
        return (questions, answers,
                {"quick": params["quick"], "pairs": params["pairs"],
                 "lexicon": params["lexicon"], "n": params["n"]},
                s.SUITE_VERSION, {})
    import suite_yue_guard
    import suite_yue_lengths
    mod = {"yue-lengths": suite_yue_lengths, "yue-guard": suite_yue_guard}[name]
    if not os.path.exists(params["pairs"]):
        raise SystemExit(f"语料缺失：{params['pairs']}（先跑 `make corpus`）")
    questions, answers, conv, t2s = mod.question_sets(params)
    return (questions, answers,
            {"quick": params["quick"], "pairs": params["pairs"],
             "limit": params["limit"]},
            mod.SUITE_VERSION, {"conv": conv, "t2s": t2s})


def export(name: str, mode: str, out_dir: str, argv) -> tuple[str, str]:
    questions, answers, echo, suite_version, _ctx = _questions_for(name, mode, argv)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{name}__{mode}"

    q_path = os.path.join(out_dir, f"{stem}.questions.jsonl")
    with open(q_path, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")

    canon = json.dumps(answers, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
    answers_hash = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]

    per_bucket: dict[str, int] = {}
    for q in questions:
        per_bucket[q["bucket"]] = per_bucket.get(q["bucket"], 0) + 1
    manifest = {
        "suite": name,
        "suite_version": suite_version,
        "mode": mode,
        "created_at": now_iso(),
        "question_count": len(questions),
        "bucket_counts": per_bucket,
        "question_set_hash": question_hash(questions),
        "answers_sha256_16": answers_hash,
        "params": echo,
        "note": ("答案不随包分发；评分用仓库侧 score_submission.py 重算并核对本清单的 "
                 "answers_sha256_16 / question_set_hash。口径版本随导出时的 suite 模块"
                 "（见 schemas.md 的版本化规则）。"),
    }
    m_path = os.path.join(out_dir, f"{stem}.manifest.json")
    with open(m_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return q_path, m_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all",
                    choices=["all", "yue-lengths", "yue-guard", "zh-sentence"])
    ap.add_argument("--mode", default="quick", choices=["quick", "full"],
                    help="quick=每桶 ≤50 题；full=全量口径（与 run_all 同 cap）")
    ap.add_argument("--out", default=OUT_DIR, help="导出目录（默认 questions/）")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--guard", default=DEFAULT_GUARD)
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON)
    ap.add_argument("--limit", type=int, default=0, help="语料扫描句对数上限")
    ap.add_argument("--n", type=int, default=0, help="zh-sentence 每桶题数上限")
    args = ap.parse_args(argv)

    suites = ["yue-lengths", "yue-guard", "zh-sentence"] if args.suite == "all" \
        else [args.suite]
    for name in suites:
        q_path, m_path = export(name, args.mode, args.out, args)
        manifest = json.load(open(m_path, encoding="utf-8"))
        print(f"[ok] {name} ({args.mode}) {manifest['question_count']} 题 "
              f"hash={manifest['answers_sha256_16']}")
        print(f"     题面 {os.path.relpath(q_path, BENCH_DIR)}")
        print(f"     清单 {os.path.relpath(m_path, BENCH_DIR)}")
    print("\n下一步：外部系统对题面作答 → predictions.jsonl → "
          "score_submission.py 评分（见 README「提交新系统」）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
