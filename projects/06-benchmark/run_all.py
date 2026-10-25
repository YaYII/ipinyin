#!/usr/bin/env python3
"""公开基准包统一入口：一个命令跑全部评测口径，输出结构化 JSON。

用法：
  python3 run_all.py --suite all --quick          # 分钟级冒烟（每桶 ≤50 题）
  python3 run_all.py --suite yue-lengths          # 单套件全量
  python3 run_all.py --suite zh-sentence --lm     # 整句评测 + 重排对照（救回/弄坏分开报）
  python3 run_all.py --suite yue-guard --nn off   # 无 torch 环境只看规则口径

每个 suite 输出 results/<suite>__<quick|full>.json，格式见 schemas.md。
退出码：有任一 suite 状态为 error 时非零；skipped 不算失败。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from bench_common import (BENCH_DIR, DEFAULT_GUARD, DEFAULT_LM, DEFAULT_MODEL,
                          DEFAULT_PAIRS, env_info, now_iso)
from schema_check import validate

SUITE_MODULES = {
    "yue-lengths": "suite_yue_lengths",
    "yue-guard": "suite_yue_guard",
    "zh-sentence": "suite_zh_sentence",
}


def build_envelope(suite: str, suite_version: str, params: dict,
                   status: str, results: dict | None, reason: str | None,
                   duration_s: float) -> dict:
    return {
        "suite": suite,
        "suite_version": suite_version,
        "spec": "projects/06-benchmark/schemas.md (envelope v1)",
        "status": status,
        "quick": params["quick"],
        "generated_at": now_iso(),
        "duration_s": round(duration_s, 2),
        "params": params,
        "environment": env_info(),
        "reason": reason,
        "results": results,
    }


def run_suite(name: str, params: dict) -> tuple[dict, str]:
    """跑单个 suite，返回 (envelope, 输出文件路径)。"""
    import importlib
    import time

    module = importlib.import_module(SUITE_MODULES[name])
    t0 = time.perf_counter()
    try:
        status, results, reason = module.run(params)
    except Exception as exc:                       # noqa: BLE001  套件崩溃不拖垮其余 suite
        status, results, reason = "error", None, f"{type(exc).__name__}: {exc}"
    env = build_envelope(name, module.SUITE_VERSION, params, status,
                         results, reason, time.perf_counter() - t0)
    errs = validate(env)
    if errs:
        env["status"] = "error"
        env["results"] = None
        env["reason"] = "输出未通过 schema 校验：" + "; ".join(errs)

    out_dir = params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    mode = "quick" if params["quick"] else "full"
    out_path = os.path.join(out_dir, f"{name}__{mode}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return env, out_path


def print_summary(env: dict, out_path: str) -> None:
    line = f"[{env['status']}] {env['suite']} v{env['suite_version']} ({env['duration_s']}s) -> {out_path}"
    if env.get("reason"):
        line += f"\n    原因：{env['reason']}"
    print(line)
    results = env.get("results") or {}
    for name, b in (results.get("buckets") or {}).items():
        if env["suite"] == "yue-lengths":
            nn = f"{b['neural']:>7.1%}" if b["neural"] is not None else "      —"
            print(f"  {name:<10} n={b['n']:<4} 照抄={b['copy_baseline']:.1%} "
                  f"规则={b['rule']:>7.1%} 神经={nn}")
        elif env["suite"] == "yue-guard":
            print(f"  {name:<10} n={b['n']:<4} 规则={b['rule']:.1%} 神经={b['neural']:.1%} "
                  f"+守卫={b['guarded']:.1%} 否决={b['rejected']} 救回={b['saved']} 弄丢={b['lost']}")
        elif env["suite"] == "zh-sentence":
            extra = (f" 救回={b['fixed']} 弄坏={b['broken']}"
                     if "fixed" in b else "")
            print(f"  {name:<8} 自洽题={b['n']:<4} Top1={b['top1']:.1%} "
                  f"前9={b['top9']:.1%} 平均键数={b['avg_keystrokes']:.1f}{extra}")
    agg = results.get("weighted") or results.get("aggregate")
    if env["suite"] == "yue-lengths" and isinstance(agg, dict) and agg.get("weighted_rule") is not None:
        nn = (f" → 神经 {agg['weighted_neural']:.1%}"
              if agg.get("weighted_neural") is not None else "")
        print(f"  加权（覆盖 {agg['weight_coverage']:.0%}）：规则 {agg['weighted_rule']:.1%}{nn}")
    if env["suite"] == "zh-sentence" and results.get("rerank"):
        rr = results["rerank"]
        print(f"  重排对照：不重排 {rr['top1_no_lm']:.1%} → 重排 {rr['top1_lm']:.1%}；"
              f"救回 {rr['saved']} / 弄坏 {rr['broken']} / 净 {rr['net']:+d}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all",
                    choices=["all", *SUITE_MODULES], help="跑哪个套件（默认 all）")
    ap.add_argument("--quick", action="store_true",
                    help="小样本模式：每桶 ≤50 题，分钟级；只用于冒烟，不用于排名")
    ap.add_argument("--out", default=os.path.join(BENCH_DIR, "results"),
                    help="JSON 输出目录（默认 projects/06-benchmark/results）")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="平行语料（默认 vendor/yue_pairs.json）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="粤语 seq2seq 权重")
    ap.add_argument("--guard", default=DEFAULT_GUARD, help="守卫评测数据（训练侧切分）")
    ap.add_argument("--lm", nargs="?", const=DEFAULT_LM, default="",
                    help="整句评测挂字级 LM 做重排对照（不带值则用 data/lm_char.marshal）")
    ap.add_argument("--lm-weight", type=float, default=0.4, help="重排插值 λ")
    ap.add_argument("--limit", type=int, default=0,
                    help="语料扫描句对数上限（默认 quick 20000 / full 150000）")
    ap.add_argument("--n", type=int, default=0,
                    help="zh-sentence 每桶题数上限（默认 quick 50 / full 300）")
    ap.add_argument("--ctx-max", type=int, default=7, help="守卫语境实证适用的最大输入长度")
    ap.add_argument("--show", type=int, default=5, help="yue-guard 每桶保留的决策样例数")
    ap.add_argument("--nn", default="auto", choices=["auto", "stub", "off"],
                    help="auto=有模型就用；stub=原样返回的桩（验证管线）；off=只评非神经列")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    suites = list(SUITE_MODULES) if args.suite == "all" else [args.suite]
    statuses = {}
    for name in suites:
        params = {
            "quick": args.quick,
            "out_dir": args.out,
            "pairs": args.pairs,
            "model": args.model,
            "guard": args.guard,
            "lm": args.lm,
            "lm_weight": args.lm_weight,
            "limit": args.limit or None,
            "n": args.n or None,
            "ctx_max": args.ctx_max,
            "show": args.show,
            "nn": args.nn,
            "device": args.device,
        }
        env, out_path = run_suite(name, params)
        statuses[name] = env["status"]
        print_summary(env, out_path)

    if any(s == "error" for s in statuses.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
