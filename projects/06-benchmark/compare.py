#!/usr/bin/env python3
"""两系统配对显著性检验（McNemar 精确检验）。

输入两份结果 JSON（同 suite、同模式、同 question_set_hash——即同题同序），
按 per_item 逐题配对，报每桶与全量的：
  b = A 对而 B 错的题数；c = B 对而 A 错的题数；
  McNemar 精确 p 值 = 2·P(X ≤ min(b,c))，X~Bin(b+c, 0.5)，封顶 1。

判读纪律（与 METHODOLOGY §6/§11 一致）：p ≥ 0.05 → 「在噪声里」；
样本量不够时（quick 每桶 ≤50）即使 p < 0.05 也只配「方向性信号」，
对外宣称可分辨需要全量 + p < 0.05。

用法：
  python3 compare.py results/yue-lengths__quick.json results/yue-lengths__quick__mymodel.json
  python3 compare.py a.json b.json --keys rule:neural --out /tmp/cmp.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from bench_common import BENCH_DIR, r4

sys.path.insert(0, BENCH_DIR)

# 各 suite 默认对比的两个 per_item 布尔列（A vs B）
DEFAULT_KEYS = {
    "yue-lengths": ("rule_ok", "neural_ok"),
    "yue-guard": ("nn_ok", "guarded_ok"),
    "zh-sentence": ("base_top1_ok", "top1_ok"),
}


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 p 值。b=c=0 时无差异可言，返回 1.0。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compare(doc_a: dict, doc_b: dict, keys: tuple[str, str] | None = None) -> dict:
    if doc_a.get("suite") != doc_b.get("suite"):
        raise SystemExit(f"suite 不同：{doc_a.get('suite')} vs {doc_b.get('suite')}")
    if doc_a.get("quick") != doc_b.get("quick"):
        raise SystemExit("quick 模式不同，题面不可比（quick/full 题数不同）")
    ha, hb = doc_a.get("results", {}).get("question_set_hash"), \
        doc_b.get("results", {}).get("question_set_hash")
    if not ha or not hb:
        raise SystemExit("结果缺少 question_set_hash（v1.0.0 旧文件），无法配对；"
                         "用当前版本重跑两份再比")
    if ha != hb:
        raise SystemExit(f"question_set_hash 不同（{ha} vs {hb}）——不是同一套题")
    suite = doc_a["suite"]
    ka, kb = keys or DEFAULT_KEYS[suite]
    items_a = {it["id"]: it for it in doc_a["results"].get("per_item") or []}
    items_b = {it["id"]: it for it in doc_b["results"].get("per_item") or []}
    if not items_a or not items_b:
        raise SystemExit("结果缺少 per_item（v1.0.0 旧文件），无法配对")

    buckets: dict[str, dict] = {}
    overall = {"n": 0, "a_right": 0, "b_right": 0, "a_only": 0, "b_only": 0}
    for pid, it_a in items_a.items():
        it_b = items_b.get(pid)
        if it_b is None:
            raise SystemExit(f"题 {pid} 只在 A 里出现——两份结果的题面不一致")
        if ka not in it_a or kb not in it_b:
            raise SystemExit(f"per_item 缺列：A 需 {ka}，B 需 {kb}"
                             f"（{suite} 默认对比 {DEFAULT_KEYS[suite]}）")
        if it_a[ka] is None or it_b[kb] is None:
            raise SystemExit(
                f"列含 null（如 zh-sentence 未挂 --lm 时 base_top1_ok 全为 null），"
                f"该对比无意义——换列（--keys）或用带 --lm 的结果")
        a_ok, b_ok = bool(it_a[ka]), bool(it_b[kb])
        bucket = buckets.setdefault(it_a["bucket"], {"n": 0, "a_right": 0,
                                                     "b_right": 0,
                                                     "a_only": 0, "b_only": 0})
        bucket["n"] += 1
        overall["n"] += 1
        if a_ok:
            bucket["a_right"] += 1
        if b_ok:
            bucket["b_right"] += 1
        if a_ok and not b_ok:
            bucket["a_only"] += 1
            overall["a_only"] += 1
        if b_ok and not a_ok:
            bucket["b_only"] += 1
            overall["b_only"] += 1

    out_rows: dict[str, dict] = {}
    for name, b in list(buckets.items()) + [("全体", overall)]:
        p = mcnemar_exact(b["a_only"], b["b_only"])
        delta = (b["b_right"] - b["a_right"]) / b["n"] if b["n"] else 0.0
        out_rows[name] = {
            "n": b["n"],
            "a_right": b["a_right"], "b_right": b["b_right"],
            "a_acc": r4(b["a_right"] / b["n"]) if b["n"] else None,
            "b_acc": r4(b["b_right"] / b["n"]) if b["n"] else None,
            "delta_pp": r4(delta * 100),
            "a_only": b["a_only"], "b_only": b["b_only"],
            "mcnemar_p": r4(p),
            "verdict": ("可分辨（p<0.05）" if p < 0.05 else "在噪声里"),
        }
    return {
        "suite": suite,
        "quick": doc_a["quick"],
        "question_set_hash": ha,
        "a": {"system": doc_a.get("system") or "?", "key": ka,
              "file": doc_a.get("_file")},
        "b": {"system": doc_b.get("system") or "?", "key": kb,
              "file": doc_b.get("_file")},
        "rows": out_rows,
        "caveat": ("quick 模式样本量不足，p<0.05 也只算方向性信号；"
                   "对外宣称可分辨需全量口径 + p<0.05（METHODOLOGY §6/§11）")
        if doc_a["quick"] else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_a")
    ap.add_argument("result_b")
    ap.add_argument("--keys", default="",
                    help="覆盖默认对比列，如 rule:neural（A列:B列，per_item 布尔字段）")
    ap.add_argument("--out", default="", help="同时写 JSON 到此路径")
    args = ap.parse_args(argv)

    a = json.load(open(args.result_a, encoding="utf-8"))
    b = json.load(open(args.result_b, encoding="utf-8"))
    a["_file"], b["_file"] = args.result_a, args.result_b
    keys = tuple(args.keys.split(":")) if args.keys else None
    rep = compare(a, b, keys)

    name_a = rep["a"]["system"]
    name_b = rep["b"]["system"]
    print(f"配对比较：{name_a}（{rep['a']['key']}） vs {name_b}（{rep['b']['key']}）"
          f"  [{rep['suite']} quick={rep['quick']} hash={rep['question_set_hash']}]\n")
    header = f"  {'桶':<10}{'n':>5} {'A对':>5} {'B对':>5} {'Δpp':>7} {'A独B错':>7} {'B独A错':>7} {'p':>7}  判定"
    print(header)
    for name, row in rep["rows"].items():
        print(f"  {name:<10}{row['n']:>5} {row['a_right']:>5} {row['b_right']:>5} "
              f"{row['delta_pp']:>+7.1f} {row['a_only']:>7} {row['b_only']:>7} "
              f"{row['mcnemar_p']:>7.4f}  {row['verdict']}")
    if rep["caveat"]:
        print(f"\n注意：{rep['caveat']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"\n已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
