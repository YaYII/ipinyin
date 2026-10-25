#!/usr/bin/env python3
"""SIGHAN 风格的中文拼写纠正（CSC）指标。

约定：JSONL 每行一个句子。
  gold：{"sentence": <错句>, "errors": [{"char": 错字, "correct": 正字, "position": int}]}
  pred：{"sentence": <同一错句>, "predictions": [{"position": int, "correct": 建议字}]}（无误报时为空表）

指标（对齐全 SIGHAN CSC 评测的常用口径）：
  检测级 detection —— 位置找没找对：
    P = |预测位置 ∩ 金标位置| / |预测位置|
    R = |预测位置 ∩ 金标位置| / |金标位置|
  纠正级 correction —— 位置对且替换字对：
    TP = 预测中「位置在金标且 correct == 金标正字」的个数
    P = TP / |全部预测|，R = TP / |金标错误|
  分母为 0 时按 0 计（照抄基线因此得 0）。

  分类型桶（homophone / shape）：桶内 recall = 桶内 TP / 桶内金标错误；
  桶内 precision = 桶内 TP / (桶内 TP + 全局 FP)——预测没有类型标签，
  落在别类金标位置上的预测不算该桶的 TP 也不算该桶的 FP，
  所有真正误报（预测位置无错）计入每个桶的 precision 分母（常见 CSC 论文口径）。

用法：
  python3 eval.py gold.jsonl pred.jsonl [--json]
"""
from __future__ import annotations

import argparse
import json
import os


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f}


def compute_metrics(gold: list[dict], preds: list[dict]) -> dict:
    """输入金标记录与预测记录（按行对齐，sentence 必须一致），返回指标字典。"""
    if len(gold) != len(preds):
        raise ValueError(f"gold {len(gold)} 行 vs pred {len(preds)} 行，数量不一致")
    for i, (g, p) in enumerate(zip(gold, preds)):
        if g["sentence"] != p["sentence"]:
            raise ValueError(f"第 {i} 行 sentence 不对齐：\n  gold={g['sentence']!r}\n  pred={p['sentence']!r}")

    det_tp = det_fp = det_fn = 0
    corr_tp = 0
    n_preds = 0
    n_errors = 0
    n_clean = 0
    type_gold: dict[str, int] = {"homophone": 0, "shape": 0, "other": 0}
    type_tp: dict[str, int] = {"homophone": 0, "shape": 0, "other": 0}

    for g, p in zip(gold, preds):
        gold_map = {e["position"]: e for e in g.get("errors", [])}
        n_errors += len(gold_map)
        if not gold_map:
            n_clean += 1
        pred_map = {q["position"]: q for q in p.get("predictions", [])}
        n_preds += len(pred_map)

        for pos, e in gold_map.items():
            etype = e.get("type", "other")
            type_gold[etype] = type_gold.get(etype, 0) + 1
        for pos, q in pred_map.items():
            e = gold_map.get(pos)
            if e is None:
                continue                       # 误报，不入任何桶的 TP
            det_tp += 1
            if q.get("correct") == e["correct"]:
                corr_tp += 1
                etype = e.get("type", "other")
                type_tp[etype] = type_tp.get(etype, 0) + 1
        det_fp += len(set(pred_map) - set(gold_map))
        det_fn += len(set(gold_map) - set(pred_map))

    corr_fp = n_preds - corr_tp
    corr_fn = n_errors - corr_tp
    global_fp = det_fp
    by_type = {}
    for t in sorted(type_gold):
        tp = type_tp[t]
        p = tp / (tp + global_fp) if tp + global_fp else 0.0
        r = tp / type_gold[t] if type_gold[t] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        by_type[t] = {"gold": type_gold[t], "tp": tp,
                      "precision": p, "recall": r, "f1": f}

    return {
        "counts": {"sentences": len(gold), "clean_sentences": n_clean,
                   "errors": n_errors, "predictions": n_preds},
        "detection": _prf(det_tp, det_fp, det_fn),
        "correction": _prf(corr_tp, corr_fp, corr_fn),
        "by_type": by_type,
    }


def format_report(m: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    c = m["counts"]
    lines.append(f"句子 {c['sentences']}（干净 {c['clean_sentences']}）｜金标错误 {c['errors']}｜预测 {c['predictions']}")
    for key, label in (("detection", "检测级"), ("correction", "纠正级")):
        d = m[key]
        lines.append(f"{label}: P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f}"
                     f"  (tp={d['tp']} fp={d['fp']} fn={d['fn']})")
    for t, d in m["by_type"].items():
        lines.append(f"  [{t}] 金标 {d['gold']}  P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--json", action="store_true", help="额外输出机器可读 JSON")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    m = compute_metrics(gold, preds)
    print(format_report(m, title=f"gold={args.gold}  pred={args.pred}"))
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
