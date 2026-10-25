#!/usr/bin/env python3
"""「新增覆盖考卷」：只用 pycorrector 合并进来的**新**混淆对造错，量扩表的真实价值。

为什么需要它：合成考卷若用原混淆表生成，所有金标纠正在旧表里本来就可达，
合并表在那种考卷上分数必然纹丝不动（实测 0.637 → 0.637）——不是扩表没用，
是考卷量不到。本脚本只从「合并表有、原表没有」的字对里造错：
  - 旧表跑这份考卷：候选里根本没有正确字，召回应接近 0；
  - 合并表跑：召回应显著起来。差值就是这次合并买到的东西。

用法：python3 eval_new_coverage.py [--n 600]
输出：data/test_newonly.jsonl + 新旧两表的检测/纠正指标对比
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_gen import _pick, _split_of, clean_sentences  # noqa: E402
from ipinyin.lm import CharLM  # noqa: E402

DATA = os.path.join(HERE, "data")


def new_only_tables(orig: dict, merged: dict) -> tuple[dict, dict]:
    """抽出「合并表新增」的 (字 → 新候选) 映射，按同音/形近分列。"""
    homo_new: dict[str, list[str]] = {}
    for ch, cands in merged["homophone"].items():
        old = set(orig["homophone"].get(ch, []))
        fresh = [c for c in cands if c not in old and c != ch]
        if fresh:
            homo_new[ch] = fresh
    shape_new: dict[str, list[str]] = {}
    for ch, cands in merged["shape"].items():
        old = set(orig["shape"].get(ch, []))
        fresh = [c for c in cands if c not in old and c != ch]
        if fresh:
            shape_new[ch] = fresh
    return homo_new, shape_new


def build_paper(n: int, homo_new: dict, shape_new: dict) -> list[dict]:
    """从 test 划分的句子里，只用新混淆对注入错误。"""
    rng = random.Random(7)
    sents = [s for s in clean_sentences() if _split_of(s) == "test"]
    rng.shuffle(sents)
    paper: list[dict] = []
    for sent in sents:
        if len(paper) >= n:
            break
        chars = list(sent)
        cn = [i for i, ch in enumerate(chars)
              if "\u4e00" <= ch <= "\u9fff" and (ch in homo_new or ch in shape_new)]
        if not cn:
            continue
        rng.shuffle(cn)
        errors = []
        for i in cn[:2]:                 # 每句最多 2 个新对错误
            table = homo_new if (chars[i] in homo_new and rng.random() < 0.7) else shape_new
            if chars[i] not in table:
                table = homo_new if chars[i] in homo_new else shape_new
            if chars[i] not in table:
                continue
            wrong = _pick(rng, table[chars[i]])
            if not wrong or wrong == chars[i]:
                continue
            errors.append({"char": wrong, "correct": chars[i],
                           "position": i, "type": "shape" if table is shape_new else "homophone"})
            chars[i] = wrong
        if errors:
            paper.append({"sentence": "".join(chars), "errors": errors})
    return paper


def score(pred_records: list[dict], gold_records: list[dict]) -> dict:
    """对齐 eval.py 的口径算检测/纠正 P/R/F1（自包含精简版）。"""
    tp = fp = fn = 0
    ctp = cfn = 0
    for pred, gold in zip(pred_records, gold_records):
        pset = {(e["position"], e["char"]) for e in pred.get("predictions", [])}
        gset = {(e["position"], e["char"]) for e in gold["errors"]}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
        ctp += len(pset & gset)
        cfn += len(gset - pset)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"detect_P": round(prec, 4), "detect_R": round(rec, 4), "detect_F1": round(f1, 4),
            "correct_tp": ctp, "correct_fn": cfn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=4.0)
    args = ap.parse_args()

    orig = json.load(open(os.path.join(DATA, "confusion.json"), encoding="utf-8"))
    merged = json.load(open(os.path.join(DATA, "confusion_merged.json"), encoding="utf-8"))
    homo_new, shape_new = new_only_tables(orig, merged)
    print(f"新增覆盖：同音 {len(homo_new)} 字 / 形近 {len(shape_new)} 字")

    paper = build_paper(args.n, homo_new, shape_new)
    out = os.path.join(DATA, "test_newonly.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for rec in paper:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    gold_errors = sum(len(r["errors"]) for r in paper)
    print(f"考卷 {len(paper)} 句 / {gold_errors} 个金标错误 → {out}")

    lm = CharLM.load(os.path.join(DATA, "lm_eval.marshal"))
    from baseline import stat_correct  # noqa: E402

    for tag, conf in (("旧表", orig), ("合并表", merged)):
        homo, shape = conf["homophone"], conf["shape"]
        preds = []
        for rec in paper:
            got = stat_correct(rec["sentence"], lm, homo, shape, args.threshold, 8)
            preds.append({"predictions": got})
        print(f"[{tag}] {json.dumps(score(preds, paper), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
