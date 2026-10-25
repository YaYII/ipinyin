#!/usr/bin/env python3
"""两个纠错基线。

(a) copy —— 照抄不改：原句直出，零预测。它给编辑式模型立「靶子」：
    docs/nn-upgrade.md P4 的前提就是 77% 的字符本来就不该动，
    这个基线把「什么都不做」的 P/R/F1 显式钉在 0/0/0。

(b) stat —— 统计基线：对句中每个汉字，从混淆表取同音 + 形近候选，
    用字级 CharLM 给「整句替换后 vs 原句」打分，增益超过阈值才替换。
    CharLM 优先用 data_gen.py 只在 train 划分上现训的 data/lm_eval.marshal
    （避免评测句泄漏）；也可 --lm 指向仓库产品 LM data/lm_char.marshal
    做对比（注意：产品 LM 见过 yue_pairs 全量，评测句会偏高，见 README）。

用法：
  python3 baseline.py --mode copy  --input data/test.jsonl --output out_copy.jsonl
  python3 baseline.py --mode stat  --input data/dev.jsonl  --output out_dev.jsonl --tune
  python3 baseline.py --mode stat  --input data/test.jsonl --output out_test.jsonl --threshold 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lm import CharLM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFUSION = os.path.join(HERE, "data", "confusion.json")
EVAL_LM = os.path.join(HERE, "data", "lm_eval.marshal")
PRODUCT_LM = os.path.join(ROOT, "data", "lm_char.marshal")

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def resolve_lm(path: str | None) -> tuple[CharLM, str]:
    """LM 选择：显式指定 > 评测 LM（train 划分现训）> 产品 LM（有污染风险，告警）。"""
    candidates = [path, EVAL_LM, PRODUCT_LM]
    for cand in candidates:
        if cand and os.path.exists(cand):
            lm = CharLM.load(cand)
            if lm is not None:
                if cand == PRODUCT_LM:
                    print(f"[警告] 使用产品 LM {cand}：它由 yue_pairs 全量训练，"
                          "本基准的评测句来自同一语料，分数会偏高（README「数据来源」）。")
                else:
                    print(f"LM: {cand}（order={lm.order}, vocab={len(lm)}）")
                return lm, cand
    raise FileNotFoundError(
        "找不到可用的 CharLM。先跑 `python3 data_gen.py` 生成 data/lm_eval.marshal，"
        f"或确认 {PRODUCT_LM} 存在。")


def stat_correct(
    sentence: str,
    lm: CharLM,
    homophone: dict[str, list[str]],
    shape: dict[str, list[str]],
    threshold: float,
    topk: int,
) -> list[dict]:
    """逐位置扫描：候选 = 同音 topk + 形近；整句替换打分，增益过阈值才替换。

    返回 predictions：[{"position", "char"(输入里的错字), "correct"(建议字)}]。
    """
    current = list(sentence)
    preds: list[dict] = []
    for i, ch in enumerate(current):
        if not ("\u4e00" <= ch <= "\u9fff"):
            continue
        cands = [c for c in homophone.get(ch, [])[:topk] if c != ch]
        cands += [c for c in shape.get(ch, []) if c != ch and c not in cands]
        if not cands:
            continue
        base = lm.logp("".join(current))
        best_cand, best_gain = None, threshold
        for cand in cands:
            trial = "".join(current[:i]) + cand + "".join(current[i + 1:])
            gain = lm.logp(trial) - base
            if gain > best_gain:
                best_cand, best_gain = cand, gain
        if best_cand is not None:
            preds.append({"position": i, "char": ch, "correct": best_cand})
            current[i] = best_cand
    return preds


def run(
    records: list[dict],
    mode: str,
    lm: CharLM | None,
    homophone: dict[str, list[str]],
    shape: dict[str, list[str]],
    threshold: float,
    topk: int,
) -> list[dict]:
    out = []
    for rec in records:
        if mode == "stat":
            preds = stat_correct(rec["sentence"], lm, homophone, shape, threshold, topk)
        else:
            preds = []
        out.append({"sentence": rec["sentence"], "predictions": preds})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("copy", "stat"), default="stat")
    ap.add_argument("--input", required=True, help="gold JSONL（data_gen.py 产物）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--lm", default=None, help="CharLM marshal 路径（默认评测 LM，回退产品 LM）")
    ap.add_argument("--threshold", type=float, default=2.0, help="logp 增益阈值（stat 模式）")
    ap.add_argument("--topk", type=int, default=8, help="每字最多考察几个同音候选")
    ap.add_argument("--tune", action="store_true",
                    help="在输入集上网格搜阈值（按纠正级 F1），用最优阈值出最终预测")
    args = ap.parse_args()

    with open(CONFUSION, encoding="utf-8") as fh:
        conf = json.load(fh)
    homophone, shape = conf["homophone"], conf["shape"]

    sys.path.insert(0, HERE)
    import eval as eval_mod  # noqa: E402  同目录的 eval.py

    records = eval_mod.load_jsonl(args.input)
    lm = None
    if args.mode == "stat":
        lm, lm_path = resolve_lm(args.lm)

    threshold = args.threshold
    if args.tune:
        assert lm is not None
        print(f"在 {args.input} 上调阈值（共 {len(records)} 句）……")
        best = (-1.0, None, None)
        for th in GRID:
            preds = run(records, "stat", lm, homophone, shape, th, args.topk)
            m = eval_mod.compute_metrics(records, preds)
            f1 = m["correction"]["f1"]
            print(f"  threshold={th:4.1f}  检测F1={m['detection']['f1']:.4f}  纠正F1={f1:.4f}")
            if f1 > best[0]:
                best = (f1, th, preds)
        threshold = best[1]
        print(f"最优 threshold={threshold}（纠正 F1={best[0]:.4f}）")
        final_preds = best[2]
    else:
        final_preds = run(records, args.mode, lm, homophone, shape, threshold, args.topk)

    with open(args.output, "w", encoding="utf-8") as fh:
        for rec in final_preds:
            rec["meta"] = {"mode": args.mode, "threshold": threshold if args.mode == "stat" else None}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写出 {args.output}（{len(final_preds)} 句，mode={args.mode}, threshold={threshold}）")

    m = eval_mod.compute_metrics(records, final_preds)
    print(eval_mod.format_report(m, title=f"[{args.mode}] {args.input}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
