#!/usr/bin/env python3
"""项目①基准评测：held-out 上量「粤语 → 普通话」的转换质量。

指标：
  整句一致率   输出与参考普通话逐字相同
  字符重合率   与参考按位置逐字比对（zip 截断到较短路长，与 experiments/eval_yue_seq2seq.py 口径一致）
  照抄不改     对照基线：直接把粤语输入当输出（粤语书面语与普通话共用大量字词，
               短句上它是强基线，模型必须证明自己赢了它）

分桶按粤语源句长：≤8 / 9-15 / 16+（字符数）。

用法：
  python3 projects/01-yue2zh/eval.py                       # 全部 held-out（约 5000 句）
  python3 projects/01-yue2zh/eval.py --n 1000              # 只抽前 1000 句
  python3 projects/01-yue2zh/eval.py --model other.pt --limit 60000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BUCKETS = [("≤8", lambda n: n <= 8), ("9-15", lambda n: 9 <= n <= 15), ("16+", lambda n: n >= 16)]


def bucket_of(n: int) -> str:
    for name, pred in BUCKETS:
        if pred(n):
            return name
    return "16+"


def char_overlap(got: str, ref: str) -> float:
    return sum(1 for a, b in zip(got, ref) if a == b) / max(len(ref), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="粤语→普通话 held-out 评测（整句一致率 / 字符重合率，分桶）")
    ap.add_argument("--model", default=os.path.join(HERE, "yue2zh_model.pt"))
    ap.add_argument("--pairs", default=None, help="语料 json；默认自动找 vendor/")
    ap.add_argument("--limit", type=int, default=100000,
                    help="必须与训练时 --limit 一致，否则切分出的 held-out 不同")
    ap.add_argument("--n", type=int, default=0, help="评测句数；0 = 全部 held-out")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto/cpu/mps")
    ap.add_argument("--samples", type=int, default=3, help="每桶展示的错例条数")
    args = ap.parse_args()

    import torch

    import convert as cv
    from train import default_pairs, load_and_split

    pairs_path = args.pairs or default_pairs()
    _, test = load_and_split(pairs_path, args.limit)
    if args.n > 0:
        test = test[: args.n]
    print(f"held-out {len(test)} 句（pairs {pairs_path}，limit {args.limit}）", flush=True)

    pack = cv.load_model(args.model, args.device)
    trained = pack["args"] or {}
    ep = pack.get("epoch", trained.get("epoch"))
    print(f"模型 {args.model}  d={trained.get('d')} layers={trained.get('layers')} "
          f"epoch={ep} device={pack['device']}", flush=True)

    t0 = time.time()
    got = cv.greedy_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"], pack["device"],
                           [p["yue"] for p in test], batch_size=args.batch)
    decode_s = time.time() - t0

    rows = []       # (bucket, nn_exact, copy_exact, nn_overlap, copy_overlap)
    samples: dict[str, list] = {name: [] for name, _ in BUCKETS}
    for p, g in zip(test, got):
        ref, src = p["zh"], p["yue"]
        b = bucket_of(len(src))
        rows.append((b, g == ref, src == ref, char_overlap(g, ref), char_overlap(src, ref)))
        if g != ref and len(samples[b]) < args.samples:
            samples[b].append((src, g, ref))

    print(f"\n{'桶':6s} {'n':>6s} | {'照抄整句':>8s} {'照抄字符':>8s} | {'神经整句':>8s} {'神经字符':>8s} | {'Δ整句':>7s}")
    order = [name for name, _ in BUCKETS] + ["全部"]
    for name in order:
        sel = rows if name == "全部" else [r for r in rows if r[0] == name]
        if not sel:
            continue
        n = len(sel)
        nn_e = sum(r[1] for r in sel) / n
        cp_e = sum(r[2] for r in sel) / n
        nn_o = sum(r[3] for r in sel) / n
        cp_o = sum(r[4] for r in sel) / n
        print(f"{name:6s} {n:6d} | {cp_e:7.1%}  {cp_o:7.1%}  | {nn_e:7.1%}  {nn_o:7.1%}  | {(nn_e - cp_e)*100:+6.1f}pt")

    print(f"\n贪心解码用时 {decode_s:.1f}s（{pack['device']}，batch {args.batch}），"
          f"单句 p50 {decode_s/len(test)*1000:.1f}ms")
    for name in order[:-1]:
        for src, g, ref in samples[name]:
            print(f"\n[{name}] 粤 {src}\n      神 {g}\n      参 {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
