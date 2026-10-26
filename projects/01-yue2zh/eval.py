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
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

BUCKETS = [("≤8", lambda n: n <= 8), ("9-15", lambda n: 9 <= n <= 15), ("16+", lambda n: n >= 16)]


def bucket_of(n: int) -> str:
    for name, pred in BUCKETS:
        if pred(n):
            return name
    return "16+"


def char_overlap(got: str, ref: str) -> float:
    return sum(1 for a, b in zip(got, ref) if a == b) / max(len(ref), 1)


def chrf_score(hyps: list[str], refs: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """corpus 级 chrF（Popović 2015 / sacrebleu 默认口径的纯 Python 实现）。

    字符 n-gram（n=1..6）在整个 corpus 上聚合计数，先按阶算 F_beta 再对阶取平均；
    忽略空白（中文无词界，空白只是排版噪声）。与 sacrebleu 的差异：它把空白当字符计入。
    """
    assert len(hyps) == len(refs)
    agg = {n: [0, 0, 0] for n in range(1, max_n + 1)}   # [匹配数, hyp 总数, ref 总数]
    for hyp, ref in zip(hyps, refs):
        hyp = re.sub(r"\s+", "", hyp)
        ref = re.sub(r"\s+", "", ref)
        for n in range(1, max_n + 1):
            h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
            r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
            agg[n][0] += sum(min(c, r[g]) for g, c in h.items())
            agg[n][1] += sum(h.values())
            agg[n][2] += sum(r.values())
    fs = []
    for n in range(1, max_n + 1):
        m, ht, rt = agg[n]
        if ht == 0 and rt == 0:
            continue          # 双侧都没有 n 阶 n-gram（句长 < n），跳过该阶
        p = m / ht if ht else 0.0
        r = m / rt if rt else 0.0
        fs.append((1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0)
    return sum(fs) / len(fs) if fs else 0.0


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
    ap.add_argument("--typo-p", type=float, default=0.0,
                    help=">0 时对粤语输入注入同音错字再评测（抗噪卷；固定种子 4321）")
    ap.add_argument("--beam", type=int, default=1, help="束宽；1=贪心")
    ap.add_argument("--rerank-lambda", type=float, default=0.0,
                    help="词典覆盖度重排权重 λ（需 beam>1；0=纯 beam）")
    ap.add_argument("--tune-lambda", action="store_true",
                    help="在训练集采样 dev 上扫 λ 再评测（避免拿测试集调参）")
    args = ap.parse_args()

    import torch

    import convert as cv
    from train import default_pairs, load_and_split

    pairs_path = args.pairs or default_pairs()
    train_pairs, test = load_and_split(pairs_path, args.limit)
    if args.n > 0:
        test = test[: args.n]
    print(f"held-out {len(test)} 句（pairs {pairs_path}，limit {args.limit}）", flush=True)

    pack = cv.load_model(args.model, args.device)
    trained = pack["args"] or {}
    ep = pack.get("epoch", trained.get("epoch"))
    print(f"模型 {args.model}  d={trained.get('d')} layers={trained.get('layers')} "
          f"epoch={ep} device={pack['device']}", flush=True)

    if args.typo_p > 0:
        import random

        from typo import build_homophones, inject_typos, load_yue_chars
        chars = load_yue_chars(os.path.join(ROOT, "data", "yue.marshal"))
        homophones = build_homophones(chars, allowed=set(pack["src_vocab"]))
        rng = random.Random(4321)
        noisy = [inject_typos(p["yue"], homophones, rng, args.typo_p) for p in test]
        changed = sum(1 for a, b in zip(noisy, (p["yue"] for p in test)) if a != b)
        test = [{"zh": p["zh"], "yue": y} for p, y in zip(test, noisy)]
        print(f"抗噪卷：注入同音错字 p={args.typo_p}（词表内可替换 {len(homophones)} 字），"
              f"{changed}/{len(test)} 句被改动", flush=True)

    t0 = time.time()
    mode = f"beam={args.beam}"
    if args.beam > 1:
        cands = cv.beam_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"], pack["device"],
                               [p["yue"] for p in test], beam=args.beam, batch_size=32)
        if args.rerank_lambda > 0 or args.tune_lambda:
            import json

            from dict_fuse import load_default_reverse_dict, rerank
            rdict = load_default_reverse_dict(corpus_pairs=json.load(
                open(pairs_path, encoding="utf-8")))
            lam = args.rerank_lambda
            if args.tune_lambda:
                import random

                from dict_fuse import rerank as _rr
                dev = random.Random(7).sample(train_pairs, min(2000, len(train_pairs)))
                dev_cands = cv.beam_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"],
                                           pack["device"], [p["yue"] for p in dev],
                                           beam=args.beam, batch_size=32)
                best_lam, best_acc = 0.0, -1.0
                for cand_lam in (0.0, 0.25, 0.5, 1.0, 2.0):
                    acc = sum(_rr(cs, p["yue"], rdict, cand_lam)[0][1] == p["zh"]
                              for cs, p in zip(dev_cands, dev)) / len(dev)
                    print(f"  [调参 dev n={len(dev)}] λ={cand_lam:<4} 整句 {acc:.1%}", flush=True)
                    if acc > best_acc:
                        best_acc, best_lam = acc, cand_lam
                lam = best_lam
                print(f"  选定 λ={lam}（dev 整句 {best_acc:.1%}）", flush=True)
            mode += f"+重排λ={lam:g}"
            got = [rerank(cs, p["yue"], rdict, lam)[0][1] for cs, p in zip(cands, test)]
        else:
            got = [cs[0][1] for cs in cands]
    else:
        got = cv.greedy_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"], pack["device"],
                               [p["yue"] for p in test], batch_size=args.batch)
    decode_s = time.time() - t0

    rows = []       # (bucket, src, got, ref)；指标在打印时按桶现算（chrF 要聚 n-gram）
    samples: dict[str, list] = {name: [] for name, _ in BUCKETS}
    for p, g in zip(test, got):
        ref, src = p["zh"], p["yue"]
        b = bucket_of(len(src))
        rows.append((b, src, g, ref))
        if g != ref and len(samples[b]) < args.samples:
            samples[b].append((src, g, ref))

    print(f"\n{'桶':6s} {'n':>6s} | {'照抄整句':>8s} {'照抄字符':>8s} {'照抄chrF':>8s}"
          f" | {'神经整句':>8s} {'神经字符':>8s} {'神经chrF':>8s} | {'Δ整句':>7s}")
    order = [name for name, _ in BUCKETS] + ["全部"]
    for name in order:
        sel = rows if name == "全部" else [r for r in rows if r[0] == name]
        if not sel:
            continue
        n = len(sel)
        srcs = [r[1] for r in sel]
        hyps = [r[2] for r in sel]
        refs = [r[3] for r in sel]
        nn_e = sum(h == r for h, r in zip(hyps, refs)) / n
        cp_e = sum(s == r for s, r in zip(srcs, refs)) / n
        nn_o = sum(char_overlap(h, r) for h, r in zip(hyps, refs)) / n
        cp_o = sum(char_overlap(s, r) for s, r in zip(srcs, refs)) / n
        nn_f = chrf_score(hyps, refs)
        cp_f = chrf_score(srcs, refs)
        print(f"{name:6s} {n:6d} | {cp_e:8.1%} {cp_o:8.1%} {cp_f:8.1%}"
              f" | {nn_e:8.1%} {nn_o:8.1%} {nn_f:8.1%} | {(nn_e - cp_e)*100:+6.1f}pt")

    print(f"\n解码模式 {mode}；用时 {decode_s:.1f}s（{pack['device']}），"
          f"单句 p50 {decode_s/len(test)*1000:.1f}ms")
    for name in order[:-1]:
        for src, g, ref in samples[name]:
            print(f"\n[{name}] 粤 {src}\n      神 {g}\n      参 {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
