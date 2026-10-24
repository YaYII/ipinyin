#!/usr/bin/env python3
"""给训练好的 seq2seq 打分：与规则版逐段对比（含生产真正会走神经网络的长句桶）。

关键点：输入法里 `YueConverter.convert()` 只对 len(text) >= 8 调神经网络，
所以整体分数会骗人 —— 必须按长度分桶看。

用法：
  python3 experiments/eval_yue_seq2seq.py experiments/yue_model.pt [--pairs P] [--beam 1] [--n 400]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from train_yue_seq2seq import EOS, PAD, SOS, Seq2Seq, encode, default_pairs  # noqa: E402

from ipinyin.t2s import Converter  # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402

NN_MIN_LEN = 8          # 与 src/ipinyin/yue.py 保持一致


def load_test(pairs_path: str, limit: int):
    """与训练脚本同种子、同切分逻辑，保证是 held-out。"""
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38][:limit]
    n_test = max(500, len(pairs) // 20)
    return pairs[:n_test]


@torch.no_grad()
def translate(model, text, src_vocab, inv_tgt, max_len=40, beam=1, device="cpu"):
    model.eval()
    ids = torch.tensor([encode(text, src_vocab)], device=device)
    memory = model.encoder(
        model.src_emb(ids) + model.pos(torch.arange(ids.size(1), device=device).unsqueeze(0)),
        src_key_padding_mask=ids == PAD)

    if beam <= 1:
        out = [SOS]
        for _ in range(max_len):
            tgt = torch.tensor([out], device=device)
            hidden = model.decoder(
                model.tgt_emb(tgt) + model.pos(torch.arange(len(out), device=device).unsqueeze(0)),
                memory,
                tgt_mask=model._causal[:len(out), :len(out)],
                memory_key_padding_mask=ids == PAD)
            nxt = model.out(hidden)[0, -1].argmax().cpu().item()
            if nxt == EOS:
                break
            out.append(nxt)
        return "".join(inv_tgt.get(i, "") for i in out[1:])

    # beam search：每步把 B 条假设拼成一个 batch 前向，取全局 top-B
    beams: list[tuple[float, list[int]]] = [(0.0, [SOS])]
    done: list[tuple[float, list[int]]] = []
    for _ in range(max_len):
        live = [(s, h) for s, h in beams if h[-1] != EOS]
        if not live:
            break
        widths = [len(h) for _, h in live]
        width = max(widths)
        tgt = torch.full((len(live), width), PAD, dtype=torch.long, device=device)
        for i, (_, h) in enumerate(live):
            tgt[i, :len(h)] = torch.tensor(h, device=device)
        hidden = model.decoder(
            model.tgt_emb(tgt) + model.pos(torch.arange(width, device=device).unsqueeze(0)),
            memory.expand(len(live), -1, -1),
            tgt_mask=model._causal[:width, :width],
            memory_key_padding_mask=ids.expand(len(live), -1) == PAD)
        logp = torch.log_softmax(model.out(hidden)[:, -1].float(), dim=-1)
        cand: list[tuple[float, list[int]]] = []
        for i, (score, hyp) in enumerate(live):
            top = torch.topk(logp[i], beam).indices.cpu().tolist()
            for tok in top:
                cand.append((score + float(logp[i, tok]), hyp + [tok]))
        cand.sort(key=lambda kv: kv[0], reverse=True)
        beams = []
        for score, hyp in cand[:beam]:
            if hyp[-1] == EOS:
                done.append((score / max(len(hyp) - 1, 1), hyp))   # 长度归一，否则偏爱短句
            else:
                beams.append((score, hyp))
        if not beams:
            break
    if not done:
        done = [(s / max(len(h) - 1, 1), h) for s, h in beams]
    done.sort(key=lambda kv: kv[0], reverse=True)
    best = done[0][1]
    return "".join(inv_tgt.get(i, "") for i in best[1:] if i != EOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(ROOT, "experiments", "yue_model.pt"))
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--limit", type=int, default=60000, help="必须与训练时 --limit 一致，否则切分不同")
    ap.add_argument("--n", type=int, default=400, help="评测句数")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    trained = ckpt.get("args") or {}
    model = Seq2Seq(len(src_vocab) + 4, len(tgt_vocab) + 4,
                    d=int(trained.get("d", 160)), layers=int(trained.get("layers", 2)))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)

    test = load_test(args.pairs or default_pairs(), args.limit)[: args.n]
    rule = YueConverter.load_dict(os.path.join(ROOT, "data", "yue.marshal"),
                                  YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))._map)
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))

    buckets = {"短句(<8)": [], "长句(>=8)": []}
    lat = []
    samples = []
    for p in test:
        t0 = time.perf_counter()
        got = translate(model, p["zh"], src_vocab, inv_tgt, beam=args.beam, device=args.device)
        lat.append((time.perf_counter() - t0) * 1000)
        base = t2s.to_traditional(rule.convert(p["zh"]))
        ref = p["yue"]
        key = "长句(>=8)" if len(p["zh"]) >= NN_MIN_LEN else "短句(<8)"
        buckets[key].append((got == ref, base == ref,
                              sum(1 for a, b in zip(got, ref) if a == b) / max(len(ref), 1),
                              sum(1 for a, b in zip(base, ref) if a == b) / max(len(ref), 1)))
        if len(samples) < 8 and len(p["zh"]) >= NN_MIN_LEN and got != ref:
            samples.append((p["zh"], base, got, ref))

    print(f"测试集 {len(test)} 条（held-out，来自 --limit {args.limit}）  beam={args.beam}")
    for name, rows in buckets.items():
        if not rows:
            continue
        n = len(rows)
        nn_e = sum(r[0] for r in rows) / n
        ru_e = sum(r[1] for r in rows) / n
        nn_o = sum(r[2] for r in rows) / n
        ru_o = sum(r[3] for r in rows) / n
        delta = (nn_e - ru_e) * 100
        print(f"  {name:10s} n={n:4d}  规则 整句 {ru_e:5.1%} 字符 {ru_o:5.1%}"
              f"   |  神经 整句 {nn_e:5.1%} 字符 {nn_o:5.1%}   Δ整句 {delta:+.1f}pt")
    allrows = buckets["短句(<8)"] + buckets["长句(>=8)"]
    n = len(allrows)
    print(f"  {'全部':10s} n={n:4d}  规则 整句 {sum(r[1] for r in allrows)/n:5.1%}"
          f"   |  神经 整句 {sum(r[0] for r in allrows)/n:5.1%}")
    lat.sort()
    print(f"  单句延迟 p50 {lat[len(lat)//2]:.0f}ms  p95 {lat[int(len(lat)*0.95)]:.0f}ms"
          f"  （{args.device}，无 KV cache）")

    if samples:
        print("\n长句错例（普 / 规则 / 神经 / 参考）：")
        for zh, base, got, ref in samples:
            print(f"  普 {zh}\n  规 {base}\n  神 {got}\n  参 {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
