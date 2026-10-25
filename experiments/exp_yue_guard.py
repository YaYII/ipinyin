#!/usr/bin/env python3
"""P1 约束解码（yue_guard）的分桶评测：守卫能不能保住两头。

对比四个口径：
  规则        纯规则转换（成语/古诗安全，口语词漏转）
  神经        seq2seq 裸输出（口语词好，成语/古诗改坏）
  神经+守卫   守卫否掉无实证改动 → 回退规则（P1 的目标形态）

另加 8-15 字长句桶：守卫不能把神经在长句上的 +28pt 弄丢。

用法：
  python3 experiments/exp_yue_guard.py --model experiments/yue_model_d256.pt \
      --guard data/yue_guard.eval.marshal
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from eval_ime_lengths import load_pairs, build_word_test  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402
from ipinyin.yue_guard import YueGuard  # noqa: E402


def load_model(path: str, device: str):
    import torch
    from train_yue_seq2seq import EOS, PAD, SOS, Seq2Seq, encode

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sv, tv = ckpt["src_vocab"], ckpt["tgt_vocab"]
    inv = {i: c for c, i in tv.items()}
    trained = ckpt.get("args") or {}
    model = Seq2Seq(len(sv) + 4, len(tv) + 4,
                    d=int(trained.get("d", 160)), layers=int(trained.get("layers", 2)))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    @torch.no_grad()
    def nn_fn(text: str) -> str:
        ids = torch.tensor([encode(text, sv)], device=device)
        mem = model.encoder(model.src_emb(ids) + model.pos(
            torch.arange(ids.size(1), device=device).unsqueeze(0)),
            src_key_padding_mask=ids == PAD)
        out = [SOS]
        for _ in range(40):
            tgt = torch.tensor([out], device=device)
            h = model.decoder(model.tgt_emb(tgt) + model.pos(
                torch.arange(len(out), device=device).unsqueeze(0)), mem,
                tgt_mask=model._causal[:len(out), :len(out)],
                memory_key_padding_mask=ids == PAD)
            nxt = int(model.out(h)[0, -1].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
        return "".join(inv.get(i, "") for i in out[1:])

    return nn_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--model", default=os.path.join(ROOT, "experiments", "yue_model_d256.pt"))
    ap.add_argument("--guard", default=os.path.join(ROOT, "data", "yue_guard.eval.marshal"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", type=int, default=8, help="每桶展示的守卫决策样例数")
    ap.add_argument("--detail", default="", help="只看这个桶的全部修复样例")
    ap.add_argument("--ctx-max", type=int, default=7, help="语境实证适用的最大输入长度")
    args = ap.parse_args()

    test = load_pairs(args.pairs, 150000)
    conv = YueConverter.load_dict(os.path.join(ROOT, "data", "yue.marshal"),
                                  YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))._map)
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
    guard = YueGuard.load(args.guard)
    assert guard is not None, f"守卫数据不存在：{args.guard}"
    guard.ctx_max_len = args.ctx_max
    nn_fn = load_model(args.model, args.device)

    buckets = build_word_test(test)
    buckets["5-7字"] = [(p["zh"], p["yue"]) for p in test if 5 <= len(p["zh"]) <= 7][:300]
    buckets["8-15字"] = [(p["zh"], p["yue"]) for p in test if 8 <= len(p["zh"]) <= 15][:300]
    idioms = [ln.strip() for ln in open(os.path.join(ROOT, "tools", "idioms.txt"), encoding="utf-8")
              if ln.strip() and not ln.startswith("#")]
    poems = [ln.strip() for ln in open(os.path.join(ROOT, "tools", "poems.txt"), encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    buckets["4字成语"] = [(w, t2s.to_traditional(w)) for w in idioms]
    buckets["5-7字古诗"] = [(w, t2s.to_traditional(w)) for w in poems]

    print(f"{'桶':<10}{'n':>5}{'规则':>8}{'神经':>8}{'+守卫':>8}{'否决':>6}{'救回':>6}{'弄丢':>6}{'守卫延迟':>9}")
    for name in ("2字", "4字成语", "5-7字", "5-7字古诗", "8-15字"):
        items = buckets.get(name) or []
        if not items:
            continue
        r_ok = n_ok = g_ok = rejected = saved = lost = 0
        lat = []
        examples: list[str] = []
        for zh, ref in items:
            rule_out = conv.convert(zh)
            if t2s.to_traditional(rule_out) == ref:
                r_ok += 1
            nn_out = nn_fn(zh)
            if t2s.to_traditional(nn_out) == ref:
                n_ok += 1
            t0 = time.perf_counter()
            final = guard.check(zh, nn_out, repair=conv.convert)
            lat.append((time.perf_counter() - t0) * 1000)
            repaired = final != nn_out
            if t2s.to_traditional(final) == ref:
                g_ok += 1
            if repaired:
                rejected += 1
                nn_right = t2s.to_traditional(nn_out) == ref
                rule_right = t2s.to_traditional(rule_out) == ref
                if rule_right and not nn_right:
                    saved += 1
                    examples.append(f"  修复→救回 {zh!r} 神经={nn_out!r}")
                elif nn_right and not rule_right:
                    lost += 1
                    examples.append(f"  修复→弄丢 {zh!r} 神经对={nn_out!r} 修成={final!r} 期望={ref!r}")
                elif t2s.to_traditional(final) == ref:
                    examples.append(f"  修复→修对 {zh!r} 神经={nn_out!r} 规则={rule_out!r} → {final!r}")
                else:
                    examples.append(f"  修复(都错) {zh!r} 神经={nn_out!r} → {final!r} 期望={ref!r}")
        n = len(items)
        lat.sort()
        print(f"{name:<10}{n:>5}{r_ok / n:>8.1%}{n_ok / n:>8.1%}{g_ok / n:>8.1%}"
              f"{rejected:>6}{saved:>6}{lost:>6}{lat[len(lat) // 2]:>8.2f}ms")
        if args.detail and name != args.detail:
            continue
        for line in (examples if args.detail else examples[: args.show]):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
