#!/usr/bin/env python3
"""给训练好的 seq2seq 打分：准确率 / 延迟 / 体积，并与现有规则版对比。"""
from __future__ import annotations

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

from train_yue_seq2seq import EOS, PAD, SOS, Seq2Seq, encode  # noqa: E402

from ipinyin.t2s import Converter  # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402


def load_test(pairs_path: str, limit: int):
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38][:limit]
    n_test = max(500, len(pairs) // 20)
    return pairs[:n_test]


@torch.no_grad()
def translate(model, text, src_vocab, inv_tgt, max_len=40):
    model.eval()
    ids = torch.tensor([encode(text, src_vocab)])
    memory = model.encoder(model.src_emb(ids) + model.pos(torch.arange(ids.size(1)).unsqueeze(0)),
                           src_key_padding_mask=ids == PAD)
    out = [SOS]
    for _ in range(max_len):
        tgt = torch.tensor([out])
        hidden = model.decoder(model.tgt_emb(tgt) + model.pos(torch.arange(len(out)).unsqueeze(0)),
                               memory, tgt_mask=nn.Transformer.generate_square_subsequent_mask(len(out)),
                               memory_key_padding_mask=ids == PAD)
        nxt = model.out(hidden)[0, -1].argmax().item()
        if nxt == EOS:
            break
        out.append(nxt)
    return "".join(inv_tgt.get(i, "") for i in out[1:])


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "yue_model.pt")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    model = Seq2Seq(len(src_vocab) + 4, len(tgt_vocab) + 4)
    model.load_state_dict(ckpt["model"])

    test = load_test("/tmp/yue_pairs_full.json", limit)
    rule = YueConverter.load_dict(os.path.join(ROOT, "data", "yue.marshal"),
                                  YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))._map)
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))

    sample = test[:300]
    nn_exact = rule_exact = 0
    nn_overlap = rule_overlap = 0.0
    for p in sample:
        got = translate(model, p["zh"], src_vocab, inv_tgt)
        nn_exact += got == p["yue"]
        nn_overlap += sum(1 for a, b in zip(got, p["yue"]) if a == b) / max(len(p["yue"]), 1)
        base = t2s.to_traditional(rule.convert(p["zh"]))
        rule_exact += base == p["yue"]
        rule_overlap += sum(1 for a, b in zip(base, p["yue"]) if a == b) / max(len(p["yue"]), 1)

    n = len(sample)
    print(f"测试样本 {n} 条（held-out）")
    print(f"  规则版   整句 {rule_exact/n:5.1%}  字符重合 {rule_overlap/n:5.1%}")
    print(f"  神经网络 整句 {nn_exact/n:5.1%}  字符重合 {nn_overlap/n:5.1%}")

    # 延迟
    lat = []
    for p in sample[:100]:
        t0 = time.perf_counter()
        translate(model, p["zh"], src_vocab, inv_tgt)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"  单句延迟 p50 {lat[len(lat)//2]:.0f}ms  p95 {lat[int(len(lat)*0.95)]:.0f}ms")

    # 量化后体积/延迟
    q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save({"model": q.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab},
               os.path.join(ROOT, "experiments", "yue_model_int8.pt"))
    size_mb = os.path.getsize(os.path.join(ROOT, "experiments", "yue_model_int8.pt")) / 1e6
    print(f"  int8 量化后体积 {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
