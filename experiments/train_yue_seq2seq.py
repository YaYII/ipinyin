#!/usr/bin/env python3
"""实验：字符级 Transformer 做「普通话 → 粤语」转换。

目标不是上线，而是量出三个数字：整句一致率、单句延迟、模型体积。
用法：python3 experiments/train_yue_seq2seq.py --pairs /tmp/yue_pairs_full.json \
        --limit 60000 --epochs 2 --out experiments/yue_model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

PAD, SOS, EOS, UNK = 0, 1, 2, 3


def build_vocab(pairs, side, limit=6000):
    freq: dict[str, int] = {}
    for p in pairs:
        for ch in p[side]:
            freq[ch] = freq.get(ch, 0) + 1
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:limit]]
    return {c: i + 4 for i, c in enumerate(chars)}


def encode(text, vocab, max_len=40):
    ids = [vocab.get(ch, UNK) for ch in text][:max_len - 2]
    return [SOS] + ids + [EOS]


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=160, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(64, d)
        encoder_layer = nn.TransformerEncoderLayer(d, heads, 512, dropout, batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d, heads, 512, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.out = nn.Linear(d, tgt_vocab)

    def forward(self, src, tgt_in):
        device = src.device
        b, s = src.shape
        pos_s = self.pos(torch.arange(s, device=device)).unsqueeze(0)
        memory = self.encoder(self.src_emb(src) + pos_s, src_key_padding_mask=src == PAD)
        t = tgt_in.shape[1]
        pos_t = self.pos(torch.arange(t, device=device)).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=device)
        hidden = self.decoder(self.tgt_emb(tgt_in) + pos_t, memory,
                              tgt_mask=causal,
                              tgt_key_padding_mask=tgt_in == PAD,
                              memory_key_padding_mask=src == PAD)
        return self.out(hidden)


def pad(batch, width):
    return [row + [PAD] * (width - len(row)) for row in batch]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/tmp/yue_pairs_full.json")
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="experiments/yue_model.pt")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
    pairs = pairs[: args.limit]
    n_test = max(500, len(pairs) // 20)
    n_valid = n_test
    test, valid, train = pairs[:n_test], pairs[n_test:2 * n_test], pairs[2 * n_test:]
    print(f"训练 {len(train)} / 验证 {len(valid)} / 测试 {len(test)}", flush=True)

    src_vocab = build_vocab(train, "zh")
    tgt_vocab = build_vocab(train, "yue")
    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    print(f"字表 src {len(src_vocab)} tgt {len(tgt_vocab)}", flush=True)

    model = Seq2Seq(len(src_vocab) + 4, len(tgt_vocab) + 4)
    print("参数", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    def batches(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        total = 0.0
        for step, chunk in enumerate(batches(train, args.batch)):
            src = [encode(p["zh"], src_vocab) for p in chunk]
            tgt = [encode(p["yue"], tgt_vocab) for p in chunk]
            sw = max(len(r) for r in src)
            tw = max(len(r) for r in tgt)
            src_t = torch.tensor(pad(src, sw))
            tgt_t = torch.tensor(pad(tgt, tw))
            logits = model(src_t, tgt_t[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_t[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train)//args.batch} loss {total/(step+1):.3f} "
                      f"{time.time()-started:.0f}s", flush=True)
        print(f"epoch {epoch} 结束 平均 loss {total/(step+1):.3f} 用时 {time.time()-started:.0f}s", flush=True)

        @torch.no_grad()
        def translate(text):
            model.eval()
            ids = torch.tensor([encode(text, src_vocab)])
            memory = model.encoder(model.src_emb(ids) + model.pos(torch.arange(ids.size(1)).unsqueeze(0)),
                                   src_key_padding_mask=ids == PAD)
            out = [SOS]
            for _ in range(40):
                tgt = torch.tensor([out])
                hidden = model.decoder(model.tgt_emb(tgt) + model.pos(torch.arange(len(out)).unsqueeze(0)),
                                       memory, tgt_mask=nn.Transformer.generate_square_subsequent_mask(len(out)),
                                       memory_key_padding_mask=ids == PAD)
                nxt = model.out(hidden)[0, -1].argmax().item()
                if nxt == EOS:
                    break
                out.append(nxt)
            return "".join(inv_tgt.get(i, "") for i in out[1:])

        exact = 0
        overlap = 0.0
        for p in valid[:400]:
            got = translate(p["zh"])
            exact += got == p["yue"]
            overlap += sum(1 for a, b in zip(got, p["yue"]) if a == b) / max(len(p["yue"]), 1)
        print(f"[验证] 整句一致 {exact/400:.1%} 字符重合 {overlap/400:.1%}", flush=True)

    torch.save({"model": model.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                "args": vars(args)}, args.out)
    print("模型已保存", args.out, os.path.getsize(args.out) / 1e6, "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
