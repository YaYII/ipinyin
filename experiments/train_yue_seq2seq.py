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
    MAX_LEN = 64            # 位置表 / 因果掩码的最大长度

    def __init__(self, src_vocab, tgt_vocab, d=160, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(self.MAX_LEN, d)
        # 因果掩码预先注册成常量：原来在 forward 里现算（generate_square_subsequent_mask），
        # 带数据相关控制流，torch.onnx 的 dynamo 导出器直接失败；变成切片后能导出变长图，
        # 也省掉每步重建 40x40 mask 的开销。persistent=False 才不会污染老 checkpoint。
        # 用 bool 而不是 float：padding mask 也是 bool，类型一致时 torch 不会走
        # F._canonical_mask 的类型转换断言（那里带 .item()，导出必炸）。
        self.register_buffer("_causal", torch.triu(
            torch.ones(self.MAX_LEN, self.MAX_LEN, dtype=torch.bool), diagonal=1), persistent=False)
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
        causal = self._causal[:t, :t]
        hidden = self.decoder(self.tgt_emb(tgt_in) + pos_t, memory,
                              tgt_mask=causal,
                              tgt_key_padding_mask=tgt_in == PAD,
                              memory_key_padding_mask=src == PAD)
        return self.out(hidden)


def pad(batch, width):
    return [row + [PAD] * (width - len(row)) for row in batch]


def default_pairs() -> str:
    """语料路径：fetch_yue_pairs.py 默认写 vendor/，旧文档写 /tmp/，两个都认。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, "vendor", "yue_pairs.json"), "/tmp/yue_pairs_full.json"):
        if os.path.exists(cand):
            return cand
    return os.path.join(root, "vendor", "yue_pairs.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="语料 json；默认自动找 vendor/ 或 /tmp/")
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300, help="预热步数（Transformer 必备，否则前期易崩）")
    ap.add_argument("--out", default="experiments/yue_model.pt")
    ap.add_argument("--device", default="auto", help="auto/cpu/mps/cuda")
    ap.add_argument("--amp", action="store_true", help="混合精度（mps/cuda 加速，约 1.3~1.8 倍）")
    ap.add_argument("--resume", action="store_true", help="从 --out 续训（含优化器状态）")
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--short-boost", type=int, default=1,
                    help="≤4 字句对的过采样倍数。输入法真实输入以 2-4 字词为主（权重 0.65），"
                         "而语料是句子级（均长 10.8 字）——纯句子训练越久越偏向句子模式，"
                         "2 字词转换反而退化（实测 12 轮 90.6%%→84.5%%）。"
                         "短句过采样把训练分布拉回打字分布。")
    ap.add_argument("--augment-guard", default="",
                    help="守卫数据 marshal（train 切分）的片段替换对并入训练集。"
                         "约 6 万条词级平行对（不好→唔好 / 之前的→之前嘅），把词级信号"
                         "从语料的 0.3%% 拉到 ~1/3——治 2 字桶退化，比过采样 445 个短句根本。"
                         "注意：key 含跨界片段（之前的），2026-10-25 实测会教坏成语桶，"
                         "已被 --augment-words 的真词版取代，保留仅作对照。")
    ap.add_argument("--augment-words", default="",
                    help="真词级平行语料 JSON（experiments/build_word_pairs.py 产出，"
                         "锚定对齐 + lexicon 真词过滤）。词级信号干净版。")
    ap.add_argument("--words-repeat", type=int, default=1,
                    help="词级对的重复倍数（控制词级样本占比；4≈15%%，6≈24%%）。")
    args = ap.parse_args()
    if not args.pairs:
        args.pairs = default_pairs()
    if not os.path.exists(args.pairs):
        print(f"找不到语料 {args.pairs}\n先跑：python3 tools/fetch_yue_pairs.py", flush=True)
        return 2

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"          # Apple M 系列
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print("训练设备", args.device, flush=True)

    random.seed(0)
    torch.manual_seed(0)
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
    pairs = pairs[: args.limit]
    n_test = max(500, len(pairs) // 20)
    n_valid = n_test
    test, valid, train = pairs[:n_test], pairs[n_test:2 * n_test], pairs[2 * n_test:]
    if args.short_boost > 1:
        # 过采样只在 train 内做：test/valid 保持原分布，验证与基准不被污染
        short = [p for p in train if len(p["zh"]) <= 4]
        train = train + short * (args.short_boost - 1)
        random.shuffle(train)
    if args.augment_guard and os.path.exists(args.augment_guard):
        import marshal as _marshal
        import zlib as _zlib
        raw = open(args.augment_guard, "rb").read()
        try:
            raw = _zlib.decompress(raw)
        except _zlib.error:
            pass
        aug: list[dict] = []
        for z, ys in _marshal.loads(raw).get("word_pairs", {}).items():
            for y in ys:
                if z and y and len(z) <= 38 and len(y) <= 38:
                    aug.append({"zh": z, "yue": y})
        train = train + aug
        random.shuffle(train)
        print(f"守卫片段对增广 +{len(aug)}（词级训练信号，train 切分无泄漏）", flush=True)
    if args.augment_words and os.path.exists(args.augment_words):
        words_aug = [p for p in json.load(open(args.augment_words, encoding="utf-8"))
                     if p.get("zh") and p.get("yue")
                     and len(p["zh"]) <= 38 and len(p["yue"]) <= 38]
        if args.words_repeat > 1:
            words_aug = words_aug * args.words_repeat
        train = train + words_aug
        random.shuffle(train)
        print(f"真词对增广 +{len(words_aug)}（×{args.words_repeat}，"
              f"词级占比 {len(words_aug) / (len(train)):.0%}）", flush=True)
    print(f"训练 {len(train)} / 验证 {len(valid)} / 测试 {len(test)}", flush=True)

    src_vocab = build_vocab(train, "zh")
    tgt_vocab = build_vocab(train, "yue")
    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    print(f"字表 src {len(src_vocab)} tgt {len(tgt_vocab)}", flush=True)

    model = Seq2Seq(len(src_vocab) + 4, len(tgt_vocab) + 4, d=args.d, layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("opt"):
            try:
                opt.load_state_dict(ckpt["opt"])
            except ValueError:
                print("优化器状态不匹配，重新初始化", flush=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"已从 {args.out} 续训，接着第 {start_epoch} 轮", flush=True)
    model.to(device)
    print("参数", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    total_steps = max(1, (len(train) // args.batch) * args.epochs)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    def batches(data, size):
        """按长度分桶：同批长度接近，少一半以上的 PAD 计算。"""
        pool: list[dict] = []
        for i in range(0, len(data), size):
            pool.extend(data[i:i + size])
            if len(pool) >= size * 40:
                pool.sort(key=lambda p: len(p["zh"]) + len(p["yue"]))
                for j in range(0, len(pool), size):
                    yield pool[j:j + size]
                pool = []
        if pool:
            pool.sort(key=lambda p: len(p["zh"]) + len(p["yue"]))
            for j in range(0, len(pool), size):
                yield pool[j:j + size]

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        total = 0.0
        for step, chunk in enumerate(batches(train, args.batch)):
            src = [encode(p["zh"], src_vocab) for p in chunk]
            tgt = [encode(p["yue"], tgt_vocab) for p in chunk]
            sw = max(len(r) for r in src)
            tw = max(len(r) for r in tgt)
            src_t = torch.tensor(pad(src, sw), device=device)
            tgt_t = torch.tensor(pad(tgt, tw), device=device)
            for group in opt.param_groups:
                group["lr"] = lr_at(global_step)
            if args.amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type in ("cuda", "mps")):
                    logits = model(src_t, tgt_t[:, :-1])
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_t[:, 1:].reshape(-1))
            else:
                logits = model(src_t, tgt_t[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_t[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            global_step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train)//args.batch} loss {total/(step+1):.3f} "
                      f"lr {lr_at(global_step):.1e} {time.time()-started:.0f}s", flush=True)
        print(f"epoch {epoch} 结束 平均 loss {total/(step+1):.3f} 用时 {time.time()-started:.0f}s", flush=True)
        torch.save({"model": model.to("cpu").state_dict(), "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab, "args": vars(args), "epoch": epoch,
                    "opt": {k: v for k, v in opt.state_dict().items() if k != "param_groups"}},
                   args.out)
        model.to(device)
        print("已保存检查点", args.out, flush=True)

        @torch.no_grad()
        def translate(text):
            model.eval()
            ids = torch.tensor([encode(text, src_vocab)], device=device)
            memory = model.encoder(model.src_emb(ids) + model.pos(torch.arange(ids.size(1), device=device).unsqueeze(0)),
                                   src_key_padding_mask=ids == PAD)
            out = [SOS]
            for _ in range(40):
                tgt = torch.tensor([out], device=device)
                # 掩码必须与输入同设备，否则 MPS 上会 "device mismatch"
                causal = model._causal[:len(out), :len(out)]
                hidden = model.decoder(model.tgt_emb(tgt) + model.pos(torch.arange(len(out), device=device).unsqueeze(0)),
                                       memory, tgt_mask=causal,
                                       memory_key_padding_mask=ids == PAD)
                nxt = model.out(hidden)[0, -1].argmax().cpu().item()
                if nxt == EOS:
                    break
                out.append(nxt)
            return "".join(inv_tgt.get(i, "") for i in out[1:])

        exact = 0
        overlap = 0.0
        checked = len(valid[:400])
        for p in valid[:400]:
            got = translate(p["zh"])
            exact += got == p["yue"]
            overlap += sum(1 for a, b in zip(got, p["yue"]) if a == b) / max(len(p["yue"]), 1)
        print(f"[验证] 整句一致 {exact/checked:.1%} 字符重合 {overlap/checked:.1%}", flush=True)

    torch.save({"model": model.to("cpu").state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                "args": vars(args), "epoch": args.epochs - 1}, args.out)
    print("模型已保存", args.out, os.path.getsize(args.out) / 1e6, "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
