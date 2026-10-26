#!/usr/bin/env python3
"""项目①：粤语 → 普通话 反向 seq2seq 基准，训练驱动。

复用 experiments/train_yue_seq2seq.py 的模型与词表工具，只交换方向：
src = 粤语（繁体、含粤字），tgt = 普通话（简体）。
切分沿用仓库惯例：random.seed(0) 洗牌 → 双边 ≤38 字过滤 → 取前 --limit →
前 max(500, 1/20) 为 held-out 测试，其余全部做训练（本项目不设验证集）。

用法：
  python3 projects/01-yue2zh/train.py --limit 100000 --epochs 3 --device auto
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from experiments.train_yue_seq2seq import PAD, Seq2Seq, build_vocab, default_pairs, encode, pad
from convert import greedy_decode


def load_and_split(pairs_path: str, limit: int):
    """与 experiments/train_yue_seq2seq.py 同种子同顺序洗牌，held-out 可复现。

    返回 (train, test)；test 是前 1/20（至少 500 条），只评它。
    """
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38][:limit]
    n_test = max(500, len(pairs) // 20)
    return pairs[n_test:], pairs[:n_test]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="语料 json；默认自动找 vendor/")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "yue2zh_model.pt"))
    ap.add_argument("--device", default="auto", help="auto/cpu/mps/cuda")
    ap.add_argument("--amp", action="store_true", help="fp16 混合精度（MPS 加速，出 NaN 就关掉重跑）")
    ap.add_argument("--resume", action="store_true", help="从 --out 续训（含优化器状态）")
    ap.add_argument("--valid-n", type=int, default=200, help="每轮结束在 held-out 上抽检的句数")
    ap.add_argument("--typo", type=float, default=0.0,
                    help="训练时对粤语侧注入同音错字的字符级概率（0=关；粤拼表 data/yue.marshal）")
    args = ap.parse_args()
    if not args.pairs:
        args.pairs = default_pairs()
    if not os.path.exists(args.pairs):
        print(f"找不到语料 {args.pairs}", flush=True)
        return 2

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print("训练设备", args.device, flush=True)

    torch.manual_seed(0)
    train, test = load_and_split(args.pairs, args.limit)
    print(f"训练 {len(train)} / held-out {len(test)}（语料 {args.pairs}，limit {args.limit}）", flush=True)

    # 方向交换：粤语进，普通话出
    src_vocab = build_vocab(train, "yue")
    tgt_vocab = build_vocab(train, "zh")
    print(f"字表 src(粤) {len(src_vocab)} tgt(普) {len(tgt_vocab)}", flush=True)

    homophones = None
    if args.typo > 0:
        from typo import build_homophones, inject_typos, load_yue_chars
        chars = load_yue_chars(os.path.join(ROOT, "data", "yue.marshal"))
        homophones = build_homophones(chars, allowed=set(src_vocab))
        rng_typo = random.Random(1234)
        print(f"同音错字增强 p={args.typo}：粤拼表 {len(chars)} 字，词表内可替换 {len(homophones)} 字",
              flush=True)

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
    started_all = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        total = 0.0
        for step, chunk in enumerate(batches(train, args.batch)):
            if homophones is not None:
                src = [encode(inject_typos(p["yue"], homophones, rng_typo, args.typo), src_vocab)
                       for p in chunk]
            else:
                src = [encode(p["yue"], src_vocab) for p in chunk]
            tgt = [encode(p["zh"], tgt_vocab) for p in chunk]
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
        print(f"epoch {epoch} 结束 平均 loss {total/(step+1):.3f} 用时 {time.time()-started:.0f}s"
              f" 累计 {time.time()-started_all:.0f}s", flush=True)

        torch.save({"model": model.to("cpu").state_dict(), "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab, "args": vars(args), "epoch": epoch,
                    "opt": {k: v for k, v in opt.state_dict().items() if k != "param_groups"}},
                   args.out)
        model.to(device)
        print("已保存检查点", args.out, flush=True)

        # held-out 抽检（不是正式分数，正式分数看 eval.py）
        model.eval()
        probe = test[: args.valid_n]
        got = greedy_decode(model, src_vocab, tgt_vocab, device, [p["yue"] for p in probe])
        exact = sum(g == p["zh"] for g, p in zip(got, probe)) / len(probe)
        overlap = sum(sum(1 for a, b in zip(g, p["zh"]) if a == b) / max(len(p["zh"]), 1)
                      for g, p in zip(got, probe)) / len(probe)
        copy_exact = sum(p["yue"] == p["zh"] for p in probe) / len(probe)
        copy_overlap = sum(sum(1 for a, b in zip(p["yue"], p["zh"]) if a == b) / max(len(p["zh"]), 1)
                           for p in probe) / len(probe)
        print(f"[held-out 抽检 n={len(probe)}] 神经 整句 {exact:.1%} 字符 {overlap:.1%}"
              f" | 照抄 整句 {copy_exact:.1%} 字符 {copy_overlap:.1%}", flush=True)
        for p, g in list(zip(probe, got))[:3]:
            print(f"  粤 {p['yue']}\n  神 {g}\n  参 {p['zh']}", flush=True)

    torch.save({"model": model.to("cpu").state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                "args": vars(args), "epoch": args.epochs - 1}, args.out)
    print("训练完成", args.out, os.path.getsize(args.out) / 1e6, "MB", f"总用时 {time.time()-started_all:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
