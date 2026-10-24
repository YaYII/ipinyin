#!/usr/bin/env python3
"""按「真实打字长度」评测粤语转换：2 / 3 / 4 / 5-7 字。

为什么要单独一个脚本：
  翻译语料是句子级的（zh 平均 10.8 字），拿它当输入法考题会严重失真。
  实际打字 2-4 字是词和成语，最多 5-7 字是古诗/长惯用语，再长人就按空格切了。
  所以主指标必须按长度分桶，而且短桶要用**词级**测试集，不能拿长句截断。

词级测试集的造法（不泄漏）：
  1. 语料按 seed=0 切出 test（前 1/20），训练只用剩下的；
  2. 在 test 里用 build_yue_phrase 的统计对齐挖 2-4 字片段对；
  3. 剔除 live 短语表 data/yue_phrase.marshal 已有的 key —— 留下的是
     「规则表没覆盖的词」，正是神经网络该补的位置。

用法：
  python3 tools/eval_ime_lengths.py                          # 规则版
  python3 tools/eval_ime_lengths.py --model experiments/yue_model.pt
"""
from __future__ import annotations

import argparse
import json
import marshal
import os
import random
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from build_yue_phrase import mine_phrases  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402

# 真实打字长度分布（按「以词为单位敲键盘」的直觉估的，可调）
WEIGHTS = {"2字": 0.40, "3字": 0.20, "4字成语": 0.25, "5-7字": 0.15}


def load_marshal(path: str) -> dict:
    raw = open(path, "rb").read()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    return marshal.loads(raw)


def load_pairs(path: str, limit: int):
    pairs = json.load(open(path, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    pairs = [p for p in pairs if p["zh"] and p["yue"] and len(p["zh"]) <= 38 and len(p["yue"]) <= 38][:limit]
    n_test = max(500, len(pairs) // 20)
    return pairs[:n_test]


def build_word_test(test_pairs: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """词级测试集：按 2/3/4 字分桶，剔除规则表已有的 key。"""
    live = load_marshal(os.path.join(ROOT, "data", "yue_phrase.marshal"))
    mined = mine_phrases(test_pairs, min_count=3)
    buckets: dict[str, list[tuple[str, str]]] = {"2字": [], "3字": [], "4字": []}
    for zh, yue in mined.items():
        if zh in live:                 # 规则表能直接命中，考它没意义
            continue
        key = f"{len(zh)}字"
        if key in buckets:
            buckets[key].append((zh, yue))
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--model", default="", help="留空则只评规则版")
    ap.add_argument("--limit", type=int, default=150000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    test = load_pairs(args.pairs, args.limit)
    conv = YueConverter.load_dict(os.path.join(ROOT, "data", "yue.marshal"),
                                  YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))._map)
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))

    nn_fn = None
    if args.model:
        import torch
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from train_yue_seq2seq import EOS, PAD, SOS, Seq2Seq, encode

        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
        sv, tv = ckpt["src_vocab"], ckpt["tgt_vocab"]
        inv = {i: c for c, i in tv.items()}
        trained = ckpt.get("args") or {}
        model = Seq2Seq(len(sv) + 4, len(tv) + 4,
                        d=int(trained.get("d", 160)), layers=int(trained.get("layers", 2)))
        model.load_state_dict(ckpt["model"])
        model.to(args.device).eval()

        @torch.no_grad()
        def nn_fn(text: str) -> str:                      # noqa: F811
            ids = torch.tensor([encode(text, sv)], device=args.device)
            mem = model.encoder(model.src_emb(ids) + model.pos(
                torch.arange(ids.size(1), device=args.device).unsqueeze(0)),
                src_key_padding_mask=ids == PAD)
            out = [SOS]
            for _ in range(40):
                tgt = torch.tensor([out], device=args.device)
                h = model.decoder(model.tgt_emb(tgt) + model.pos(
                    torch.arange(len(out), device=args.device).unsqueeze(0)), mem,
                    tgt_mask=model._causal[:len(out), :len(out)],
                    memory_key_padding_mask=ids == PAD)
                nxt = int(model.out(h)[0, -1].argmax())
                if nxt == EOS:
                    break
                out.append(nxt)
            return "".join(inv.get(i, "") for i in out[1:])

    # 短桶用词级测试集；5-7 字用语料句子（那个长度上语料就是真实输入）
    buckets = build_word_test(test)
    buckets["5-7字"] = [(p["zh"], p["yue"]) for p in test if 5 <= len(p["zh"]) <= 7][:300]
    # 「不该改」的对照桶：成语和古诗在书面粤语里就是原文，正确答案是原样不改
    idioms = [ln.strip() for ln in open(os.path.join(ROOT, "tools", "idioms.txt"), encoding="utf-8")
              if ln.strip() and not ln.startswith("#")]
    poems = [ln.strip() for ln in open(os.path.join(ROOT, "tools", "poems.txt"), encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    buckets["4字成语"] = [(w, t2s.to_traditional(w)) for w in idioms]
    buckets["5-7字古诗"] = [(w, t2s.to_traditional(w)) for w in poems]

    print(f"{'桶':<10}{'n':>5}{'照抄':>8}{'规则':>9}{'神经':>9}{'Δ':>9}{'神经延迟':>10}")
    total_w = 0.0
    rule_w = nn_w = 0.0
    for name in ("2字", "3字", "4字成语", "5-7字", "5-7字古诗"):
        items = buckets.get(name) or []
        if not items:
            print(f"{name:<10}{0:>5}  （语料里没有可用样本）")
            continue
        r_ok = n_ok = c_ok = 0
        lat = []
        for zh, ref in items:
            # 对照基线：原样不改（只做简→繁）能对多少。分数虚高时这一列会暴露出来
            if t2s.to_traditional(zh) == ref:
                c_ok += 1
            if t2s.to_traditional(conv.convert(zh)) == ref:
                r_ok += 1
            if nn_fn:
                t0 = time.perf_counter()
                got = t2s.to_traditional(nn_fn(zh))
                lat.append((time.perf_counter() - t0) * 1000)
                if got == ref:
                    n_ok += 1
        n = len(items)
        rule = r_ok / n
        nn = n_ok / n if nn_fn else float("nan")
        lat.sort()
        w = WEIGHTS.get(name, 0.0)
        total_w += w
        rule_w += w * rule
        if nn_fn:
            nn_w += w * nn
        lat_s = f"{lat[len(lat)//2]:.1f}ms" if lat else "—"
        delta = f"{(nn - rule) * 100:+.1f}pt" if nn_fn else "—"
        print(f"{name:<10}{n:>5}{c_ok / n:>8.1%}{rule:>9.1%}{nn:>9.1%}{delta:>9}{lat_s:>10}")

    if nn_fn and total_w:
        print(f"\n按真实打字分布加权（有样本的桶，权重和 {total_w:.0%}）："
              f"规则 {rule_w / total_w:.1%} → 神经 {nn_w / total_w:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
