#!/usr/bin/env python3
"""粤语 → 普通话 批量转换 CLI（项目①基准）。

用法：
  echo '今日唔得閒，聽日先講。' | python3 projects/01-yue2zh/convert.py
  python3 projects/01-yue2zh/convert.py in.txt
  python3 projects/01-yue2zh/convert.py --model other.pt < in.txt

整段文字按终止标点/换行分句，逐句贪心解码，一行一句输出。
顶层的清洗/分句/编排函数不依赖 torch，test_convert.py 可直接单测；
模型相关代码惰性导入，只有真正转换时才加载。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_MODEL = os.path.join(HERE, "yue2zh_model.pt")

_CR_RE = re.compile(r"\r\n?")
_ZW_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_SENT_RE = re.compile(r"([。！？；!?…]+)")
_PUNCT_ONLY_RE = re.compile(r"[\W_]+", re.UNICODE)


def clean_text(text: str) -> str:
    """换行统一成 \n，去掉零宽字符。"""
    return _ZW_RE.sub("", _CR_RE.sub("\n", text))


def segment_sentences(text: str) -> list[str]:
    """按终止标点（。！？；!?…）和换行分句；标点跟随前句，空句与纯标点句丢弃。"""
    sentences: list[str] = []
    for para in text.split("\n"):
        parts = _SENT_RE.split(para)
        for i in range(0, len(parts), 2):
            body = parts[i]
            delim = parts[i + 1] if i + 1 < len(parts) else ""
            s = (body + delim).strip()
            if not s or _PUNCT_ONLY_RE.fullmatch(s):
                continue
            sentences.append(s)
    return sentences


def convert_text(text: str, decode_fn) -> list[str]:
    """清洗 → 分句 → 逐句翻译。decode_fn(sentences) 返回等长译句列表。"""
    sentences = segment_sentences(clean_text(text))
    if not sentences:
        return []
    return decode_fn(sentences)


def load_model(path: str = DEFAULT_MODEL, device: str = "auto"):
    """加载训练检查点，返回 {model, src_vocab, tgt_vocab, device, args, epoch}。"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import torch

    from experiments.train_yue_seq2seq import Seq2Seq

    if device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trained = ckpt.get("args") or {}
    model = Seq2Seq(len(ckpt["src_vocab"]) + 4, len(ckpt["tgt_vocab"]) + 4,
                    d=int(trained.get("d", 256)), layers=int(trained.get("layers", 4)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return {"model": model.to(device), "src_vocab": ckpt["src_vocab"],
            "tgt_vocab": ckpt["tgt_vocab"], "device": device, "args": trained,
            "epoch": ckpt.get("epoch")}


def greedy_decode(model, src_vocab, tgt_vocab, device, sentences: list[str],
                  batch_size: int = 128, max_len: int = 42) -> list[str]:
    """批量贪心解码（无 KV cache，与 experiments/eval_yue_seq2seq.py 的单句贪心等价）。

    src 用粤语词表编码，tgt 用普通话词表反查；输入超 40 字截断（与训练一致）。
    """
    import torch

    from experiments.train_yue_seq2seq import EOS, PAD, SOS, encode

    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    results: list[str] = [""] * len(sentences)
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start:start + batch_size]
            rows = [encode(s, src_vocab) for s in chunk]
            width = max(len(r) for r in rows)
            src = torch.tensor([r + [PAD] * (width - len(r)) for r in rows], device=device)
            src_pad = src == PAD
            memory = model.encoder(
                model.src_emb(src) + model.pos(torch.arange(width, device=device)).unsqueeze(0),
                src_key_padding_mask=src_pad)
            batch = len(chunk)
            out = torch.full((batch, 1), SOS, dtype=torch.long, device=device)
            finished = torch.zeros(batch, dtype=torch.bool, device=device)
            for _ in range(max_len):
                t = out.size(1)
                hidden = model.decoder(
                    model.tgt_emb(out) + model.pos(torch.arange(t, device=device)).unsqueeze(0),
                    memory, tgt_mask=model._causal[:t, :t],
                    memory_key_padding_mask=src_pad)
                nxt = model.out(hidden[:, -1]).argmax(-1)
                nxt = nxt.masked_fill(finished, PAD)   # 已结束的行保持 PAD，不再生长
                out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
                finished |= nxt == EOS
                if bool(finished.all()):
                    break
            for b, row in enumerate(out[:, 1:].tolist()):
                results[start + b] = "".join(
                    inv_tgt.get(tok, "") for tok in row if tok not in (EOS, PAD))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="粤语 → 普通话 逐句转换（贪心解码）")
    ap.add_argument("infile", nargs="?", default=None, help="输入文件；缺省读 stdin")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto/cpu/mps")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    text = open(args.infile, encoding="utf-8").read() if args.infile else sys.stdin.read()
    if not text.strip():
        return 0
    pack = load_model(args.model, args.device)
    outs = convert_text(text, lambda sents: greedy_decode(
        pack["model"], pack["src_vocab"], pack["tgt_vocab"], pack["device"], sents,
        batch_size=args.batch))
    sys.stdout.write("\n".join(outs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
