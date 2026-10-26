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
import json
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


def beam_decode(model, src_vocab, tgt_vocab, device, sentences: list[str],
                beam: int = 4, batch_size: int = 32,
                max_len: int = 42) -> list[list[tuple[float, str]]]:
    """批量束搜索：每句返回 beam 个候选 [(长度归一分, 译文)]，降序。

    与 greedy_decode 同口径（同 encode/同掩码）；beam=1 时第一条候选应与贪心一致。
    长度归一 score/(len-1) 防止偏爱短句，与 experiments/eval_yue_seq2seq.py 的
    单句 beam 一致。分组步进：同一批内各源句独立选 top-beam，EOS 进 done 冻结。
    """
    import torch

    from experiments.train_yue_seq2seq import EOS, PAD, SOS, encode

    inv_tgt = {i: c for c, i in tgt_vocab.items()}
    results: list[list[tuple[float, str]]] = []
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

            n = len(chunk)
            # live[i]: 该源句的 (score, tokens) 列表；done[i]: 已出 EOS 的候选
            live: list[list[tuple[float, list[int]]]] = [[(0.0, [SOS])] for _ in range(n)]
            done: list[list[tuple[float, list[int]]]] = [[] for _ in range(n)]
            for _ in range(max_len):
                if not any(live):
                    break
                flat: list[tuple[int, float, list[int]]] = []   # (源句号, 分数, tokens)
                for i, group in enumerate(live):
                    for score, toks in group:
                        flat.append((i, score, toks))
                width_t = len(flat[0][2])
                tgt = torch.tensor([f[2] for f in flat], device=device)
                mem_idx = torch.tensor([f[0] for f in flat], device=device)
                hidden = model.decoder(
                    model.tgt_emb(tgt) + model.pos(torch.arange(width_t, device=device)).unsqueeze(0),
                    memory[mem_idx], tgt_mask=model._causal[:width_t, :width_t],
                    memory_key_padding_mask=src_pad[mem_idx])
                logp = torch.log_softmax(model.out(hidden[:, -1]).float(), dim=-1)

                # 逐源句选 top-beam（组在 flat 里连续，直接切片）
                new_live: list[list[tuple[float, list[int]]]] = [[] for _ in range(n)]
                pos = 0
                for i, group in enumerate(live):
                    g = len(group)
                    if g == 0:
                        continue
                    total = logp[pos:pos + g] + torch.tensor(
                        [f[1] for f in flat[pos:pos + g]], device=device).unsqueeze(1)
                    k = min(beam, total.numel())
                    topv, topi = total.reshape(-1).topk(k)
                    for val, idx in zip(topv.tolist(), topi.tolist()):
                        hyp, tok = divmod(idx, total.size(1))
                        toks = flat[pos + hyp][2] + [tok]
                        if tok == EOS:
                            done[i].append((val / max(len(toks) - 1, 1), toks))
                        else:
                            new_live[i].append((val, toks))
                    pos += g
                live = new_live

            for i in range(n):
                if not done[i]:                      # 一步 EOS 都没出：用 live 兜底
                    done[i] = [(s / max(len(t) - 1, 1), t) for s, t in live[i]]
                done[i].sort(key=lambda kv: kv[0], reverse=True)
                cands = []
                for score, toks in done[i][:beam]:
                    text = "".join(inv_tgt.get(t, "") for t in toks[1:] if t != EOS)
                    cands.append((score, text))
                results.append(cands)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="粤语 → 普通话 逐句转换（贪心解码）")
    ap.add_argument("infile", nargs="?", default=None, help="输入文件；缺省读 stdin")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto/cpu/mps")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--beam", type=int, default=1,
                    help="束宽（1=贪心，默认）；beam≈4 提整句约 +0.8pt，延迟约 11 倍")
    ap.add_argument("--rerank-lambda", type=float, default=0.0,
                    help=">0 时用反向词典对 beam 候选做覆盖度重排（dev 调参选 0.25）")
    args = ap.parse_args()

    text = open(args.infile, encoding="utf-8").read() if args.infile else sys.stdin.read()
    if not text.strip():
        return 0
    pack = load_model(args.model, args.device)

    def decode(sents: list[str]) -> list[str]:
        if args.beam > 1:
            cands = beam_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"],
                                pack["device"], sents, beam=args.beam)
            if args.rerank_lambda > 0:
                from dict_fuse import load_default_reverse_dict, rerank
                corpus_path = os.path.join(HERE, "data", "merged_pairs.json")
                corpus = (json.load(open(corpus_path, encoding="utf-8"))
                          if os.path.exists(corpus_path) else None)
                rdict = load_default_reverse_dict(corpus_pairs=corpus)
                return [rerank(cs, s, rdict, args.rerank_lambda)[0][1]
                        for cs, s in zip(cands, sents)]
            return [cs[0][1] for cs in cands]
        return greedy_decode(pack["model"], pack["src_vocab"], pack["tgt_vocab"],
                             pack["device"], sents, batch_size=args.batch)

    outs = convert_text(text, decode)
    sys.stdout.write("\n".join(outs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
