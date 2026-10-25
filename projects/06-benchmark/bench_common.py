#!/usr/bin/env python3
"""基准套件共享工具：仓库路径、环境信息、数值与 JSON 落盘。

所有 suite 模块从这里拿仓库根路径与 sys.path 注入，保证：
  1. 题面构造直接复用 tools/ 与 experiments/ 的原实现（口径不漂移）；
  2. 大数据文件（vendor/yue_pairs.json 等）用相对路径引用，不复制。
"""
from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BENCH_DIR, os.pardir, os.pardir))

for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PAIRS = os.path.join(ROOT, "vendor", "yue_pairs.json")
DEFAULT_MODEL = os.path.join(ROOT, "experiments", "yue_model_d256.pt")
DEFAULT_GUARD = os.path.join(ROOT, "data", "yue_guard.eval.marshal")
DEFAULT_LEXICON = os.path.join(ROOT, "data", "lexicon.tsv")
DEFAULT_LM = os.path.join(ROOT, "data", "lm_char.marshal")


def repo(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def file_ok(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def r4(x: float | None) -> float | None:
    """比率统一保留 4 位小数：JSON 稳定、可 diff。"""
    return None if x is None else round(float(x), 4)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def stub_nn():
    """桩模型：原样返回输入（简→繁交给评测里的 t2s 外壳）。

    给无 torch 环境的测试/演示用，产出结构完整但不代表任何真实系统。
    """
    return lambda text: text


def load_yue_nn(model_path: str, device: str = "cpu"):
    """加载粤语 seq2seq（与 experiments/exp_yue_guard.py 同一加载方式）。

    返回 (nn_fn, None) 或 (None, 原因)。nn_fn(text)->str，贪心解码。
    """
    if not file_ok(model_path):
        return None, f"模型文件不存在：{model_path}"
    try:
        import torch
    except Exception as exc:                       # noqa: BLE001
        return None, f"torch 不可用：{exc}"

    from train_yue_seq2seq import EOS, PAD, SOS, Seq2Seq, encode

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
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

    return nn_fn, None


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def seconds(self) -> float:
        return round(time.perf_counter() - self.t0, 2)
