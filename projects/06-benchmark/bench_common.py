#!/usr/bin/env python3
"""基准套件共享工具：仓库路径、环境信息、数值与 JSON 落盘。

所有 suite 模块从这里拿仓库根路径与 sys.path 注入，保证：
  1. 题面构造直接复用 tools/ 与 experiments/ 的原实现（口径不漂移）；
  2. 大数据文件（vendor/yue_pairs.json 等）用相对路径引用，不复制。
"""
from __future__ import annotations

import hashlib
import json
import math
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
DEFAULT_WORD_PAIRS = os.path.join(ROOT, "data", "yue_word_pairs.json")


def load_variant_map() -> dict[str, set[str]]:
    """合法变体表：key → {合法粤语写法集合}（v1.3.0 词桶多答案判分）。

    来源全部是 train 切分的统计（守卫 word_pairs 多目标 + 锚定挖矿的
    yue_word_pairs.json），与考题（test 切分）互斥——用训练侧证据定义
    「什么算对」，不是拿考题答案放水。语料里同一个词本就有多个合法
    写法（什么→乜嘢/咩），单一标准答案会把合法变异记成错。
    """
    variants: dict[str, set[str]] = {}
    try:
        import marshal
        import zlib
        raw = open(DEFAULT_GUARD, "rb").read()
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        for z, ys in (marshal.loads(raw).get("word_pairs") or {}).items():
            variants.setdefault(z, set()).update(ys)
    except (OSError, ValueError, EOFError):
        pass
    try:
        for p in json.load(open(DEFAULT_WORD_PAIRS, encoding="utf-8")):
            if p.get("zh") and p.get("yue"):
                variants.setdefault(p["zh"], set()).add(p["yue"])
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return variants


def accept_forms(key: str, ref: str, variants: dict[str, set[str]],
                 shell=None) -> list[str]:
    """判分可接受的（繁体化）答案集合：标准答案 + 训练侧实证变体。"""
    forms = {ref, *variants.get(key, ())}
    if shell is not None:
        forms = {shell.to_traditional(f) for f in forms}
    return sorted(forms)


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


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    """Wilson 分数区间（95% 默认）。k=成功数、n=总数。

    60~300 条分辨不了 1~3pp 的可操作形式：两个系统的区间不重叠才算可分辨。
    n=0 时返回 None。
    """
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return {"lo": r4(max(0.0, center - half)), "hi": r4(min(1.0, center + half))}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: str | None) -> dict | None:
    """数据/模型文件指纹：相对仓库根的路径 + sha256 前 16 位 + 字节数。

    对标 lm-evaluation-harness 的 dataset revision pinning：结果 JSON 里钉死
    本次评测实际吃进的每个文件，别人复核时先对指纹再对数字。
    """
    if not path or not os.path.exists(path):
        return None
    rel = os.path.relpath(path, ROOT)
    return {"path": rel, "sha256_16": sha256_file(path)[:16],
            "bytes": os.path.getsize(path)}


def question_hash(items: list) -> str:
    """题面序列的规范哈希（防调包：提交评分前先对题面指纹）。

    items 为可 JSON 化的题面列表（dict / tuple 先转 list）。
    """
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
