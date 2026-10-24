#!/usr/bin/env python3
"""把训练好的 checkpoint 导出为 ONNX（运行时只需要 onnxruntime，不需要 torch）。

用法：python3 experiments/export_onnx.py experiments/yue_model.pt [输出目录]
产出：yue_nn.onnx + yue_nn_vocab.json（放到 ~/.local/share/ipinyin/data/ 即生效）
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from train_yue_seq2seq import Seq2Seq  # noqa: E402


class Step(torch.nn.Module):
    """输入 (src, tgt_prefix)，输出每一步的 logits，方便在 Python 里做贪心解码。"""

    def __init__(self, model: Seq2Seq):
        super().__init__()
        self.model = model

    def forward(self, src, tgt):
        return self.model(src, tgt)


def main() -> int:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "yue_model.pt")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "experiments")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
    # 架构超参必须跟训练时一致，否则 state_dict 尺寸不匹配（训练用 --d 256 --layers 4 就炸）
    trained = ckpt.get("args") or {}
    d, layers = int(trained.get("d", 160)), int(trained.get("layers", 2))
    model = Seq2Seq(len(src_vocab) + 4, len(tgt_vocab) + 4, d=d, layers=layers)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "yue_nn.onnx")
    torch.onnx.export(
        Step(model),
        # 用最大长度做 dummy：旧导出器会把内部 reshape 的形状固化，
        # 固化在 40 就统一按 40 补齐输入（掩码仍由模型内部按 PAD 计算）
        (torch.full((1, 40), 5, dtype=torch.long), torch.full((1, 40), 7, dtype=torch.long)),
        onnx_path,
        input_names=["src", "tgt"],
        output_names=["logits"],
        dynamic_axes={"src": {0: "b", 1: "s"}, "tgt": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
        opset_version=17,
        dynamo=False,          # 旧导出器更稳（新 dynamo 对 Transformer mask 支持不全）
    )
    with open(os.path.join(out_dir, "yue_nn_vocab.json"), "w", encoding="utf-8") as fh:
        json.dump({"src": src_vocab, "tgt": tgt_vocab}, fh, ensure_ascii=False)
    print(f"导出 {onnx_path}（{os.path.getsize(onnx_path)/1e6:.1f} MB）")
    print("把 yue_nn.onnx + yue_nn_vocab.json 放进 ~/.local/share/ipinyin/data/ 即启用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
