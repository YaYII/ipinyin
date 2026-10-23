#!/usr/bin/env python3
"""把导出的 yue_nn.onnx 量化成 int8（体积减半以上，CPU 推理更快）。

用法：python3 tools/quantize_yue_nn.py experiments/yue_nn.onnx
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.yue_nn import quantize  # noqa: E402


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "yue_nn.onnx")
    dst = os.path.join(os.path.dirname(src), "yue_nn_int8.onnx")
    if not os.path.exists(src):
        print(f"找不到 {src}", file=sys.stderr)
        return 1
    if not quantize(src, dst):
        print("量化失败（需要 onnxruntime）", file=sys.stderr)
        return 1
    print(f"{os.path.getsize(src)/1e6:.1f} MB → {os.path.getsize(dst)/1e6:.1f} MB：{dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
