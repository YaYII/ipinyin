"""可选的神经粤语转换（ONNX Runtime）。

模型文件存在才启用：
  ~/.local/share/ipinyin/data/yue_nn.onnx
  ~/.local/share/ipinyin/data/yue_nn_vocab.json
没有模型或没装 onnxruntime 时静默返回 None，输入法自动退回规则版。
"""
from __future__ import annotations

import json
import os
from typing import Optional

PAD, SOS, EOS, UNK = 0, 1, 2, 3


class NeuralYue:
    __slots__ = ("_session", "_src", "_inv_tgt", "_max_len")

    def __init__(self, session, src_vocab: dict[str, int], tgt_vocab: dict[str, int], max_len: int = 40):
        self._session = session
        self._src = src_vocab
        self._inv_tgt = {i: ch for ch, i in tgt_vocab.items()}
        self._max_len = max_len

    @classmethod
    def load(cls, data_dir: str) -> Optional["NeuralYue"]:
        onnx_path = os.path.join(data_dir, "yue_nn.onnx")
        vocab_path = os.path.join(data_dir, "yue_nn_vocab.json")
        if not (os.path.exists(onnx_path) and os.path.exists(vocab_path)):
            return None
        try:
            import numpy as np            # noqa: F401
            import onnxruntime as ort
        except ImportError:
            return None
        try:
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            vocab = json.load(open(vocab_path, encoding="utf-8"))
        except Exception:
            return None
        return cls(session, vocab.get("src", {}), vocab.get("tgt", {}))

    def translate(self, text: str) -> str:
        import numpy as np

        width = self._max_len
        ids = [SOS] + [self._src.get(ch, UNK) for ch in text][: width - 2] + [EOS]
        src = np.full((1, width), PAD, dtype=np.int64)
        src[0, : len(ids)] = ids
        out = [SOS]
        for _ in range(width - 1):
            tgt = np.full((1, width), PAD, dtype=np.int64)
            tgt[0, : len(out)] = out
            logits = self._session.run(None, {"src": src, "tgt": tgt})[0]
            nxt = int(logits[0, len(out) - 1].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
        return "".join(self._inv_tgt.get(i, "") for i in out[1:])

    def __len__(self) -> int:
        return len(self._src)


def quantize(src_path: str, dst_path: str) -> bool:
    """把 fp32 的 ONNX 量化成 int8（体积约降一半，精度略降）。"""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        return False
    try:
        quantize_dynamic(src_path, dst_path, weight_type=QuantType.QInt8)
        return True
    except Exception:
        return False
