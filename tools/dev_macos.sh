#!/usr/bin/env bash
# macOS 开发环境（不改动系统 Python，不装 IBus）
#
# 用途：在 Apple Silicon / Intel Mac 上开发、跑单测、训练粤语神经模型。
# 注意：IBus 引擎本体只能在 Linux 上装，macOS 上跑不了输入法，但纯逻辑层
#      （ime / lexicon / segmenter / user_model）和训练/评测全都能跑。
#
# 用法：
#   ./tools/dev_macos.sh              # 建 .venv + 装依赖 + 跑单测
#   ./tools/dev_macos.sh --with-nn    # 额外装 torch/onnx（训练用，约 300MB）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_NN=0
for arg in "$@"; do
  case "$arg" in
    --with-nn) WITH_NN=1 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

# 大陆网络下 PyPI 官方源装 torch 实测 >10 分钟且常卡住，默认走清华源；
# 想用官方源就设 PIP_INDEX_URL。
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [ ! -d .venv ]; then
  echo "== 建虚拟环境 .venv =="
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q --index-url "$INDEX" pytest

if [ "$WITH_NN" = "1" ]; then
  echo "== 安装训练依赖（torch / pyarrow / onnx / onnxruntime）=="
  python -m pip install -q --index-url "$INDEX" torch pyarrow onnx onnxruntime onnxscript
  python - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None)
print(f"torch {torch.__version__} | mps 可用: {bool(mps and mps.is_available())}")
PY
fi

echo "== 跑单元测试（纯逻辑层，IBus 无关）=="
# 固定用项目内 basetemp：沙箱/受限环境里系统临时目录可能不可写，
# pytest 会报 "PermissionError: EEXIST ... pytest-of-<user>"
python -m pytest tests -q --basetemp=.pytest-tmp

cat <<'EOF'

开发环境就绪。常用命令：

  source .venv/bin/activate
  python -m pytest tests -q --basetemp=.pytest-tmp   # 单测（macOS 可跑）
  python tools/eval_quality.py                  # 拼音质量基准
  python tools/fetch_yue_pairs.py               # 下语料（自动走 hf-mirror）
  python experiments/train_yue_seq2seq.py \
      --limit 150000 --epochs 4 --d 256 --layers 4 --device mps
  python experiments/eval_yue_seq2seq.py --beam 4

macOS 上不能做的：make install（IBus 引擎）、make e2e（需要 ibus daemon）、
packaging/build_deb.sh（dpkg）。这些请放到 Linux 机器或容器里跑。
EOF
