# 训练粤语神经模型（Apple M 系列 / 有 GPU 的机器）

目标：把 `普通话 → 粤语` 的整句一致率从规则版的 ~12% 推到 35% 以上。
模型是可选的：没有模型文件时输入法自动走规则，不受影响。

> 实测（M2 Pro / 16GB / torch 2.14 / MPS，2026-10-24）：
> 15 万句 × 4 轮、d=256、4 层 = **约 9 分钟**，验证集整句一致
> `23.5% → 32.2% → 35.2% → 36.5%`，字符重合 79.6%。
> 同一批 held-out 上规则版整句 12.0% / 字符 60.4%。

## 0. 准备（约 5 分钟）

```bash
git clone https://github.com/YaYII/ipinyin.git && cd ipinyin
./tools/dev_macos.sh --with-nn      # 建 .venv + 装依赖 + 跑单测
source .venv/bin/activate
```

手动装也行，但**大陆网络别直接用官方源**（实测装 torch 卡十多分钟）：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch pyarrow onnx onnxruntime onnxscript
```

## 1. 下载平行语料（169,949 句对，约 18MB）

```bash
python3 tools/fetch_yue_pairs.py       # → vendor/yue_pairs.json
```

脚本按 `HF_ENDPOINT` 环境变量 → huggingface.co → hf-mirror.com 顺序尝试。
大陆网络下 huggingface.co 会超时（25s），会自动落到镜像，不用手动配。

## 2. 训练

```bash
python3 experiments/train_yue_seq2seq.py \
    --limit 150000 --epochs 4 --d 256 --layers 4 --device mps \
    --out experiments/yue_model.pt
```

- `--device auto` 会依次试 `cuda` / `mps` / `cpu`；M 系列上显式写 `mps` 更直观
- `--resume` 从 `--out` 续训（会恢复优化器状态和第几轮）
- 训练脚本自动按长度分桶，同类长度拼一批，比原来少一半以上的 PAD 计算
- 有 warmup + cosine 衰减；原来没有，前期 loss 会先炸一段

参考进度（M2 Pro，15 万句，d=256/4 层，batch 96）：

| 轮次 | 平均 loss | 验证整句 | 验证字符 | 用时 |
| --- | --- | --- | --- | --- |
| 0 | 3.437 | 23.5% | 70.8% | 126s |
| 1 | 1.923 | 32.2% | 76.9% | 120s |
| 2 | 1.739 | 35.2% | 79.0% | 124s |
| 3 | 1.675 | 36.5% | 79.6% | 124s |

## 3. 导出并量化

```bash
python3 experiments/export_onnx.py experiments/yue_model.pt experiments
python3 tools/quantize_yue_nn.py experiments/yue_nn.onnx
```

产出：`yue_nn_int8.onnx`（d=256/4 层约 **12MB**）+ `yue_nn_vocab.json`。
导出脚本会从 checkpoint 里读 `d` / `layers`，不用手动对齐架构。

体积想更小就训练时降到 `--d 192 --layers 3`（约 7MB），分数会掉 1~2 个点。

## 4. 启用

```bash
cp experiments/yue_nn_int8.onnx ~/.local/share/ipinyin/data/yue_nn.onnx
cp experiments/yue_nn_vocab.json ~/.local/share/ipinyin/data/
systemctl --user restart ipinyin-engine
grep 神经 ~/.cache/ipinyin/engine.log | tail -1     # 看到"粤语神经模型已加载"即生效
```

**只在粤语模式（Tab 切过去）、且只对首选候选推理** —— 候选列表里每个候选都会调一次
`convert()`，2~4 字词单次 7~10ms，9 个候选全算就是 60ms+/键，14 字整句实测逐键
p50 从 0.1ms 涨到 38ms。不想要了：删掉那两个文件，或设 `IPINYIN_YUE_NN=0`。

**注意语域风险**：模型在口语词上明显更好（2 字 60%→91%），但会把成语和古诗改坏
（4 字成语误改 27.5%、古诗误改 50%，规则版分别是 2.5% 和 0%）。
详见 [nn-upgrade.md](nn-upgrade.md)。

## 5. 自己验分

```bash
python3 tools/eval_ime_lengths.py --model experiments/yue_model.pt   # 按真实打字长度
python3 experiments/eval_yue_seq2seq.py experiments/yue_model.pt --beam 1   # 长句
```

`eval_ime_lengths.py` 是主指标：2/3/4/5-7 字分桶，带「照抄不改」对照列。
`--beam 4` 实测**不会更好**（36.1% vs 37.7%）且慢 3 倍，别开。

## 已知限制

- 现有 ONNX 图内部 Reshape 形状固化在 40，变长输入会报 `cannot reshape`；
  想要变长/KV cache 得改模型结构后用 dynamo 导出（见 `docs/nn-upgrade.md`）
- 量化到 int8 掉约 2 个点（37.7% → 35.0%）
- 不要在候选重建路径里做神经推理，延迟扛不住
