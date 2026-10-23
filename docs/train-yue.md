# 训练粤语神经模型（Apple M 系列 / 有 GPU 的机器）

目标：把 `普通话 → 粤语` 的整句一致率从规则版的 ~30% 推到 45% 以上。
模型是可选的：没有模型文件时输入法自动走规则，不受影响。

## 0. 准备（约 5 分钟）

```bash
git clone https://github.com/YaYII/ipinyin.git && cd ipinyin
python3 -m venv .venv && source .venv/bin/activate
pip install torch pyarrow onnxscript onnx onnxruntime
```

## 1. 下载平行语料（169,949 句对，约 20MB）

```bash
python3 tools/fetch_yue_pairs.py          # → vendor/yue_pairs.json
```

## 2. 训练

```bash
python3 experiments/train_yue_seq2seq.py \
    --limit 150000 --epochs 4 --d 256 --layers 4 \
    --device auto --out experiments/yue_model.pt
```

- `--device auto` 会自动选 `mps`（Apple M 系列）/ `cuda` / `cpu`
- 每个 epoch 都会存检查点；中断后加 `--resume` 继续
- 控制台会打印 `[验证] 整句一致 x% 字符重合 y%`，这就是进度指标

参考：纯 CPU（8 核）上 5.4 万句对 2 轮 ≈ 18 分钟，验证集整句一致 6.0% → 12.8%。
M2（MPS）预计快 5~10 倍，150k×4 轮大约 1~2 小时。

## 3. 导出并量化

```bash
python3 experiments/export_onnx.py experiments/yue_model.pt experiments
python3 tools/quantize_yue_nn.py experiments/yue_nn.onnx
```

产出：`yue_nn_int8.onnx`（约 3~5MB）+ `yue_nn_vocab.json`

## 4. 启用

```bash
cp experiments/yue_nn_int8.onnx ~/.local/share/ipinyin/data/yue_nn.onnx
cp experiments/yue_nn_vocab.json ~/.local/share/ipinyin/data/
systemctl --user restart ipinyin-engine
grep 神经 ~/.cache/ipinyin/engine.log | tail -1     # 看到"粤语神经模型已加载"即生效
```

长句（≥8 字）走神经模型，短词仍走规则（稳定、快）。
不想要了：删掉那两个文件，或设 `IPINYIN_YUE_NN=0`。

## 5. 自己验分

```bash
python3 tools/eval_cantonese.py vendor/yue_pairs.json
```
