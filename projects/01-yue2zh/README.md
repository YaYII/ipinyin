# 项目①：粤语 → 普通话 反向阅读助手（基准）

用户贴入粤语书面语（繁体、带粤字「唔/嘅/咗/喺/哋…」），输出可直接读懂的普通话。
本目录是**字符级 Transformer seq2seq 的基准（baseline）**：先量出「不接任何规则、
纯端到端」能做到什么水平，再决定后面是继续堆模型还是走规则融合。

模型方向与 `experiments/train_yue_seq2seq.py`（普通话→粤语）相反：src = 粤语，tgt = 普通话。
模型类、词表构建、编码函数直接从该脚本 import（只读复用，未改动 `experiments/`）。

## 文件

| 文件 | 说明 |
|---|---|
| `train.py` | 训练驱动：加载语料、交换方向、MPS 训练、每轮存检查点并在 held-out 抽检；`--typo` 同音错字增强 |
| `convert.py` | 批量转换 CLI：stdin/文件 → 逐句贪心解码 → 普通话输出 |
| `eval.py` | held-out 评测：整句一致率 + 字符重合率 + chrF，按句长分桶，「照抄不改」对照；`--typo-p` 抗噪卷 |
| `typo.py` | 同音错字模块：`data/yue.marshal` 粤拼表按声韵（去调）分组；注入函数纯逻辑可单测 |
| `fetch_extra_corpus.py` | 三源语料合并（repo + msgchi + botisan），去重、zh 归一简体 → `data/merged_pairs.json` |
| `test_convert.py` / `test_typo.py` / `test_eval.py` | pytest 单测（纯逻辑，不加载模型） |
| `yue2zh_model*.pt` | 三代训练产物（34.9 / 38.3 / 38.3 MB），**不提交**，重训可复现 |
| `train*.log` / `eval*.log` | 各次训练与评测的完整输出留档 |
| `data/merged_pairs.json` | 合并语料 267,896 对（27 MB，gitignore，`fetch_extra_corpus.py` 再生） |

## 数据来源与切分

- 语料：`vendor/yue_pairs.json`，169,949 句对（`tools/fetch_yue_pairs.py` 抓取，
  字段 `zh` 简体普通话 / `yue` 繁体粤语，来源为网络评论语料）。
- 切分沿用仓库惯例（与 `experiments/train_yue_seq2seq.py` 完全一致）：
  `random.seed(0)` 洗牌 → 过滤空句与双边 >38 字的句子 → 取前 `--limit`（本项目 100,000）
  → **前 max(500, 1/20) = 5,000 句为 held-out 测试**，只评它；其余 95,000 句全部做训练。
  不设验证集，每轮训练结束在 held-out 前 200 句上抽检防跑飞。
- `eval.py --limit` 必须与训练一致，否则切出的 held-out 不同。

## 训练配置与实测

```
python3 projects/01-yue2zh/train.py --limit 100000 --epochs 3
```

- 结构：d=256、encoder/decoder 各 4 层、4 头，字符级词表（上限 6000，实际 src 4662 / tgt 4283），
  共 8.7M 参数，检查点 34.9 MB。
- 设备 MPS（Apple M 系列），fp32，warmup 300 步 + cosine 退火，lr 3e-4，batch 96（默认值）。
- **总训练时长 267 秒（约 4.5 分钟）**，末轮平均 loss 1.830。
- 训练中 held-out 抽检（n=200）整句一致率：epoch0 16.0% → epoch1 23.5% → epoch2 29.0%
  （字符 62.1% → 68.5% → 69.7%）。

## held-out 评测分数（5,000 句，贪心解码）

```
python3 projects/01-yue2zh/eval.py            # 全部 held-out
python3 projects/01-yue2zh/eval.py --n 1000   # 抽前 1000 句
```

| 句长桶（粤语句字符数） | n | 照抄整句 | 照抄字符 | 神经整句 | 神经字符 | Δ整句 |
|---|---:|---:|---:|---:|---:|---:|
| ≤8   |   472 |  2.1% | 35.9% | 36.4% | 68.1% | +34.3pt |
| 9-15 | 2,972 |  0.1% | 34.5% | 38.4% | 77.5% | +38.3pt |
| 16+  | 1,556 |  0.0% | 32.7% | 18.7% | 67.6% | +18.7pt |
| 全部 | 5,000 |  0.3% | 34.0% | 32.1% | 73.6% | +31.8pt |

- 「照抄不改」= 把粤语输入原样当输出。粤语与普通话共享大量字词，但它整句几乎
  从不成立（0.3%），字符重合率 ~34%——差异集中在粤字与语序上，正是模型要学的部分。
- 模型在全部桶上都大幅超过照抄基线；9-15 字句最好（38.4% / 77.5%），
  16+ 长句明显衰减（18.7%），是最薄弱的一段。
- 速度：MPS batch 256 下 5,000 句贪心解码 14.3 s（约 2.9 ms/句）；CPU 约 10 ms/句。

## 用法

```bash
PY=/Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python

# 训练（约 4.5 分钟，MPS）
$PY projects/01-yue2zh/train.py --limit 100000 --epochs 3

# 转换：stdin 或文件，逐句一行输出
echo '今日唔得閒，聽日先講。' | $PY projects/01-yue2zh/convert.py
# => 今天不行空，明天才说。
$PY projects/01-yue2zh/convert.py input.txt

# 评测
$PY projects/01-yue2zh/eval.py

# 单测（不需要模型文件）
$PY -m pytest projects/01-yue2zh/test_convert.py
```

`convert.py` 的顶层清洗/分句函数不依赖 torch；模型加载与解码惰性导入，
`load_model()` / `greedy_decode()` 也被 `train.py`、`eval.py` 复用。

## 已知局限

- **语气词与一词多义最伤**：语气词译得犹豫，句尾常多字或走样
  （「唔通真係好似之前巴打咁咩」→「难道真的像之前兄弟**那么**吗」，参考是「这样」；
  「唔係佢寫嘅噉」→「不是他写的这样」，参考是「似的」）；「唔得閒」→「不行空」，应为「没空」。
- **长句衰减**：16+ 桶整句一致率 18.7%，长句后半段容易跑偏或蹦出不存在的词
  （「所有人都向住同一點進發」→「所有人都向著**和**一点进发」；「返工」→「回班」）。
- **贪心解码**：无 beam search，错了不能回头；偶发局部重复（早期 epoch 更明显）。
- **语料是网络评论**：口语覆盖好，书面语/领域词弱；粤语源句里的错别字与生僻词会被「忠实」译错
  （「係路」实为「细路」的谐音，模型译成「是路」；「倆仔乸」→「小子母」）。
- 词表上限 6000，未登录字统一映射 UNK，生僻字会丢。
- 输出是简体普通话（tgt 侧本身是简体），没有繁体选项。

## 下一步

1. **接入输入法第四模式**：`convert.py` 的分句 + 批量贪心已具备按句转换的形态，
   可包一层与 `src/ipinyin/yue.py` 相同的接口；长句（≥8 字）走本模型、短句走规则，与
   正向的 `NN_MIN_LEN` 策略对称。
2. **粤拼注音**：给粤字附 jyutping，既做输出附加行，也可作为解码端的辅助特征。
3. **融合**：把 `data/yue.marshal` 反向词典（粤→普）的查表结果与神经输出做冲突检测/置信融合，
   专治语气词与高频词；预期短句桶提升最明显。
4. 训练侧：beam search、16+ 长句过采样、把 `limit` 提到全部 17 万句。

## 提交说明

- 本目录的 `yue2zh_model.pt`（34.9 MB）与 `*.log` 为本地训练产物，**不提交**——
  由目录内 `.gitignore` 拦截；重跑上面的训练命令即可按同一种子复现（`random.seed(0)` + `torch.manual_seed(0)`）。
- 未改动 `src/`、`experiments/`、`tools/`、`Makefile` 中的任何文件。


## 语料扩容重训（2026-10-26）

`fetch_extra_corpus.py` 合并三源：repo 167,562 + msgchi（raptorkwok/
cantonese-chinese-parallel-corpus-base，zh 侧繁体已归一简体）93,782 +
botisan 6,552 = **267,896 对**（+60%）。
勘误：调研时把 indiejoseph/yue-zh-translation 当新增量，实测与现有语料
**逐对相同**（169,949 对 100% 重合），真增量是 msgchi/botisan。

同卷对比（合并语料 held-out 12,000 句，两模型同卷同解码）：

| 桶 | 旧模型（17 万训） | 新模型（24 万训） | Δ |
|---|---:|---:|---:|
| ≤8 | 20.2% | 23.5% | +3.3 |
| 9-15 | 32.0% | 37.4% | +5.4 |
| 16+ | 15.6% | **22.8%** | **+7.2** |
| 全部整句 | 24.6% | **30.0%** | **+5.4** |
| 全部字符 | 59.4% | 61.7% | +2.3 |

预测兑现：**语料扩在最弱的长句桶上收益最大（+7.2pt）**。训练 15.2 分钟
（MPS fp16，240k 句 × 3 epochs）。注意旧 README 的 31.0% 是旧语料 held-out
上的数，新卷更杂（照抄基线 0.3%→0.8%），跨卷不可比。

下一步：语气词专项（囉/噉 类与语境词典融合）、beam search、输入法第四模式
接入（长句走本模型、短句走规则）。


## 同音错字增强 + chrF（2026-10-26，第二轮）

社区调研落地两件事（借鉴 [AlienKevin/bart-canto-mando-bing-300K-typo](https://huggingface.co/AlienKevin/bart-canto-mando-bing-300K-typo)
的 typo 增强、[CantonMT](https://arxiv.org/html/2403.11346v1) 的评测口径）：

1. **同音错字增强**（`typo.py` + `train.py --typo`）：`data/yue.marshal` 的 27,007 字
   粤拼表按「声母+韵母」（去声调）分组同音字；训练时对粤语侧每字以 p=0.12 概率在线注入
   词表内同音字。**键字与候选都限制在 src 词表内**——否则会把 UNK 生僻字「修好」，
   注入方向就反了（冒烟时抓到过这个 bug）。种子 1234，同一句每轮噪声不同。
2. **chrF 指标**（`eval.py`）：Popović 2015 / sacrebleu 默认口径的纯 Python 实现
   （字符 1..6-gram corpus 级聚合、F_beta=2、忽略空白、双侧皆空的阶跳过），
   手工可口算样例单测兜底（`test_eval.py`）。

训练与 merged 模型**完全同配置**（同语料、limit 240000、3 epochs、batch 96、amp fp16，
唯一差别 `--typo 0.12`）：12.4 分钟，末轮 loss 3.035（输入带噪声，损失天然偏高，
与 merged 的数字不可直接比）。

### 两张卷对照（同一 held-out 12,000 句，贪心解码）

干净卷（输入无错字）：

| 模型 | ≤8 | 9-15 | 16+ | 全部整句 | 全部字符 | 全部 chrF |
|---|---:|---:|---:|---:|---:|---:|
| merged（无增强） | 23.5% | 37.4% | 22.8% | **30.0%** | **61.7%** | **63.7%** |
| typo（p=0.12） | 21.7% | 33.4% | 19.8% | 26.9% | 60.2% | 61.5% |

抗噪卷（同一 held-out 注入同音错字 p=0.12，种子 4321；8,570/12,000 句被改动；
`eval.py --typo-p 0.12`）：

| 模型 | ≤8 | 9-15 | 16+ | 全部整句 | 全部字符 | 全部 chrF |
|---|---:|---:|---:|---:|---:|---:|
| merged（无增强） | 12.8% | 9.3% | 2.8% | 8.2% | 49.5% | 43.9% |
| typo（p=0.12） | 14.9% | 18.3% | 8.1% | **14.6%** | **55.1%** | **52.1%** |

结论：**代价干净卷整句 −3.1pt，换噪声卷整句近翻倍（8.2%→14.6%）、chrF +8.2pt**，
长句桶受益最大（2.8%→8.1%，近 3 倍）。真实用户输入（论坛/聊天风格）介于两卷之间。
部署取向：干净文本（文章、字幕）用 merged；口语抓取转写用 typo。

已知偏差：合成噪声是「均匀同音采样」，真实错字分布不均匀（高频字更易混、形近也占一环），
抗噪卷分数是机制验证，不是真实分布的精确估计。p 未调参（0.06~0.08 可能更平衡）。

复现：

```bash
PY=/Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python
$PY projects/01-yue2zh/train.py --pairs projects/01-yue2zh/data/merged_pairs.json \
    --limit 240000 --epochs 3 --batch 96 --amp --typo 0.12 \
    --out projects/01-yue2zh/yue2zh_model_typo.pt
$PY projects/01-yue2zh/eval.py --model projects/01-yue2zh/yue2zh_model_typo.pt \
    --pairs projects/01-yue2zh/data/merged_pairs.json --limit 240000            # 干净卷
$PY projects/01-yue2zh/eval.py --model projects/01-yue2zh/yue2zh_model_typo.pt \
    --pairs projects/01-yue2zh/data/merged_pairs.json --limit 240000 --typo-p 0.12  # 抗噪卷
```

遗留：p 调参 / 按真实错字分布加权采样、查表词免疫错字（P2 词典融合可与此叠加：
查表命中的词锁定后，错字只影响未命中段）。
