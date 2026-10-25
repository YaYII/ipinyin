# 相关项目对标与借鉴方向

> 2026-10-25 调研。按「核心引擎 + 六个项目方向」组织，每条给出可执行的行动项。

## 核心引擎（输入法）

### rime-octagram 八股文（lotem/librime-octagram + rime-octagram-data）
Rime 的语法模型插件：`.gram` 二进制模型 + `grammar.yaml`，与词库**联合解码**。
- 印证：我们 `attach_lm` 的「切分图 × 语言模型重排」是同构架构，方向正确
- 可借鉴：① 他们提供**字模型/紧凑尺寸**多种发布变体 —— 我们的 `lm_char.marshal`
  可做裁剪/量化分级（3.7MB → 更小兜底版）；② 「训练自己的语言模型」指南把
  个人语料加权做进训练 —— 与 `make lm` 的个人语料路线一致，可参考其权重口径

### llm-ime（Deali-Axy/llm-ime）
本地 GGUF LLM（llama.cpp）做拼音引擎，实验性、Web 验证为主。
- 印证：chinese-ime-lm 的结论（4.25M 专用模型 52/52 完胜 Qwen2.5-0.5B 的 42/52）
  —— 通用 LLM 不是输入法的答案，**专用小模型**仍是正解
- 远期实验位：llama.cpp 跑小模型做下一词预测（简中 P2 的神经版），非当前优先

## 01 粤→普反向阅读助手

**语料直接扩容**（行动项，收益最确定）：
- [hon9kon9ize/yue-zh-translation](https://huggingface.co/datasets/indiejoseph/yue-zh-translation)：21.2 万句对
- msgchi（Kwok et al. 2023，[论文](https://dl.acm.org/doi/fullHtml/10.1145/3639233.3639332)）：13 万句对粤中书
- 现有 17 万 + 以上 ≈ 40~50 万句对，重训反向模型（训练只要 4 分钟），
  弱项「语气词/多义字、16+ 长句 17.7%」应能明显改善

## 02 批量转换 CLI

- [kobichiu/mandarin-to-cantonese-subtitle-mt](https://github.com/kobichiu/mandarin-to-cantonese-subtitle-mt)：80-90 年代港片字幕建的普→粤平行语料 —— 同为字幕场景，可借语料 + 对标
- HF Space [zh-yue-translator](https://huggingface.co/spaces/indiejoseph/zh-yue-translator)：现成对比产品
- 行动项：把长句守卫漏拦（`亡羊补牢` 10 字）的修复排上 —— 02 的 README 已记录

## 03 粤语 TTS 读音前端（两个大机会）

1. **Piper 还没有粤语 voice**（[issue #511](https://github.com/rhasspy/piper/issues/511) 开着）：
   我们的 G2P + [WenetSpeech-Yue](https://github.com/ASLP-lab/WenetSpeech-Yue)
   （**21,800 小时**开源粤语语音）+ Common Voice zh-HK = 训练第一个开源 Piper 粤语声。
   蓝海，且我们手里已有 G2P。
2. **[hon9kon9ize/yue-g2p-benchmark](https://github.com/orgs/hon9kon9ize/repositories)**：
   粤语 G2P 公开评测 —— 把我们的 G2P 上这个秤。我们的弱项（多音字取最常用读音、
   无消歧，`還` 表还/syun4 vs 還畀/waan4）预计会暴露，正好对着修。
3. [Cantoseg](https://github.com/ayaka14732/cantoseg)（pip 可装）：统计分词替代
   我们的「词库最长匹配」，词级注音精度应能提升。
4. espeak-ng 已有粤语 G2P 规则 —— 做对比基线。

## 04 中文地区用语转换

- OpenCC 词表是我们的底座（已用）；升级方向确认：**neural + 守卫解决一词多义**
  （程序/质量/土豆），16 句语境考卷就是回归集
- 语料：维基 zh-cn/zh-tw 双版本 + 社区维护的两岸词汇对照表 → 挖替换对 →
  复用 `tools/build_yue_guard.py` 产出 `tw_guard.marshal`

## 05 中文错别字纠正

- [pycorrector](https://github.com/shibing624/pycorrector)：该领域最大开源库。
  三条可抄：
  1. **混淆集数据**直接用（我们的 confusion.json 可与之合并去重）
  2. 评测口径对齐 SIGHAN（sentence/char 级我们已实现）
  3. [macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese)
     是"大模型路线"的现成 baseline —— 与我们「3~5ms 小模型」路线做对比实验
- [nghuyong/Chinese-text-correction-papers](https://github.com/nghuyong/Chinese-text-correction-papers)：
  CSC/CGEC 论文追踪清单
- EGCM（EMNLP 2022 Findings）：错误混淆集引导的纠错架构 —— 与我们守卫思路同源，
  可引用

## 06 公开基准包

- [hon9kon9ize/yue-g2p-benchmark]：G2P 评测对接（把 03 的 G2P 送测，顺手引流）
- [rime/rime-essay](https://github.com/rime/rime-essay)（LGPL-3.0）：Rime 官方词库
  + 语言模型，八股文数据的上游 —— 关注其许可以确定我们 .gram 类产物能否兼容

## 数据清单（重训弹药）

| 数据 | 规模 | 用途 |
|---|---|---|
| vendor/yue_pairs.json（现有） | 17 万句对 | 正/反向训练 |
| hon9kon9ize/yue-zh-translation | 21.2 万句对 | 合并重训 01/02 |
| msgchi | 13 万句对 | 同上 |
| WenetSpeech-Yue | 21,800 小时语音 | 03 的 TTS 训练 |
| Common Voice zh-HK | 若干小时 | 同上 |

## 优先行动（建议顺序）

1. **合并语料重训 01/02**（半天）：下载两份新语料 → 去重 → 正反向各重训 → 上基准
2. **03 的 G2P 上 yue-g2p-benchmark**（半天）：暴露多音字弱点，按分数修
3. **05 混淆集合并 pycorrector**（2 小时）：立刻提升统计基线的召回
4. **03 训 Piper 粤语声**（大项目，等 2 做完）：第一个开源粤语 Piper voice
