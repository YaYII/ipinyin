# 项目③：粤语 TTS 读音前端（G2P）

普通话文本 → 粤语书面文本 → 粤拼音节序列 → 音素序列。复用仓库已有的两块积木：

- ①普通话→书面粤语：`src/ipinyin/yue.py`（规则词表 + 语序规则，可选神经模型 `data/yue_nn.onnx` + 守卫 `src/ipinyin/yue_guard.py`）
- ②粤语字音/词音粤拼表：`data/yue.marshal`（`chars` 27,007 字、`words` 101,811 词，最长词 20 字）

两者拼起来就是一个可接开源 TTS（Piper/VITS）的粤语 G2P 前端。**训练声学模型属于下一步，不在本轮。**

## 管线图

```
普通话输入「2026年10月25日是星期日，我们一起去饮茶。」
   │
   │ ① normalize（g2p.py）
   │    全角→半角；数字→粤语读法（4位+年→逐位，月/日/点/其余整数→进位数量，
   │    单量词前 二→两）；标点归一到 , . ! ?；引号括号剥掉
   ▼
「二零二六年十月二十五日是星期日,我们一起去饮茶.」
   │
   │ ② 普→粤（YueConverter：yue_rules 语序 → 词表最长匹配；--nn 时 seq2seq + YueGuard）
   │ ③ 繁化（t2s.Converter.to_traditional）——粤拼表以繁体为键，
   │    不繁化则 迟/还/这/园 这类字整字缺读
   ▼
「二零二六年十月二十五日係星期日,我哋一齊去飲茶.」
   │
   │ ④ 粤拼切分（g2p.segment）：词级最长匹配优先 → 字级兜底 → 缺字报告
   ▼
ji6 lin4 ji6 liu1 nin4 sap6 jyut6 ji6 sap6 ng5 jat6 hai6 …
   │
   │ ⑤ 音素化（phonemes.py）：音节=声母+韵母+声调，映射 Piper 风格 IPA token，
   │    声调作为独立 token（T1..T6）
   ▼
j iː T6 l iːn T4 j iː T6 l iːu T1 n iːn T4 s ɐp̚ T6 j yːt̚ T6 …
```

## 用法

```bash
PY=/Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python

$PY projects/03-tts-g2p/g2p.py "今天天气很好，我们去公园散步。"
$PY projects/03-tts-g2p/g2p.py --file projects/03-tts-g2p/samples.txt
$PY projects/03-tts-g2p/g2p.py "..." --nn      # 神经转换（需 onnxruntime，口语词更地道）
$PY projects/03-tts-g2p/g2p.py "..." --json    # 对齐 token + 音素，机器可读

$PY projects/03-tts-g2p/phonemes.py --write    # 重新生成 phoneme_inventory.json
$PY projects/03-tts-g2p/coverage.py            # 覆盖率报告（默认 samples.txt）
$PY projects/03-tts-g2p/coverage.py --file 你的文本.txt

$PY -m pytest projects/03-tts-g2p/test_g2p.py  # 66 个确定性单测
```

## 音素清单（从 data/yue.marshal 全表实测提取）

| 类别 | 数量 | 内容 |
|---|---|---|
| 声母 | 19（另有零声母） | b p m f d t n l g k ng gw kw h w z c s j |
| 韵母 | 59 | aa 系 9 + a 系 9 + e 系 8 + i 系 8 + o 系 7 + u 系 6 + oe/eoi 系 7 + yu 系 3 + 成音节 m/ng |
| 声调 | 6 | Jyutping 1–6（表内没有把入声单列 7–9 变体） |
| 扫描音节 | 328,706 | 唯一坏音节 `jen1`（円，表值应为 jyun1，见下） |

拆分规则（确定性，无歧义，实现见 `phonemes.split_syllable`）：

1. 音节 = 声母 + 韵母 + 声调数字；去掉末位数字得 core。
2. core 本身是 `m`/`ng` → 成音节鼻音，无声母（ng5 ŋ̩、m4 m̩）。
3. 否则按声母表**从长到短**匹配前缀（gw/kw/ng 优先于 b/g/n），余下整体是韵母。Jyutping 韵母不以声母字母开头，贪心无歧义。
4. 易混对靠拼写本身区分：**aa 系是长 aː**（aam/aan/aang/aap/aat/aak），**a 系是短 ɐ**（am/an/ang/ap/at/ak）；`-ng` 结尾是鼻韵尾（韵母的一部分，ong6 = ∅+ong），与成音节 ng5（声母位为空）不同；入声尾 -p/-t/-k 是唯闭音，IPA 写 p̚ t̚ k̚。

Piper 占位方案：每个音节 → `[声母 IPA?, 韵母 IPA, T<调>]`（如 ngo5 → `ŋ ɔː T5`；零声母不发空 token）。完整符号表和 IPA 映射见 `phoneme_inventory.json`——训练时把这些 token 全部写进 `phoneme_id_map` 即可。

## 覆盖率实测（coverage.py）

| 口径 | 样例（12 句自拟） | 压力测试（小说随机 360 句） |
|---|---|---|
| 字级覆盖率（有粤拼的汉字 / 全部汉字） | **100.0%**（缺 0 字） | **100.00%**（缺 0 字） |
| 词级命中率（整词注音 token / 汉字 token） | 35.3%（词 47 / 字 86 / 缺 0） | 16.4%（词 1,329 / 字 6,788 / 缺 0） |
| 音节产出 / 汉字数 | 1.00（应≈1.00） | 1.000 |

字级 100% 对 TTS 可用性是硬指标：粤拼表 27,007 字（含 6,458 个扩展 A、1,981 个扩展 B 字）+ 管线里的繁化步骤，常规中文文本几乎没有缺读法的字。词级命中率低不是错误——切分器对未入词表的搭配逐字注音，音节数照样 1:1；它只说明词表还有扩容空间（整词读音才带词典里核过的多音字消歧）。

缺字报告的用法：`coverage.py` 会把缺读法的字按出现次数列出，按单补表即可。已知数据瑕疵 1 例：`円`（日元）表值 `jen1` 不是合法粤拼（应为 jyun1），管线会点名报告 `日円:jen1` 并在音素流里跳过该音节，不会悄悄吞字。

## 样例对照（原文 → 粤语 → 粤拼，规则版输出）

| 原文 | 粤语 | 粤拼 |
|---|---|---|
| 今天天气很好，我们去公园散步。 | 今日天氣好好,我哋去公園散步. | gam1 jat6 tin1 hei3 hou2 hou2 ngo5 dei6 heoi2 gung1 jyun2 saan3 bou6 |
| 我不知道他在说什么，请你再讲一次。 | 我唔知佢喺講乜嘢,請你再講一次. | ngo5 m4 zi1 heoi5 hai2 gong3 mat1 je5 ceng2 nei5 zoi3 gong2 jat1 ci3 |
| 明天上午九点钟开会，请不要迟到。 | 聽日上午九點鍾開會,請唔好遲到. | ting1 jat6 soeng6 ng5 gau2 dim2 zung1 hoi1 wui2 ceng2 m4 hou2 ci4 dou3 |
| 2026年10月25日是星期日，我们一起去饮茶。 | 二零二六年十月二十五日係星期日,我哋一齊去飲茶. | ji6 lin4 ji6 liu1 nin4 sap6 jyut6 ji6 sap6 ng5 jat6 hai6 sing1 kei4 jat6 ngo5 dei6 jat1 cai4 heoi3 jam2 caa4 |
| 给你一本书，看完之后还给我。 | 畀一本書,睇完之後你還畀我. | bei2 jat1 bun2 syu1 dai6 jyun4 zi1 hau6 nei5 syun4 bei2 ngo5 |
| 这条路好难走，我们坐车去吧。 | 這條路好難行,我哋坐車去吧. | ze2 tiu4 lou6 hou2 naan4 haang4 ngo5 dei6 co5 ce1 heoi2 baa1 |
| 他有三个哥哥和一个姐姐，都住在广州。 | 佢有三個哥哥同一個姐姐,都住喺廣州. | heoi5 jau5 saam1 go3 go1 go1 tung4 jat1 go3 ze2 ze2 dou1 zyu6 hai2 gwong2 zau1 |
| 机场快线每十五分钟一班，票价一百二十三元。 | 機場快線每十五分鐘一班,票價一百二十三元. | gei1 coeng4 faai3 sin3 mui5 sap6 ng5 fan1 zung1 jat1 baan1 piu3 gaa3 jat1 baak3 ji6 sap6 saam1 jyun4 |

（完整 12 句见 `samples.txt`，跑 `g2p.py --file samples.txt` 可复现，含音素流。）

## 如何接 Piper

本轮只做前端，语音模型训练是下一步。接法：

1. **语料准备**：拿粤语平行/口语文本（`vendor/yue_train.parquet` 可起步），过本管线的 ②③④⑤ 得到对齐的 (文本, IPA 音素序列) 对，标注 音频↔音素序列。
2. **符号表**：把 `phoneme_inventory.json` 里的 `ipa_initial` ∪ `ipa_final` ∪ `T1..T6` ∪ 停顿 `, . ! ?` 全部写进 Piper 的 `phoneme_id_map`。规模约 19+59+6+4 ≈ 88 个符号，很小，不需要 BPE。
3. **声调**：当前占位方案是每个音节后跟离散 token `T1..T6`（对齐 VITS/piper 的音素序列习惯）。等真训模型时可二选一：保留 T-token（简单、可控），或换成 Chao 声调字母进韵母 token（更细，先不做）。
4. **训练**：`piper` 仓库标准流程（配置里 `phoneme_type: TEXT` / 自定义 symbol 集），单说话人 1–2 小时语料起步。粤语底模可参考开源 `Cantonese Piper voices`（jyutping 训练）做迁移。
5. **推理端**：`g2p.py --json` 的 `phonemes` 字段就是喂给 onnx 声学模型的输入序列。

## 已知局限

- **多音字**：字表每字只存一个最常用读音，无上下文消歧。典型：`還` 表值 `syun4`（還係），但「還畀我」里应读 `waan4`；「還錢」同。整词命中时用词表读音（准确），逐字兜底时可能选错。缓解：扩 `words` 词表（当前 10.2 万词，词级命中 16–35%）；根本解法是给字级注音加上下文模型（下一步）。
- **变调未处理**：粤语声调丰富（如「今日」gam1 jat6 实读 gam1 jat6 但 双音节词常有变调：橙 caang4→caang2、围 wai4→wai2 之类的小称/口语变调），当前一律按字典调输出。TTS 听感会偏「字正腔圆」但不影响可懂度。
- **数字**：已覆盖 整数年份（逐位）、月/日/钟点/普通整数/千分位（进位数量）、单量词前二→两。**TODO**：小数（3.5）、百分数（50%）、电话/编号（11 位以上现在保守地保持原样）、负数、分数、序数专读（第X 已随数量读法顺带工作）。
- **拉丁字母**：`OK`、`App` 等原样保留不注音（token 类型 latin），TODO：拼读或词典映射。
- **数据瑕疵**：`円→jen1` 一处非法音节（2 个词条），管线会报告并跳过，可顺手修表。
- **语速/停顿**：标点归一为 `, . ! ?` 四档停顿 token，粒度粗（逗号内小停顿未分）。
- **神经模式**：`--nn` 依赖 onnxruntime + `data/yue_nn.onnx`；加载失败会警告并退回规则版，不影响确定性测试。
