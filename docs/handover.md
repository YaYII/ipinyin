# 交接手册

> 给接手开发/完善本项目的人。读完应知道：代码在哪、怎么跑起来、怎么验证改动、
> 怎么训练和发布模型、哪些坑前人已经替你踩过。
> PR 流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)；本文是它的完整版。
> 深挖优先级和六个衍生项目的成绩表见 [projects/README.md](../projects/README.md)。

## 〇、30 秒了解现状（2026-10-26）

一个自己写的 **Linux 拼音输入法（IBus 引擎）**：整句切分 + 前缀补全 + 个人化；
候选四列 `简体/繁体/粤语/粤拼`，Tab 切换。粤语链路是本项目的技术核心：
**神经模型（ONNX int8，随仓库分发）+ 实证守卫（YueGuard）**。

当前生产状态（held-out 实测，判据见第四节）：

| 项 | 数字 |
| --- | --- |
| 粤语 2 字词（真词口径 n=25） | 神经+守卫 96.0% |
| 粤语 4 字成语 / 5-7 字古诗（答案=不改） | **100% / 100%**（典籍白名单） |
| 粤语 5-7 字口语 / 8-15 字句 | 42.0% / 46.7%（整句口径） |
| 粤语加权（按打字分布） | 守卫后生产路径 89.4% |
| 简中整句重排（λ=0.6 上下文 LM） | Top1 85.5% |
| 测试 | 引擎核心 72 + 六项目 180 全绿 |

生产模型是「真词增广 + 8 轮」版（2026-10-26 晋升并重导出 ONNX）。
最大开放问题：P4 编辑式模型、ONNX 变长导出、外部大语料（许可证待验证）、
守卫 5-7 字桶按桶调参。详见第九节。

## 一、仓库地图

| 位置 | 内容 |
| --- | --- |
| `src/ipinyin/` | 引擎核心。`ime.py` 纯状态机（不依赖 IBus，可单测）、`engine.py` 只做适配、`segmenter.py`/`lexicon.py`/`lm.py` 简中链路、`yue.py`/`yue_rules.py` 规则转换、`yue_nn.py` 神经模型加载与推理、`yue_guard.py` 实证守卫、`t2s.py` 简繁、`user_model.py` 个人化 |
| `data/` | 词表与模型，**全部入库**（clone 即用）：`lexicon.tsv`、`s2t.marshal`、`yue.tsv`/`yue.marshal`/`yue_phrase.marshal`（规则三表）、`yue_guard.marshal`（守卫实证）、`yue_nn.onnx`+`yue_nn_vocab.json`（生产神经模型）、`yue_word_pairs.json`（真词增广语料） |
| `experiments/` | 训练与导出脚本、模型权重（**入库**）、训练日志。`train_yue_seq2seq.py`、`export_onnx.py`、`build_word_pairs.py` |
| `tools/` | 构建/评测/开发脚本。`dev_macos.sh`（Mac 一键环境）、`eval_*.py`（各口径评测）、`build_yue_guard.py`、`gh_push.py`（备用推送） |
| `tests/` | 引擎核心单测（`make test`）；`e2e_ibus.py` 要真 IBus 总线 |
| `projects/` | 六个衍生项目基线（01 粤→普反向、02 批量转换 CLI、03 TTS 前端、04 地区用语、05 错别字、06 公开基准包），各有 README/测试/评测 |
| `docs/` | `nn-upgrade.md`（粤语神经：原理+实测+路线 P0-P4）、`nn-chinese.md`（简中链路）、`train-yue.md`（训练步骤）、`macos.md`、`personalization.md`、`related-work.md` |
| `vendor/` | 平行语料（`yue_pairs.json` 等） |

接手顺序建议：`README.md` → 本手册 → `docs/nn-upgrade.md` → 你要动的模块。

## 二、环境与跑起来

**Linux（部署目标）**：

```bash
./install.sh                 # 装到 ~/.local/share/ipinyin，含神经模型
# 神经模式可选依赖（缺了会明示提示，不静默降级）：
pip3 install --user onnxruntime && ./install.sh && ibus restart
```

**Mac（开发/训练）**：`./tools/dev_macos.sh --with-nn` 一键铺环境（torch/onnx 全家桶）。
Mac 上做不了：`make install`（IBus）、`make e2e`（要 ibus daemon）、`build_deb.sh`。
**分工：Mac 开发和训练，Linux 装和测真机。**

**测试**：

```bash
make test            # 引擎核心（72 个）
make projects-test   # 六个项目（分进程跑：02/04 各有 convert.py 会撞模块名）
# pytest 直跑注意：沙箱环境必须 --basetemp=/tmp/xxx，否则 tmp 目录权限报错
```

常用 make 目标：`lexicon`（编译词库）、`lm`（语言模型）、`train-yue`、
`eval-yue` / `eval-yue-guard` / `eval-sentence` / `eval-docs` / `eval-coverage`
（各口径评测，见下节）、`yue-guard`（重建守卫数据）、`sogou-dict`（词库扩容）、
`macos-testpad`（Mac 打字测试板）、`push`（备用推送）。

## 三、引擎管线（改代码前必读）

**简中链路**：`segmenter.py` 一元 LM + DP 把拼音流切词 → 词库先验排序 →
上下文 LM 重排（`λ=0.6`：词库分 + λ·上下文分，全量扫描 0.3~1.0 的甜点，
Top1 83.2→85.5%）→ 候选四列。领域语料可单独调 λ（`attachLM(lm, weight)`）。

**粤语链路三件套**（`YueConverter.convert`，neural=True 时）：

1. **NeuralYue**（`yue_nn.py`）：ONNX 贪心解码。输入宽度**固化在 40 token**
   （旧导出器的 Reshape 形状，减 SOS/EOS 实际 38 字上限——超长输入必须先分句，
   见 projects/02 的 `sentences.py`）；
2. **YueGuard**（`yue_guard.py`）：神经输出对齐后逐段验证「改动有实证」——
   词级替换对 / 单字+邻接语境（仅短输入）/ 简繁外壳豁免 / 典籍白名单（成语/古诗
   直接给繁体外壳）/ 长句宽松路径（目标是语料见过的粤语 n-gram 即可）。无实证的
   片段块级回退规则版。原理与实测见 `docs/nn-upgrade.md` §三；
3. **规则版**（`yue.py` + `yue_rules.py`）：语序规则 → 统计短语表 → 人工词表，
   永远是兜底。

**性能红线**：候选列表逐个调 `convert` 会把单键延迟从 3ms 拖到 60ms+——
神经推理**只对首选候选**做（调用点在 `ime.py`），候选重建走规则；整句缓存 256 条。

**ctx_max 的口径差异**（易踩）：生产 `YueGuard` 类默认 `ctx_max_len=5`，
`experiments/exp_yue_guard.py` 评测口径 7，projects/02 批量工具显式 7。
**改任何默认值要全仓 grep 旧值**——2026-10-26 自审抓过 4 处漏同步（λ 和 ctx_max 都中过招）。

## 四、评测与口径纪律（本项目的方法论核心）

| 口径 | 命令 | 用途 / 注意 |
| --- | --- | --- |
| 简中整句 | `make eval-sentence` / `eval-docs` | **8 字封顶**（用户定的口径，别改） |
| 粤语分桶 | `make eval-yue-guard` | 2字/成语/古诗/5-7字/8-15字 + 照抄基线列 |
| 补词效果 | `make eval-coverage` | 自洽整句口径量不到缺词，看逐词覆盖 |
| 公开基准 | `projects/06-benchmark/run_all.py` | 三套件 + 题包/提交闭环 + 归档 + CI（`.github/workflows/benchmark.yml`） |

纪律（每条都有血泪史）：

1. **报数必须与 results/ 归档同源**——README 曾残留旧跑数字；
2. **分桶看，别只看整句**：句子级验证看不见分桶回退（P3 判决链的由来：裸加轮数
   整句大涨、2字桶 90.6→84.5，差点晋升）；
3. **每个新口径对照「照抄不改」基线**，识别靠不改骗的分；
4. **合法变体判分**（06 v1.3）：什么→乜嘢/咩 都对，单一标准答案会冤枉；
5. **改口径必须升 `suite_version`**（语义变 major / 加字段 minor），旧结果存档
   `results/archive-*/`；
6. 统计显著性用 McNemar（`projects/06-benchmark/compare.py`），报数带 n 和 p 值。

## 五、模型迭代工作流（train → 判决 → export → ship）

1. **训练**：`docs/train-yue.md`；现生产 =「真词增广 ×4 + 8 轮」
   （复现命令在 `docs/nn-upgrade.md` P3 节）。改数据先想清楚分布——训练语料
   偏句子（≤4 字句对仅 0.3%），加轮数前先看 2 字桶；
2. **判决**：分桶基准 + McNemar，**全桶无回退**才算赢；候选模型留
   `experiments/`（如 `yue_model_d256_e12.pt` 落选版）可复现对照；
3. **导出（清单项，不是可选）**——量化产物只落到 `experiments/`，生产文件要手动覆盖
   （2026-10-26 踩过：晋升了 torch 权重忘了下面后两行，生产 ONNX 挂旧模型跑了一天）：

   ```bash
   python3 experiments/export_onnx.py experiments/yue_model_d256.pt experiments/  # fp32+vocab
   python3 tools/quantize_yue_nn.py                                              # → experiments/yue_nn_int8.onnx
   cp experiments/yue_nn_int8.onnx data/yue_nn.onnx
   cp experiments/yue_nn_vocab.json data/yue_nn_vocab.json
   ```

   仓库约定训练产物全入库（experiments/ 的 fp32/int8 副本一并 commit）；
4. **验证**：`make test` + `make projects-test`；生产路径四口径用
   `projects/02-batch-convert/eval.py`（照抄/规则/整句36/单位7 同台）；
5. **发布**：commit + `git push`。

**Linux 装机升级**（模型与守卫是配套验证的组合，三个文件一起换）：

```bash
cp data/yue_nn.onnx data/yue_nn_vocab.json data/yue_guard.marshal \
   ~/.local/share/ipinyin/data/ && ibus restart
```

## 六、六个衍生项目

| # | 目录 | 一句话 | 基线 |
| --- | --- | --- | --- |
| 01 | `projects/01-yue2zh/` | 粤语→普通话反向 seq2seq（含抗错字版） | 整句 30.0%；抗噪卷 14.6% vs 8.2% |
| 02 | `projects/02-batch-convert/` | 文章/SRT/VTT 批量转粤语 CLI（守卫+术语表+QC+可疑句排序） | 成语/古诗 100%、8-15字 46.7% |
| 03 | `projects/03-tts-g2p/` | 粤语 TTS 读音前端（粤拼→IPA，接 Piper） | 加权 74.7% |
| 04 | `projects/04-locale-convert/` | zh-CN↔zh-TW 用词转换 + 歧义报告 | 整词 83.4% |
| 05 | `projects/05-typo-correct/` | 错别字纠正（SIGHAN 风格评测） | 纠正 F1 0.637 |
| 06 | `projects/06-benchmark/` | 公开基准包：三套件统一入口 + 提交闭环 + 排行榜 | `--quick` 9 秒 |

深挖优先级建议在 `projects/README.md` 末尾。各项目自带 README/测试/评测与「下一步」清单。

## 七、协作与发布约定

- **commit 风格**：`type(scope): 中文摘要`；body 列要点，实测数字带口径/n/p 值；
- **入库 / 不入库**：训练产物（模型权重、日志、小评测集）入库——开源不隐瞒；
  可再生的大产物不入库（01 的 merged_pairs 27MB、05 的 train_typos 2 万句等，
  见各 `.gitignore`）；
- **推送**：`git push origin master` 直连一般可用（2026-10-25 起恢复）；
  `make push` 是备用路径，注意两个坑：它**从工作区读文件内容**（只适合推刚 commit
  的 HEAD，推积压历史提交会把工作区内容错灌进老提交）且一次只推一个提交；
- **会话交接**：本机的会话日志在 `.workbuddy/memory/<日期>.md`；
- 数据/隐私：个人化数据都在用户本机 `~/.local/share/ipinyin/`，不上传。

## 八、踩坑速查（前人替你踩过）

1. pytest 在沙箱报 tmp 权限错 → `--basetemp=/tmp/xxx`；
2. shell 的工作目录会停在上一条 `cd` 的地方 → 相对路径操作前先确认 cwd（曾在
   06 目录里误建 `.github/`）；
3. `git reset --hard` 会冲掉未提交的工具修复 → reset 前先 `git status` 看清；
4. gh_push 推中文路径文件时 `ls-tree` 因 quotepath 落空（已修，`diff-tree` 加了
   `-c core.quotepath=false`）；
5. **macOS 激活 = 公证门槛**（TISRegisterInputSource 对未公证 app 静默拒绝，
   全试过无效）：短期用 `make macos-testpad`，长期要 $99/年开发者计划 +
   `xcrun notarytool`；
6. **别手写 com.apple.HIToolbox 启用条目**——指向未注册输入源的条目会让系统
   「键盘」面板崩溃（已清理过一次）；
7. IMK Swift 坑：informal 协议方法要 `@objc(选择器)` 暴露；`IMKServer` 必须
   `controllerClass:delegateClass:` 全传；
8. ONNX 图 Reshape 固化 width=40：变长输入报 "cannot reshape"，推理侧按 40
   补齐、提前 EOS 停；根治要 dynamo 导出器或 P4 编辑式模型；
9. `user_model.observe` 曾有内存泄漏（按键期建 cell）——已修，别把这类模式引回来；
10. 评测前清神经缓存再计时，否则先跑的模式把缓存焐热、后跑的延迟失真。

## 九、下一步（接手从这里挑）

按价值排序，明细见各文档：

1. **P4 编辑式模型**（`docs/nn-upgrade.md` §四；GECToR/FCGEC 已验证范式）：
   token 级「保留/替换」标注，一次前向 44ms→3~5ms，天然只改该改的位置，
   根治守卫保守性与 02 的切片成本；
2. **外部大语料重训**：raptorkwok 13 万对 + hon9kon9ize 38k（**先验证许可证**），
   按第五节工作流走判决链；
3. **P2 候选级融合**：规则版与神经版分歧时给第二候选（Tab 已有切换机制）；
4. **02 批量工具**：回译打分（联动 01 反向模型，`--flag` 升级）、CanCLID 词库
   四件套接入、更聪明的切分；
5. **守卫按桶调参**：5-7 字口语桶全量下净弄丢 23 条（06 全量归档的发现），
   ctx_max/修复策略按桶分；
6. **04 一词多义神经化**（16 句语境考卷已备好）、**05 P4 编辑式上秤**（目标
   F1>0.64）、**03 接 Piper 训声**、**06 外部提交注册表**。

---

*维护约定：手册跟着仓库状态走——大改动（模型晋升、口径变化、新项目）落地时
同步更新第〇节现状表和第九节；踩了新坑往第八节加一行。*
