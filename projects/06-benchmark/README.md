# 项目⑥：公开基准包（benchmark suite）

把仓库三套自建评测口径打包成**可对外发布的基准套件**：统一入口、统一 JSON
输出、统一方法学。三个 suite：

| suite | 考什么 | 口径来源（题面/判分直接复用，不另起炉灶） |
|---|---|---|
| `yue-lengths` | 粤语转换按真实打字长度分桶：2字/3字/4字成语/5-7字/5-7字古诗，带「照抄不改」对照列与打字分布加权 | `tools/eval_ime_lengths.py` |
| `yue-guard` | 守卫决策：规则 vs 神经裸输出 vs 神经+守卫；救回/弄丢分开报 + 守卫样例 | `experiments/exp_yue_guard.py` |
| `zh-sentence` | 简中整句：自洽题面、Top1/前9/平均键数；`--lm` 加跑重排对照（救回/弄坏分开报） | `tools/eval_sentence.py`（+ `tools/eval_personal_docs.py` 的方法论） |

```
run_all.py          统一入口：--suite {all,yue-lengths,yue-guard,zh-sentence} --quick
suite_*.py          三个套件模块（题面构造 import 自 tools/，口径零漂移）
bench_common.py     仓库路径、环境信息、神经模型加载、Wilson 区间、文件/题面指纹
schema_check.py     结果 JSON 校验器（schemas.md 的可执行形态）
export_questions.py 导出题面包（答案不随包，C-Eval 模式）
score_submission.py 评分外部提交 predictions.jsonl → 标准结果信封
compare.py          两系统配对显著性检验（McNemar 精确检验，同题同序强制）
leaderboard.py      从 results/ 汇总 LEADERBOARD.md（quick 表不具排名资格）
ci-benchmark.yml    GitHub Actions 示例（复制到 .github/workflows/ 启用）
schemas.md          JSON 格式规范 + 版本化规则（改口径必须升 suite_version）+ 版本记录
METHODOLOGY.md      方法学：分桶、照抄列、救回/弄坏、自洽乐观偏差、门控二分、
                    防作弊边界、样本量与功效、统计工具
test_runner.py      pytest：schema 校验、--quick 可跑通、v1.1.0 新字段、
                    导出→评分往返、McNemar 数学（无 torch 依赖处用桩模型）
results/            实测产出（--quick 实跑结果 + 外部提交评分样例）
```

## 快速开始

```bash
PY=/Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python
cd projects/06-benchmark

$PY run_all.py --suite all --quick            # 全部三套件，每桶 ≤50 题，~9s
$PY run_all.py --suite zh-sentence --quick --lm   # 整句 + 重排对照（救回/弄坏）
$PY compare.py results/yue-lengths__full.json results/yue-lengths__full.json
                                              # 两系统配对 McNemar（总分看着差 24pp、
                                              #   全量 n=377 p<0.0001 → 可分辨）
$PY export_questions.py --suite all --mode quick  # 导出题面包给外部系统作答
$PY leaderboard.py                            # results/ → LEADERBOARD.md
$PY -m pytest test_runner.py -v               # 套件自测（25 个）
```

数据前置（缺什么套件就 graceful skip，`status: "skipped"`，不算失败）：

| 依赖 | 生成方式 | 谁需要 |
|---|---|---|
| `vendor/yue_pairs.json` | `make corpus` | 全部三个 |
| `data/lexicon.tsv` 等 marshal | `make lexicon` / 入库 | zh-sentence |
| `experiments/yue_model_d256.pt` | `make train-yue` | yue-lengths 的神经列、yue-guard |
| `data/yue_guard.eval.marshal` | `python3 tools/build_yue_guard.py`（默认 eval 切分，不入库） | 仅 yue-guard |
| `data/lm_char.marshal` | `make lm`（可选） | 仅 zh-sentence 的 `--lm` |

## --quick 实测（2026-10-26，v1.3.0 口径，M 系列 macOS，Python 3.13.12 / torch 2.14.1）

总耗时 **~13s**（yue-lengths 4.7s + yue-guard 6.8s + zh-sentence 1.6s，含模型加载）。
所有准确率数字逐位可复现（种子固定），JSON 里每个输入文件带 sha256 指纹。

```
[ok] yue-lengths v1.3.0 (3.6s)
  2字         n=10   照抄=0.0%   规则=40.0% 神经=100.0%  ← 真词桶+合法变体判分（§1）
  4字成语      n=40   照抄=100.0% 规则=97.5% 神经=82.5%   ← 神经把成语改坏，照抄列钉死上限
  5-7字       n=50   照抄=14.0%  规则=36.0% 神经=56.0%
  5-7字古诗    n=12   照抄=100.0% 规则=75.0% 神经=58.3%
[ok] yue-guard v1.3.0 (5.1s)   ctx_max=5 + 典籍白名单
  2字         n=10   规则=40.0% 神经=100.0% +守卫=100.0% 否决=0  救回=0  弄丢=0
  4字成语      n=40   规则=97.5% 神经=82.5%  +守卫=100.0% 否决=7  救回=6  弄丢=0
  5-7字       n=50   规则=36.0% 神经=56.0%  +守卫=54.0%  否决=4  救回=0  弄丢=0
  5-7字古诗    n=12   规则=75.0% 神经=58.3%  +守卫=100.0% 否决=6  救回=2  弄丢=0
  8-15字      n=50   规则=16.0% 神经=42.0%  +守卫=42.0%  否决=1  救回=0  弄丢=0
[ok] zh-sentence v1.1.0 (1.6s)  含 --lm 重排对照（λ=0.6）
  2-4字     自洽题=23 Top1=100.0% 前9=100.0% 平均键数=10.9 救回=1 弄坏=0
  5-7字     自洽题=20 Top1=85.0%  前9=95.0%  平均键数=17.9 救回=2 弄坏=1
  8-15字    自洽题=10 Top1=80.0%  前9=80.0%  平均键数=36.5 救回=1 弄坏=0
  16+字     自洽题=7  Top1=57.1%  前9=57.1%  平均键数=51.7 救回=0 弄坏=0
  重排对照：不重排 81.7% → 重排 86.7%；救回 4 / 弄坏 1 / 净 +3
```

（zh-sentence 归档为 `--lm` 版本：Top1 列即重排后的生产配置口径，不重排的
原始口径在 `rerank.top1_no_lm`；quick 与 full 均同此。）

## 全量实测（2026-10-26，v1.3.0 口径，对外基线，`results/*__full.json`）

全量三套件合计 **~36s**（yue-lengths 10.9s + yue-guard 23.0s + zh-sentence 2.4s）。
题量：yue-lengths 377 / yue-guard 677 / zh-sentence 256（2 字桶经 v1.2.0 净化
后全真词、v1.3.0 起接受合法变体，见 METHODOLOGY §1）。**对外报数以此为准。**

```
[ok] yue-lengths v1.3.0 (8.7s)
  2字         n=25   照抄=0.0%   规则=36.0% 神经=100.0% 95%CI[86.7,100]
  4字成语      n=40   照抄=100.0% 规则=97.5% 神经=82.5%
  5-7字       n=300  照抄=16.7%  规则=31.7% 神经=45.3%
  5-7字古诗    n=12   照抄=100.0% 规则=75.0% 神经=58.3%
  加权（覆盖 80%）：规则 54.4% → 神经 84.3%
[ok] yue-guard v1.3.0 (18.0s)   ctx_max=5 + 典籍白名单（成语/古诗 → 繁体外壳）
  2字         n=25   神经=100.0% +守卫=100.0% 救回=0   弄丢=0
  4字成语      n=40   神经=82.5% +守卫=100.0% 救回=6   弄丢=0
  5-7字       n=300  神经=45.3% +守卫=43.7%  救回=1   弄丢=5
  5-7字古诗    n=12   神经=58.3% +守卫=100.0% 救回=2   弄丢=0
  8-15字      n=300  神经=46.7% +守卫=46.3%  救回=0   弄丢=2
  生产路径加权（2字.4/成语.25/5-7字.15，覆盖80%）：守卫后 89.4%
[ok] zh-sentence v1.1.0 (2.4s)  含 --lm 重排对照（λ=0.6，v1.3 扫描甜点）
  2-4字     自洽题=30  Top1=93.3% 前9=100.0%
  5-7字     自洽题=134 Top1=88.8% 前9=97.8%
  8-15字    自洽题=64  Top1=89.1% 前9=95.3%
  16+字     自洽题=28  Top1=32.1% 前9=32.1%
  重排对照：不重排 77.3% → 重排 85.5%；救回 23 / 弄坏 2 / 净 +21（λ=0.4 时为 83.2%）
```

**这轮基准真正干了活**（METHODOLOGY §12 全程记录）：裁决六次升级——
两条调参修复（守卫 ctx_max 7→5；重排 λ 0.4→0.6：zh Top1 +2.3pp）、一条
原则性守卫修复（简→繁外壳免实证 + 典籍白名单：成语/古诗守卫后 100%）、
三个模型否决/打平后第四个破局晋升生产（**真词增广 8 轮**，裸神经
McNemar p=0.0026、全桶无回退）：粤语加权 80.3%→84.3%，守卫后生产路径
87.4%→**89.4%**，2字真词 100%、成语/古诗守卫后 100%。

### 实际 JSON 样例（`results/yue-lengths__quick.json`，有删节）

```json
{
  "suite": "yue-lengths",
  "suite_version": "1.1.0",
  "spec": "projects/06-benchmark/schemas.md (envelope v1)",
  "system": "ipinyin(内置)",
  "status": "ok",
  "quick": true,
  "generated_at": "2026-10-25T21:55:39+08:00",
  "duration_s": 5.34,
  "params": {"quick": true, "nn": "auto", "ctx_max": 7, "device": "cpu", "...": "..."},
  "environment": {"python": "3.13.12", "platform": "macOS-26.6.2-arm64-arm-64bit-Mach-O", "torch": "2.14.1"},
  "reason": null,
  "results": {
    "buckets": {
      "2字": {"n": 25, "copy_baseline": 0.0, "rule": 0.36, "neural": 0.96,
               "neural_median_latency_ms": 10.3745, "weight": 0.4,
               "wilson": {"copy_baseline": {"lo": 0.0, "hi": 0.1332},
                          "rule": {"lo": 0.2025, "hi": 0.5548},
                          "neural": {"lo": 0.8046, "hi": 0.9929}}}
    },
    "weighted": {"weighted_rule": 0.5434, "weighted_neural": 0.7816,
                 "weight_coverage": 0.8},
    "empty_buckets": ["3字"],
    "neural_note": null,
    "inputs": [
      {"path": "vendor/yue_pairs.json", "sha256_16": "89e2056ffc6f76bc", "bytes": 18995253},
      {"path": "experiments/yue_model_d256.pt", "sha256_16": "70470551515614d7", "bytes": 35959386},
      {"path": "tools/idioms.txt", "sha256_16": "913582d611741995", "bytes": 727}
    ],
    "question_set_hash": "edd8159f7d56f588",
    "per_item": [
      {"id": "2字:0", "bucket": "2字", "input": "很不", "expected": "好唔",
       "copy_ok": false, "rule_out": "好不", "rule_ok": false,
       "neural_out": "好唔", "neural_ok": true, "latency_ms": 248.182}
    ]
  }
}
```

提交评分样例：`results/yue-lengths__full__规则转换.json`（v1.1.0 口径存档）——
把内部规则转换当外部提交走了一遍完整闭环（题面导出 → 作答 → 评分），其
neural 列与内部 rule 列逐桶一致，即提交评分与内部评测同一判分口径的活证据。

守卫样例（`results/yue-guard__quick.json` 的 `results.examples.4字成语[0]`，
守卫存在的理由一行看懂）：

```json
{"input": "亡羊补牢", "neural": "亡羊補監", "guarded": "亡羊补牢",
 "expected": "亡羊補牢", "outcome": "saved"}
```

graceful skip 的输出（如模型缺失）：

```json
{"suite": "yue-guard", "status": "skipped", "results": null,
 "reason": "模型文件不存在：…/yue_model_d256.pt", "...": "..."}
```

字段定义与版本化规则见 **schemas.md**；数字怎么读、能宣称什么见
**METHODOLOGY.md**（一句话版：`--quick` 只能冒烟，60~300 条分辨不了 1~3pp；
两系统同桶 Wilson 区间不重叠才值得跑配对检验，p<0.05（全量）才算可分辨）。

## 提交新系统进基准

两条路：**文件提交**（推荐，外部系统无需接触本仓库代码）与**函数注入**
（同仓库开发时方便）。

### 路线 A：文件提交（C-Eval 模式）

```bash
# ① 仓库侧导出题面（答案不随包分发）
$PY export_questions.py --suite yue-lengths --mode full
#    → questions/yue-lengths__full.questions.jsonl + .manifest.json

# ② 外部系统对题面作答：每行 {"id": "4字成语:3", "output": "亡羊補牢"}
#    （zh-sentence 则为 {"id": "5-7字:2", "candidates": ["今日天氣好", …]}）

# ③ 仓库侧评分（重建答案 + 双哈希防调包 + 同一套判分逻辑）
$PY score_submission.py --suite yue-lengths --mode full \
    --system MyModel --predictions mypreds.jsonl
#    → results/yue-lengths__full__MyModel.json，直接进 leaderboard
$PY leaderboard.py
```

提交 PR 必附：

1. `predictions.jsonl` + 你的系统名（`--system`）与一句话架构说明；
2. 声明训练语料是否与 `vendor/yue_pairs.json` 重叠（重叠 = 分数虚高，见
   METHODOLOGY §7；我们的语料开源，这靠自觉，防不住有意过拟合——§10 写明
   了防线边界）；
3. 全量口径（`--mode full`）——quick 提交只做管线自测；
4. 声明单题延迟（yue 套件内部评测自动统计中位数；文件提交需自带）。

zh-sentence 的题面是拼音键串，yue 的是简体原文；判分统一在仓库侧完成
（yue = 繁体外壳后比对；zh = 首选/前9），yue-guard 提交还会得到我们守卫
加持后的 `guarded`/`saved`/`lost` 对照列。

### 路线 B：函数注入（同仓库开发）

```python
# my_system.py —— 粤语套件：简体输入 → 繁体粤语输出
def convert(text: str) -> str:
    return my_model.translate(text)          # 贪心/beam 均可，耗时会被单独统计

from suite_yue_lengths import run
status, results, reason = run({"quick": False, "_nn_fn": convert})

# zh-sentence：拼音键串 → 候选词列表
def candidates(key: str) -> list[str]:
    return my_engine.predict(key)[:9]        # ["今天天气", "今天", ...]

from suite_zh_sentence import run
status, results, reason = run({"quick": False, "candidate_fn": candidates,
                               "lm": "data/lm_char.marshal", "lm_weight": 0.4})
```

注入路线请同时在结果 JSON 的 `system` 字段写明系统名（`run_all.py --system`）。

## 发布流程

本目录当前状态：**v1.1.0，quick + 全量均已实跑归档，CI 已启用，可对外发布**。

发布一次「基准结果」的固定动作（口径没变时）：

1. `run_all.py --suite all`（全量）+ `leaderboard.py`，确认
   `results/*__full.json` 与 `LEADERBOARD.md` 是本次归档；
2. 报数引用格式：`suite@版本 + 模式 + n`，例：
   `yue-lengths@1.1.0 full n=617 加权 75.4%；2字桶 90.6% [95%CI 86.4, 93.5]`
   （区间取自 results/ 的桶级 `wilson` 字段）；
3. `git commit`（results/、questions/、LEADERBOARD.md 一并入库，历史可溯）。

发布一次「口径变更」的固定动作（改了桶/指标/权重/默认参数时）：

1. 先跑旧口径存档（全量），再改代码；
2. 升 `suite_version`（schemas.md 的规则：字段语义变 major、加字段 minor、
   修 bug patch + 前后对照数字），在 commit message 写明变更；
3. 重跑全量归档，`LEADERBOARD.md` 里旧结果随旧版本文件保留，不覆盖。

接受外部提交（已验证的流程）：

1. 贡献者拿 `questions/<suite>__full.questions.jsonl` + `.manifest.json` 作答；
2. PR 附 `predictions.jsonl` + 系统名 + 语料重叠声明（README「提交新系统」清单）；
3. 维护者本地 `score_submission.py` 评分（双哈希校验自动执行）→
   `leaderboard.py` 入榜 → 合入时全量 JSON 与逐题记录一起入库。

## CI 集成

`ci-benchmark.yml` 是可跑的 GitHub Actions 模板，复制到仓库根
`.github/workflows/benchmark.yml` 即生效。三个 job：

- **selftest**：无数据环境跑 `test_runner.py`——schema/版本化/McNemar 数学
  等测试永远可跑，依赖数据的测试自动 skip（graceful skip 本身就是被测行为）；
- **smoke**：`run_all --suite all --quick --nn off`，无语料时全 suite
  skipped、退出码 0，结果 JSON 存 artifact——管线坏没坏的冒烟门禁；
- **quick-matrix**（手动触发）：装 CPU torch + 语料后真跑 quick，数字存档。

## 方法学与借鉴来源

本套件的方法学不是闭门造车，三处显式借鉴（详见 METHODOLOGY 各节）：

- [`metasequoiaime/chinese-ime-lm`](https://github.com/metasequoiaime/chinese-ime-lm)
  —— 口径来源与纪律来源：对数概率按字数归一（§5）、「每条结论带数字和
  样本量」、「评测能用的门控运行时用不了」（§9 门控二分）；
- [C-Eval](https://github.com/hkust-nlp/ceval) —— 提交机制：题面分发、
  答案不随包、仓库侧评分（§10，含我们防线的诚实边界）；
- [EleutherAI/lm-evaluation-harness](https://github.com/eleutherai/lm-evaluation-harness)
  —— 可复现纪律：数据指纹钉死进结果（`inputs` 字段）、任务版本化
  （`suite_version` + schemas.md 版本记录表）。

## 边界

- 本目录只**引用**大文件（`../../vendor/`、`../../data/`、`../../tools/`），
  不复制；`vendor/` 不入库，clone 后先 `make corpus`。
- 不修改 `src/`、`experiments/`、`tools/`、Makefile——口径升级只动本目录
  并升 `suite_version`。
- `test_runner.py` 在数据缺失的机器上会逐测试 skip（schema、版本化纪律、
  统计工具的测试永远可跑），全量通过环境见上方实测段。
- `results/` 里的 JSON 是最新一次实跑的存档（同模式覆盖写，历史用 git
  追溯）；`questions/` 与 `LEADERBOARD.md` 为生成物。
