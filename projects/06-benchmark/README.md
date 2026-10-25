# 项目⑥：公开基准包（benchmark suite）

把仓库三套自建评测口径打包成**可对外发布的基准套件**：统一入口、统一 JSON
输出、统一方法学。三个 suite：

| suite | 考什么 | 口径来源（题面/判分直接复用，不另起炉灶） |
|---|---|---|
| `yue-lengths` | 粤语转换按真实打字长度分桶：2字/3字/4字成语/5-7字/5-7字古诗，带「照抄不改」对照列与打字分布加权 | `tools/eval_ime_lengths.py` |
| `yue-guard` | 守卫决策：规则 vs 神经裸输出 vs 神经+守卫；救回/弄丢分开报 + 守卫样例 | `experiments/exp_yue_guard.py` |
| `zh-sentence` | 简中整句：自洽题面、Top1/前9/平均键数；`--lm` 加跑重排对照（救回/弄坏分开报） | `tools/eval_sentence.py`（+ `tools/eval_personal_docs.py` 的方法论） |

```
run_all.py        统一入口：--suite {all,yue-lengths,yue-guard,zh-sentence} --quick
suite_*.py        三个套件模块（题面构造 import 自 tools/，口径零漂移）
bench_common.py   仓库路径、环境信息、神经模型加载（与 experiments/ 同一加载方式）
schema_check.py   结果 JSON 校验器（schemas.md 的可执行形态）
schemas.md        JSON 格式规范 + 版本化规则（改口径必须升 suite_version）
METHODOLOGY.md    方法学：分桶、照抄列、救回/弄坏、自洽乐观偏差、样本量与功效
test_runner.py    pytest：schema 校验、--quick 可跑通（无 torch 依赖处用桩模型）
results/          实测产出（本次为 --quick 实跑结果，可覆盖重生成）
```

## 快速开始

```bash
PY=/Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python
cd projects/06-benchmark

$PY run_all.py --suite all --quick            # 全部三套件，每桶 ≤50 题，~11s
$PY run_all.py --suite yue-lengths            # 单套件全量口径
$PY run_all.py --suite zh-sentence --quick --lm   # 整句 + 重排对照（救回/弄坏）
$PY -m pytest test_runner.py -v               # 套件自测（16 个）
```

数据前置（缺什么套件就 graceful skip，`status: "skipped"`，不算失败）：

| 依赖 | 生成方式 | 谁需要 |
|---|---|---|
| `vendor/yue_pairs.json` | `make corpus` | 全部三个 |
| `data/lexicon.tsv` 等 marshal | `make lexicon` / 入库 | zh-sentence |
| `experiments/yue_model_d256.pt` | `make train-yue` | yue-lengths 的神经列、yue-guard |
| `data/yue_guard.eval.marshal` | `python3 tools/build_yue_guard.py`（默认 eval 切分，不入库） | 仅 yue-guard |
| `data/lm_char.marshal` | `make lm`（可选） | 仅 zh-sentence 的 `--lm` |

## --quick 实测（2026-10-25，M 系列 macOS，Python 3.13.12 / torch 2.14.1）

总耗时 **~9s**（yue-lengths 3.6s + yue-guard 4.6s + zh-sentence 0.7s，含模型加载；
首跑含 torch 预热约 11s）。

```
[ok] yue-lengths v1.0.0 (3.58s)
  2字         n=50   照抄=0.0%   规则=60.0% 神经=86.0%
  4字成语      n=40   照抄=100.0% 规则=97.5% 神经=72.5%   ← 神经把成语改坏，照抄列钉死上限
  5-7字       n=50   照抄=14.0%  规则=36.0% 神经=56.0%
  5-7字古诗    n=12   照抄=100.0% 规则=75.0% 神经=50.0%
  加权（覆盖 80%）：规则 67.2% → 神经 76.2%
[ok] yue-guard v1.0.0 (4.58s)
  2字         n=50   规则=60.0% 神经=86.0% +守卫=90.0% 否决=3  救回=2  弄丢=0
  4字成语      n=40   规则=97.5% 神经=72.5% +守卫=95.0% 否决=23 救回=10 弄丢=0
  5-7字       n=50   规则=36.0% 神经=56.0% +守卫=44.0% 否决=29 救回=1  弄丢=6
  5-7字古诗    n=12   规则=75.0% 神经=50.0% +守卫=75.0% 否决=8  救回=3  弄丢=0
  8-15字      n=50   规则=16.0% 神经=34.0% +守卫=32.0% 否决=3  救回=0  弄丢=0
[ok] zh-sentence v1.0.0 (0.69s)
  2-4字     自洽题=23 Top1=95.7% 前9=100.0% 平均键数=10.9
  5-7字     自洽题=20 Top1=80.0% 前9=95.0%  平均键数=17.9
  8-15字    自洽题=10 Top1=70.0% 前9=80.0%  平均键数=36.5
  16+字     自洽题=7  Top1=57.1% 前9=57.1%  平均键数=51.7
```

`zh-sentence --quick --lm`（~0.9s）的重排对照：不重排 81.7% → 重排 85.0%，
**救回 3 / 弄坏 1 / 净 +2**。完整 JSON 见 `results/`。

### 实际 JSON 样例（`results/yue-lengths__quick.json`，有删节）

```json
{
  "suite": "yue-lengths",
  "suite_version": "1.0.0",
  "spec": "projects/06-benchmark/schemas.md (envelope v1)",
  "status": "ok",
  "quick": true,
  "generated_at": "2026-10-25T19:17:10+08:00",
  "duration_s": 3.58,
  "params": {"quick": true, "nn": "auto", "ctx_max": 7, "device": "cpu", "...": "..."},
  "environment": {"python": "3.13.12", "platform": "macOS-26.6.2-arm64", "torch": "2.14.1"},
  "reason": null,
  "results": {
    "buckets": {
      "2字": {"n": 50, "copy_baseline": 0.0, "rule": 0.6, "neural": 0.86,
               "neural_median_latency_ms": 8.63, "weight": 0.4},
      "4字成语": {"n": 40, "copy_baseline": 1.0, "rule": 0.975, "neural": 0.725,
                  "neural_median_latency_ms": 14.264, "weight": 0.25}
    },
    "weighted": {"weighted_rule": 0.6722, "weighted_neural": 0.7616,
                 "weight_coverage": 0.8},
    "empty_buckets": ["3字"],
    "neural_note": null
  }
}
```

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
**METHODOLOGY.md**（一句话版：`--quick` 只能冒烟，60~300 条分辨不了 1~3pp）。

## 提交新系统进基准

入口约定：新系统实现一个纯函数，通过套件模块的参数注入即可，
与内置系统同题同判分、同 schema 落盘。

**粤语套件（yue-lengths / yue-guard）** —— 简体输入 → 繁体粤语输出：

```python
# my_system.py
def convert(text: str) -> str:
    return my_model.translate(text)          # 贪心/beam 均可，耗时会被单独统计

# 跑基准（同目录下）：
from suite_yue_lengths import run
status, results, reason = run({"quick": False, "_nn_fn": convert})
# suite_yue_guard 同理；yue-lengths 也可不注入：--nn auto 会加载 --model 指定的权重
```

**简中整句（zh-sentence）** —— 拼音键串 → 候选词列表（查 Top1/前9）：

```python
def candidates(key: str) -> list[str]:
    return my_engine.predict(key)[:9]        # ["今天天气", "今天", ...]

from suite_zh_sentence import run
status, results, reason = run({"quick": False, "candidate_fn": candidates,
                               "lm": "data/lm_char.marshal", "lm_weight": 0.4})
```

提交清单（PR 必附）：

1. `suite_version` 不变——你换的是被测系统，不是基准；
2. 注明训练语料是否与 `vendor/yue_pairs.json` 重叠（重叠 = 分数虚高，见
   METHODOLOGY §7）；
3. 全量口径（不带 `--quick`）+ 完整 JSON 落 `results/`，附
   `params`/`environment`；
4. 声明单题延迟（yue 套件自动统计中位数；zh-sentence 需自带计时）。

## 边界

- 本目录只**引用**大文件（`../../vendor/`、`../../data/`、`../../tools/`），
  不复制；`vendor/` 不入库，clone 后先 `make corpus`。
- 不修改 `src/`、`experiments/`、`tools/`、Makefile——口径升级只动本目录
  并升 `suite_version`。
- `test_runner.py` 在数据缺失的机器上会逐测试 skip（schema 与版本化纪律
  的测试永远可跑），全量通过环境见上方实测段。
