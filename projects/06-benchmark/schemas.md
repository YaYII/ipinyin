# 基准结果 JSON 规范（schemas.md）

本目录所有 suite 的输出共用一个**信封 + 套件专属 results** 的结构。
`run_all.py` 写盘前会用 `schema_check.py` 自检；`test_runner.py` 对每份产出
断言零错误。人类可读的口径说明见 METHODOLOGY.md。

## 文件与命名

```
results/<suite>__<quick|full>.json
```

`quick` / `full` 对应 `--quick` 开关。同一 suite 同一模式**覆盖写**（文件即最新状态，
历史用 git 追溯）。

## 信封字段（所有 suite 共有）

| 字段 | 类型 | 说明 |
|---|---|---|
| `suite` | string | `yue-lengths` \| `yue-guard` \| `zh-sentence` |
| `suite_version` | string | 该 suite 口径版本，semver（见下「版本化规则」） |
| `spec` | string | 指向本规范（当前 envelope v1） |
| `status` | string | `ok` \| `skipped` \| `error` |
| `quick` | bool | 是否小样本模式 |
| `generated_at` | string | ISO 8601 带时区（`2026-10-25T19:10:38+08:00`） |
| `duration_s` | number | 套件运行秒数（≥0） |
| `params` | object | 本次运行的全部参数（含数据/模型绝对路径，作溯源） |
| `environment` | object | `python` / `platform` / `torch`（无 torch 时为 `null`） |
| `reason` | string \| null | `skipped`/`error` 时必填，说明原因；`ok` 时 `null` |
| `results` | object \| null | 套件专属结果；`skipped` 时必须为 `null` |

约定：所有比率字段四舍五入到 4 位小数、取值 [0,1]；计数为非负整数；
**不允许出现 NaN/Infinity**（`json.dump` 关闭 allow_nan 的效果由 schema 校验保证）。

`status` 语义：

- `ok`：跑完，`results` 有效。注意 yue-lengths 在无模型/无 torch 时仍是 `ok`，
  神经列为 `null`（照抄 + 规则两列不受影响）——降级是口径内行为，不是 skip。
- `skipped`：前置条件不满足（语料/模型/守卫数据缺失）。**不算失败**，
  `--suite all` 会继续跑其余 suite，退出码不受影响。
- `error`：套件抛异常或输出未过 schema 校验。`--suite all` 继续，退出码非零。

## yue-lengths 的 results

```jsonc
{
  "buckets": {
    "2字": {                       // 桶名：2字 | 3字 | 4字成语 | 5-7字 | 5-7字古诗
      "n": 50,                     // 该桶实际题数（--quick 每桶 ≤50）
      "copy_baseline": 0.0,        // 照抄不改（简→繁外壳）的准确率
      "rule": 0.6,                 // 规则转换准确率
      "neural": 0.86,              // 神经转换准确率；未评时 null
      "neural_median_latency_ms": 8.63,   // 神经单题延迟中位数；未评时 null
      "weight": 0.4                // 真实打字长度分布权重（与 4字成语/5-7字古诗 为 0）
    }
  },
  "weighted": {
    "weighted_rule": 0.6722,       // Σ(有样本桶的 rule × weight) / Σweight
    "weighted_neural": 0.7616,     // 同上对 neural；无神经列时 null
    "weight_coverage": 0.8         // Σweight / 1.0；<1 说明有桶为空（看 empty_buckets）
  },
  "empty_buckets": ["3字"],        // 题数为 0 被跳过的桶（挖矿口径下 3字 常为空）
  "neural_note": null              // 神经列降级/桩模型时的说明，正常为 null
}
```

## yue-guard 的 results

```jsonc
{
  "buckets": {
    "2字": {                       // 桶名：2字 | 4字成语 | 5-7字 | 5-7字古诗 | 8-15字
      "n": 50,
      "copy_baseline": null,       // 守卫口径不评照抄列（规则列承担对照），恒为 null
      "rule": 0.6,                 // 规则准确率
      "neural": 0.86,              // 神经裸输出准确率
      "guarded": 0.9,              // 神经+守卫准确率
      "rejected": 3,               // 守卫否决（做了块级修复）的题数
      "accepted": 47,              // 守卫放行（原样采纳神经输出）的题数 = n - rejected
      "saved": 2,                  // 修复且救回：规则对、神经错 → 修复后对
      "lost": 0,                   // 修复且弄丢：神经对、规则错 → 修复后错
      "guarded_median_latency_ms": 0.0163,
      "weight": null               // 守卫口径不做打字分布加权，恒为 null
    }
  },
  "aggregate": {"rejected": 3, "saved": 2, "lost": 0},   // 全桶计数求和
  "examples": {                    // 每桶最多 params.show 条守卫决策样例
    "4字成语": [
      {"input": "亡羊补牢", "neural": "亡羊補監", "guarded": "亡羊补牢",
       "expected": "亡羊補牢", "outcome": "saved"}
      // outcome ∈ saved | lost | fixed_right | both_wrong
    ]
  },
  "empty_buckets": [],
  "neural_note": null
}
```

## zh-sentence 的 results

```jsonc
{
  "buckets": {
    "2-4字": {                     // 桶名：2-4字 | 5-7字 | 8-15字 | 16+字
      "n": 23,                     // 自洽题数（题面合法、进入计分的题）
      "invalid": 27,               // 题面不合法被剔除的题数（透明度： optimism 的来源）
      "top1": 0.957,               // 第一候选 == 原句
      "top9": 1.0,                 // 原句出现在前 9 候选
      "avg_keystrokes": 10.9,      // 平均键数（拼音字母数，含自洽过滤前不计）
      "fixed": 1,                  // 仅 --lm 时：本来错、重排救回
      "broken": 0                  // 仅 --lm 时：本来对、重排弄坏
    }
  },
  "aggregate": {"n": 60, "top1": 0.783, "top9": 0.883},
  "rerank": {                      // 未传 --lm 时为 null
    "lm": "…/data/lm_char.marshal", "lm_weight": 0.4,
    "top1_no_lm": 0.817, "top1_lm": 0.85,
    "saved": 3, "broken": 1, "net": 2
  },
  "empty_buckets": [],
  "questions_scanned": 20000       // 题库扫描的句子总数（split 前的池子）
}
```

## 版本化规则（必须遵守）

1. **`suite_version` 只描述口径，不描述被测系统。** 换模型、换权重、重训
   → 版本不变（那是不动基准的实验）；改基准本身才升版本。
2. 以下改动**必须升 minor 或 major**，并在 git commit message 里写明：
   - 增删桶、改桶名、改桶的题面构造（如挖矿 min_count、长度边界、idioms/poems 词表）；
   - 增删指标字段、改指标定义（如 saved/lost 的判定）；
   - 改默认样本量、改加权权重表 `WEIGHTS`；
   - 改守卫口径参数默认值（`ctx_max`）。
3. **字段语义变化 = major**（旧 JSON 与新 JSON 不可同表比较）；
   **只加字段 = minor**；**修 bug 使数字整体小幅变化 = patch**，但必须在
   commit message 给出同一系统升级前后的对照数字。
4. 跨版本比较没有意义：发布结果必须注明 `suite` + `suite_version` +
   `quick` 与否 + 样本量（n）。改口径前先跑一次旧口径存档（results/ 里的
   JSON 就是存档），再升版本。
5. `schema_check.py` 是本规范的可执行形态：改了字段必须同步改它并让
   `test_runner.py` 过。
