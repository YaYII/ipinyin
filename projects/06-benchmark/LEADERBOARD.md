# LEADERBOARD

由 `leaderboard.py` 从 results/ 自动汇总；重新生成：
`python3 leaderboard.py`。报数纪律见 METHODOLOGY §6 —— **quick 表不具排名资格**，只用于冒烟与回归。

## yue-lengths（全量）

| 系统 | 版本 | n | 加权 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.3.0 | 377 | 84.3% | 2026-10-26 | `yue-lengths__full.json` |
| 规则转换 | 1.3.0 | 377 | 54.4% | 2026-10-26 | `yue-lengths__full__规则转换.json` |

## yue-guard（全量）

| 系统 | 版本 | n | guarded 微平均 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.3.0 | 677 | 51.3% | 2026-10-26 | `yue-guard__full.json` |

## zh-sentence（全量）

| 系统 | 版本 | n | Top1 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.1.0 | 256 | 85.5% | 2026-10-26 | `zh-sentence__full.json` |

## yue-lengths（--quick 冒烟）
> ⚠️ 冒烟数据（每桶 ≤50 题），**不具排名资格**。

| 系统 | 版本 | n | 加权 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.3.0 | 112 | 86.3% | 2026-10-26 | `yue-lengths__quick.json` |

## yue-guard（--quick 冒烟）
> ⚠️ 冒烟数据（每桶 ≤50 题），**不具排名资格**。

| 系统 | 版本 | n | guarded 微平均 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.3.0 | 162 | 67.9% | 2026-10-26 | `yue-guard__quick.json` |

## zh-sentence（--quick 冒烟）
> ⚠️ 冒烟数据（每桶 ≤50 题），**不具排名资格**。

| 系统 | 版本 | n | Top1 | 日期 | 文件 |
|---|---|---|---|---|---|
| ipinyin(内置) | 1.1.0 | 60 | 86.7% | 2026-10-26 | `zh-sentence__quick.json` |

