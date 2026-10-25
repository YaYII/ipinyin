#!/usr/bin/env python
"""项目②基准评测：held-out 语料上量「普通话 → 粤语」批量管线的转换质量。

四个口径同台（正确 = t2s 归一后与参考逐字相同，与 experiments/exp_yue_guard.py
同一判据、同一 seed=0 切分）：

  照抄      直接把输入当输出 —— 识别「分数靠不改」的假象（成语/古诗桶它是满分陷阱）
  规则      纯规则 + t2s（CLI 的 --no-nn 模式）
  整句36    神经 + 守卫，36 字一次性整句（口语流畅，但 >7 字走守卫宽松路径，
            句中成语会被静默改坏 —— 语料桶里少见，demo/测试有实录）
  单位7     神经 + 守卫，7 字转换单位（默认；每段都过严格实证路径）

分桶与 docs/nn-upgrade.md 对齐：2字词（语料对齐挖出、剔除规则表已覆盖）、
4字成语（tools/idioms.txt，答案=不改）、5-7字口语、5-7字古诗（tools/poems.txt）、
8-15字句子。每个桶另报守卫修复段数与 ms/句。

用法：
  python3 projects/02-batch-convert/eval.py              # 每桶 ≤300 句（约 2 分钟）
  python3 projects/02-batch-convert/eval.py --n 50       # 快跑
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_ime_lengths import build_word_test, load_pairs   # noqa: E402
from ipinyin.t2s import Converter                          # noqa: E402

import convert                                             # noqa: E402

MODES = [("照抄", "copy"), ("规则", "rule"), ("整句36", 36), ("单位7", 7)]


def load_buckets(pairs_path: str, limit: int, n: int) -> list[tuple[str, list[tuple[str, str]]]]:
    """分桶，与 docs/nn-upgrade.md / exp_yue_guard.py 的口径一致。"""
    test = load_pairs(pairs_path, limit)
    buckets: list[tuple[str, list[tuple[str, str]]]] = []
    word2 = build_word_test(test).get("2字") or []
    buckets.append(("2字词", word2[:n or None]))
    buckets.append(("4字成语", [(w, "") for w in _read_lines("idioms.txt")[:n or None]]))
    buckets.append(("5-7字", [(p["zh"], p["yue"]) for p in test
                              if 5 <= len(p["zh"]) <= 7][:n or None]))
    buckets.append(("5-7字古诗", [(w, "") for w in _read_lines("poems.txt")[:n or None]]))
    buckets.append(("8-15字", [(p["zh"], p["yue"]) for p in test
                               if 8 <= len(p["zh"]) <= 15][:n or None]))
    return buckets


def _read_lines(name: str) -> list[str]:
    path = os.path.join(ROOT, "tools", name)
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def ref_of(mode_src: str, ref: str, t2s: Converter) -> str:
    """成语/古诗桶的参考答案 = 原样不改（t2s 到繁体）；语料桶用参考句。"""
    return ref if ref else t2s.to_traditional(mode_src)


def score_bucket(items: list[tuple[str, str]], mode: str | int,
                 pipes: dict, t2s: Converter) -> tuple[float, int, float]:
    """返回 (整句一致率, 守卫修复段数, ms/句)。mode ∈ copy/rule/36/7。"""
    pipe = pipes.get(mode)
    ok = fixes = 0
    total_ms = 0.0
    if pipe is not None and pipe.conv.nn is not None:
        pipe.conv.nn._cache.clear()        # 各模式都按冷缓存计时，避免先跑的模式焐热缓存
    for zh, ref in items:
        if mode == "copy":
            got, ms = t2s.to_traditional(zh), 0.0
        else:
            pipe.records.clear()
            t0 = time.perf_counter()
            got = pipe.convert_line(zh)
            ms = (time.perf_counter() - t0) * 1000
            fixes += sum(r.guard_fixes for r in pipe.records)
        if got == ref_of(zh, ref, t2s):
            ok += 1
        total_ms += ms
    n = len(items)
    return (ok / n if n else 0.0), fixes, (total_ms / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="批量转换管线 held-out 评测")
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--limit", type=int, default=150000,
                    help="与训练一致的语料上限（决定 held-out 切分）")
    ap.add_argument("--n", type=int, default=300, help="每个语料桶最多评几句；0 = 不截断")
    ap.add_argument("--samples", type=int, default=3, help="每桶展示的 单位7 错例条数")
    args = ap.parse_args()

    conv = convert.load_engine()
    if conv.nn is None:
        print("提示：神经模型未加载，整句36/单位7 两列会与规则列相同。", file=sys.stderr)
    t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))

    conv_rule = convert.load_engine(use_nn=False)
    pipes = {
        "rule": convert.Pipeline(conv_rule, t2s=t2s),
        36: convert.Pipeline(conv, t2s=t2s, max_len=36),
        7: convert.Pipeline(conv, t2s=t2s, max_len=7),
    }

    buckets = load_buckets(args.pairs, args.limit, args.n)
    header = f"{'桶':<8}{'n':>4}" + "".join(f"{name:>8}" for name, _ in MODES) + \
             f"{'守卫7':>6}{'ms/句7':>8}{'ms/句36':>9}"
    print(header)
    for name, items in buckets:
        if not items:
            continue
        cells, fix7, lat7, lat36 = [], 0, 0.0, 0.0
        errs: list[str] = []
        for mode_name, mode in MODES:
            acc, fixes, ms = score_bucket(items, mode, pipes, t2s)
            cells.append(f"{acc:>7.1%}")
            if mode == 7:
                fix7, lat7 = fixes, ms
            elif mode == 36:
                lat36 = ms
        if args.samples:
            pipe7 = pipes[7]
            for zh, ref in items:
                pipe7.records.clear()
                got = pipe7.convert_line(zh)
                if got != ref_of(zh, ref, t2s):
                    errs.append(f"    {zh!r} → {got!r} 期望 {ref_of(zh, ref, t2s)!r}")
        print(f"{name:<8}{len(items):>4}" + "".join(cells)
              + f"{fix7:>6}{lat7:>7.1f} {lat36:>8.1f}")
        for line in errs[: args.samples]:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
