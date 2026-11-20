"""粤语质量基准：把种子句子跑过转换器，产出给 AI 裁判用的样本。

为什么需要它：之前判断「地道不地道」全靠人工挑几个例子，很容易自欺。
这个脚本只做一件事——把 data/eval/yue_seed.tsv 里的普通话逐句转换，
输出 data/eval/yue_cases.json，作为 AI 裁判（ts_judge）的输入。

用法：
    python3 tools/eval_yue_ai.py            # 生成样本
    python3 tools/eval_yue_ai.py --show     # 顺便打印出来看
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SEED = os.path.join(ROOT, "data", "eval", "yue_seed.tsv")
OUT = os.path.join(ROOT, "data", "eval", "yue_cases.json")


def main(argv: list[str]) -> int:
    from ipinyin import engine

    converter = engine.get_yue()
    if not os.path.exists(SEED):
        print(f"缺 {SEED}", file=sys.stderr)
        return 1

    cases: list[dict] = []
    with open(SEED, encoding="utf-8") as handle:
        for line in handle:
            zh = line.strip()
            if not zh or zh.startswith("#"):
                continue
            yue = converter.convert(zh, neural=True)
            cases.append({"zh": zh, "yue": yue})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(cases, handle, ensure_ascii=False, indent=1)

    print(f"{len(cases)} 条 → {OUT}")
    if "--show" in argv:
        for index, case in enumerate(cases):
            print(f"{index:2d} {case['zh']} → {case['yue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
