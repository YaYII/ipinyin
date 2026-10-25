#!/usr/bin/env python3
"""把项目③的 G2P 送测 hon9kon9ize/yue-g2p-benchmark。

基准协议（读自其 data.py，2026-10-26）：
  - 数据行：文本里 ▁ 标记一个目标字；去掉 ▁ 后 query_id 指向该字；
  - 模型接口：model(texts) → 每文本一个「逐字粤拼」列表（可含 None，
    None 记 4 个音素全错）；
  - 金标支持 / 分隔的多读音；指标 = Accuracy + PER（音节四元组
    onset/nucleus/coda/tone 的 Hamming，取多读音最小距离）。

本文件自带同口径的评分实现（规则表与 calculate_accuracy 逐行对齐其源码，
来源：github.com/hon9kon9ize/yue-g2p-benchmark，MIT 未标注、引用注明出处），
不依赖克隆该仓库即可出分。数据放 vendor/yue_g2p_benchmark/（fetch 脚本下载）。

用法：python3 benchmark_adapter.py            # 三个数据集全跑
      python3 benchmark_adapter.py --dataset wordshk
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from functools import reduce

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

ANCHOR = "▁"

# ---- 评分（逐行对齐 hon9kon9ize/yue-g2p-benchmark data.py） ------------------

JYUTPING_TO_PHONEME_RULES = {
    r"ng": "N",
    r"g(w|(?=u(?!N|k)))": "G",
    r"k(w|(?=u(?!N|k)))": "K",
    r"aa": "A",
    r"oe": "O",
    r"eo": "E",
    r"yu": "Y",
    r"e(?=i)|i(?=N|k)": "I",
    r"o(?=u)|u(?=N|k)": "U",
    r"^(?=[mN]\d)": "__",
    r"(?<=^h)(?=[mN]\d)": "_",
    r"^(?=[aeiouAEIOUY])": "_",
    r"(?<=^..)(?=\d)": "_",
}
PHONEMES_PER_SYLLABLE = 4


def jyutping_to_phonemes(jyutping: str) -> str:
    phonemes = reduce(lambda pron, rule: re.sub(*rule, pron),
                      JYUTPING_TO_PHONEME_RULES.items(), jyutping)
    if len(phonemes) != PHONEMES_PER_SYLLABLE:
        raise ValueError(f"Invalid jyutping: {jyutping}, {phonemes}")
    return phonemes


def score(predictions: list[list[str | None]], texts: list[str],
          query_ids: list[int], golds: list[set[str]]) -> tuple[float, float]:
    total = correct = errors = 0
    for text, qid, true_phons, preds in zip(texts, query_ids, golds, predictions):
        if len(preds) > qid:
            total += 1
            pred = preds[qid]
            if pred is None:
                errors += PHONEMES_PER_SYLLABLE
                continue
            pred_phon = jyutping_to_phonemes(pred)
            if pred_phon in true_phons:
                correct += 1
            errors += min(sum(a != b for a, b in zip(pred_phon, alt))
                          for alt in true_phons)
    if not total:
        return 0.0, 1.0
    return correct / total, errors / total / PHONEMES_PER_SYLLABLE


# ---- 我们的模型：G2P 的逐字对齐输出 ---------------------------------------

class IPinyinG2P:
    """逐字粤拼：G2P.segment 的 token 展开成（字 → 音节）序列。

    基准输入已是粤语书面语（wordshk 词条/古文/口语转写），跳过普→粤转换，
    直接 segment + 注音；词 token 的 k 个字与 k 个音节一一对应。
    """

    def __init__(self) -> None:
        from g2p import G2P

        self.core = G2P(neural=False)

    def predict(self, texts: list[str]) -> list[list[str | None]]:
        out: list[list[str | None]] = []
        for text in texts:
            per_char: list[str | None] = []
            for tok in self.core.segment(text):
                if tok.kind in ("word", "char"):
                    syls = tok.syllables
                    # 音节数与字数不齐（脏数据）时按位对齐，缺位补 None
                    for k in range(len(tok.surface)):
                        per_char.append(syls[k] if k < len(syls) else None)
                elif tok.kind == "missing":
                    per_char.append(None)
                else:                        # punct / latin：占位，query 一般不落这
                    per_char.extend([None] * len(tok.surface))
            out.append(per_char)
        return out


def run(dataset: str, data_dir: str, model: IPinyinG2P) -> dict:
    raw = open(os.path.join(data_dir, f"{dataset}.sent"), encoding="utf-8").read().rstrip().split("\n")
    qids = [line.index(ANCHOR) for line in raw]
    texts = [line.replace(ANCHOR, "") for line in raw]
    labels = open(os.path.join(data_dir, f"{dataset}.lb"), encoding="utf-8").read().rstrip().split("\n")
    golds = [{jyutping_to_phonemes(j) for j in alt.split("/")} for alt in labels]

    preds = model.predict(texts)
    acc, per = score(preds, texts, qids, golds)
    return {"n": len(texts), "accuracy": round(acc, 4), "per": round(per, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "wordshk", "classical", "colloquial"])
    ap.add_argument("--data", default=os.path.join(ROOT, "vendor", "yue_g2p_benchmark"))
    args = ap.parse_args()

    names = ["wordshk", "classical", "colloquial"] if args.dataset == "all" else [args.dataset]
    model = IPinyinG2P()
    print(f"{'数据集':<12}{'n':>6}{'Accuracy':>10}{'PER':>8}")
    for name in names:
        r = run(name, args.data, model)
        print(f"{name:<12}{r['n']:>6}{r['accuracy']:>10.4f}{r['per']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
