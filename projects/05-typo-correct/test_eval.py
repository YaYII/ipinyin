#!/usr/bin/env python3
"""pytest：eval.py 指标计算的正确性（全部用手工构造、能口算出答案的例子）。

跑法：cd projects/05-typo-correct && python -m pytest test_eval.py -v
"""
from __future__ import annotations

import importlib.util
import os
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_eval():
    spec = importlib.util.spec_from_file_location("typo_eval", os.path.join(HERE, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_mod = _load_eval()


def approx(x):
    return pytest.approx(x, abs=1e-9)


# ---------------------------------------------------------------- 例子

def perfect_case():
    gold = [
        {"sentence": "我今天再高兴不过了",  # 「在」被写成「再」
         "errors": [{"char": "再", "correct": "在", "position": 3, "type": "homophone"}]},
        {"sentence": "天气不错", "errors": []},
    ]
    preds = [
        {"sentence": "我今天再高兴不过了",
         "predictions": [{"position": 3, "correct": "在"}]},
        {"sentence": "天气不错", "predictions": []},
    ]
    return gold, preds


def copy_case():
    gold = [
        {"sentence": "我今天再高兴不过了",
         "errors": [{"char": "再", "correct": "在", "position": 3, "type": "homophone"}]},
        {"sentence": "天气不错", "errors": []},
    ]
    preds = [
        {"sentence": "我今天再高兴不过了", "predictions": []},
        {"sentence": "天气不错", "predictions": []},
    ]
    return gold, preds


def mixed_case():
    """能口算的混合场景：
    gold 2 个错：s1 同音(检测到且改对)、s2 形近(检测到但建议字错)、s3 干净句被误报。
    预测 4 个 → 检测 TP=2 FP=2 FN=0；纠正 TP=1, P=1/4, R=1/2, F1=1/3。"""
    gold = [
        {"sentence": "我今天再高兴",  # 0我 1今 2再 3高 4兴
         "errors": [{"char": "再", "correct": "在", "position": 2, "type": "homophone"}]},
        {"sentence": "他木天来",      # 末→木 形近
         "errors": [{"char": "木", "correct": "末", "position": 1, "type": "shape"}]},
        {"sentence": "天气不错", "errors": []},
    ]
    preds = [
        {"sentence": "我今天再高兴",
         "predictions": [{"position": 2, "correct": "在"},   # 检测对、纠正对
                         {"position": 4, "correct": "高"}]},  # 误报
        {"sentence": "他木天来",
         "predictions": [{"position": 1, "correct": "未"}]},  # 检测对、纠正错（建议 未≠末）
        {"sentence": "天气不错",
         "predictions": [{"position": 0, "correct": "天"}]},  # 干净句误报
    ]
    return gold, preds


# ---------------------------------------------------------------- 用例

def test_perfect_scores_one():
    gold, preds = perfect_case()
    m = eval_mod.compute_metrics(gold, preds)
    assert m["detection"]["precision"] == approx(1.0)
    assert m["detection"]["recall"] == approx(1.0)
    assert m["detection"]["f1"] == approx(1.0)
    assert m["correction"]["precision"] == approx(1.0)
    assert m["correction"]["recall"] == approx(1.0)
    assert m["correction"]["f1"] == approx(1.0)


def test_copy_baseline_is_all_zero():
    """照抄不改基线：无预测 → P/R/F1 全 0（分母为 0 按 0 计）。"""
    gold, preds = copy_case()
    m = eval_mod.compute_metrics(gold, preds)
    for key in ("detection", "correction"):
        assert m[key]["precision"] == 0.0
        assert m[key]["recall"] == 0.0
        assert m[key]["f1"] == 0.0
    assert m["counts"]["predictions"] == 0
    assert m["counts"]["errors"] == 1
    assert m["counts"]["clean_sentences"] == 1


def test_mixed_case_hand_computed():
    gold, preds = mixed_case()
    m = eval_mod.compute_metrics(gold, preds)
    # 检测级：TP=2, FP=2, FN=0
    assert m["detection"]["tp"] == 2
    assert m["detection"]["fp"] == 2
    assert m["detection"]["fn"] == 0
    assert m["detection"]["precision"] == approx(0.5)
    assert m["detection"]["recall"] == approx(1.0)
    assert m["detection"]["f1"] == approx(2 * 0.5 * 1.0 / 1.5)
    # 纠正级：TP=1，分母是全部 4 个预测；金标错误共 2 个 → R=1/2
    assert m["correction"]["tp"] == 1
    assert m["correction"]["precision"] == approx(0.25)
    assert m["correction"]["recall"] == approx(0.5)
    assert m["correction"]["f1"] == approx(2 * 0.25 * 0.5 / 0.75)


def test_type_buckets():
    gold, preds = mixed_case()
    m = eval_mod.compute_metrics(gold, preds)
    # 同音桶：金标 1，预测命中且字对 → TP=1；全局 FP=2 计入 precision 分母
    h = m["by_type"]["homophone"]
    assert h["gold"] == 1 and h["tp"] == 1
    assert h["precision"] == approx(1.0 / 3.0)
    assert h["recall"] == approx(1.0)
    assert h["f1"] == approx(0.5)
    # 形近桶：检测到了但建议字错 → TP=0
    s = m["by_type"]["shape"]
    assert s["gold"] == 1 and s["tp"] == 0
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_clean_sentence_with_no_prediction_is_neutral():
    gold, _ = perfect_case()
    preds = [
        {"sentence": gold[0]["sentence"], "predictions": []},
        {"sentence": gold[1]["sentence"], "predictions": []},
    ]
    m = eval_mod.compute_metrics(gold, preds)
    assert m["detection"]["fp"] == 0          # 干净句零预测不产生误报
    assert m["counts"]["clean_sentences"] == 1


def test_misaligned_sentence_raises():
    gold, preds = perfect_case()
    preds[0]["sentence"] = "不一样的句子"
    with pytest.raises(ValueError, match="不对齐"):
        eval_mod.compute_metrics(gold, preds)


def test_row_count_mismatch_raises():
    gold, preds = perfect_case()
    with pytest.raises(ValueError, match="数量不一致"):
        eval_mod.compute_metrics(gold, preds[:1])


def test_prf_zero_denominators():
    d = eval_mod._prf(0, 0, 0)
    assert d["precision"] == 0.0 and d["recall"] == 0.0 and d["f1"] == 0.0
