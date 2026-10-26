"""dict_fuse.py 单测：反向词典构建 / 共现校验 / 最长匹配切分 / 覆盖度重排（纯逻辑，不依赖 torch）。"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dict_fuse as df


def test_build_reverse_dict_unanimous_only():
    pairs = [
        [("我们", "我哋"), ("你们", "你哋")],        # tsv 源
        [("没空", "唔得閒"), ("小孩", "細路")],      # word_pairs 源
    ]
    d = df.build_reverse_dict(pairs)
    assert d == {"我哋": "我们", "你哋": "你们", "唔得閒": "没空", "細路": "小孩"}


def test_build_reverse_dict_drops_conflicts_and_identity():
    pairs = [
        [("的", "嘅"), ("是", "嘅")],        # 同一粤语词两个普通话词 → 冲突丢弃
        [("咖啡", "咖啡")],                  # 恒等映射无信息 → 丢弃
    ]
    assert df.build_reverse_dict(pairs) == {}


def test_build_reverse_dict_cross_source_conflict_dropped():
    # 两个来源对同一 yue 词给出不同 zh → 仍算冲突
    pairs = [[("了解", "瞭解")], [("理解", "瞭解")]]
    assert df.build_reverse_dict(pairs) == {}


def test_build_reverse_dict_len_bounds():
    pairs = [[("长词词词词词词词词词词词", "粵語詞詞詞詞詞詞詞詞詞詞詞"), ("短", "短")]]
    d = df.build_reverse_dict(pairs, min_len=1, max_len=5)
    assert d == {}          # 键超 max_len、恒等映射，都进不来


def test_corpus_verification_disambiguates_synonyms():
    # 我们/咱们→我哋 是同义多映射：语料共现让「我们」胜出，而不是整条误杀
    pairs = [[("我们", "我哋"), ("咱们", "我哋")]]
    corpus = [{"zh": "我们走了", "yue": "我哋走咗"}] * 4 + [{"zh": "咱们走吧", "yue": "我哋啦"}]
    assert df.build_reverse_dict(pairs, corpus_pairs=corpus) == {"我哋": "我们"}


def test_corpus_verification_vetoes_antonym_noise():
    # 挖矿反义噪声：唔得閒（没空）被对到「有空」，语料里共现为 0 → 整条否决
    pairs = [[("有空", "唔得閒")]]
    corpus = [{"zh": "我没空", "yue": "我唔得閒"}] * 5
    assert df.build_reverse_dict(pairs, corpus_pairs=corpus) == {}


def test_corpus_rare_key_falls_back_to_unanimity():
    # 语料见次不足的生僻词退回全票一致规则（yue.tsv 人工单票可信）
    pairs = [[("枯竭", "乾涸")]]
    assert df.build_reverse_dict(pairs, corpus_pairs=[]) == {"乾涸": "枯竭"}


def test_segment_longest_match():
    d = {"我哋": "我们", "我": "我", "係": "是"}
    hits = df.segment("我哋係", d)
    assert hits == [(0, 2, "我们"), (2, 3, "是")]    # 我哋 整词优先于单字 我


def test_segment_skips_unknown():
    d = {"係": "是"}
    assert df.segment("乜嘢係", d) == [(2, 3, "是")]


def test_dict_coverage_basic():
    d = {"唔得閒": "没空", "聽日": "明天"}
    assert df.dict_coverage("唔得閒", "我没空", d) == 1.0
    assert df.dict_coverage("唔得閒", "我不行", d) == 0.0
    # 唔得閒 3 字 + 聽日 2 字；候选只含前者译文 → 3/5
    assert abs(df.dict_coverage("唔得閒聽日", "没空再说", d) - 0.6) < 1e-9


def test_dict_coverage_no_hits_returns_zero():
    assert df.dict_coverage("乜嘢", "什么", {"係": "是"}) == 0.0


def test_rerank_promotes_dict_covered_candidate():
    d = {"唔得閒": "没空"}
    cands = [(-0.50, "我不行"), (-0.55, "我没空")]
    out = df.rerank(cands, "唔得閒", d, lam=1.0)
    assert out[0][1] == "我没空"          # 覆盖度把低分候选顶上来


def test_rerank_lambda_zero_keeps_order():
    d = {"唔得閒": "没空"}
    cands = [(-0.50, "我不行"), (-0.55, "我没空")]
    out = df.rerank(cands, "唔得閒", d, lam=0.0)
    assert out[0][1] == "我不行"


def test_real_sources_build():
    if not (os.path.exists(df.YUE_TSV) and os.path.exists(df.YUE_WORD_PAIRS)):
        return
    merged = os.path.join(HERE, "data", "merged_pairs.json")
    corpus = json.load(open(merged, encoding="utf-8")) if os.path.exists(merged) else None
    d = df.load_default_reverse_dict(corpus_pairs=corpus)
    assert len(d) > 1000
    # 高频功能词必须正确（共现校验后）
    assert d.get("嘅") == "的"
    assert d.get("我哋") == "我们"
    # 反义噪声必须被否决，或至少不许映射到「有空」
    assert d.get("唔得閒") in (None, "没空")
