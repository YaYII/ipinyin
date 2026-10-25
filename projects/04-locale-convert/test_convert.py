#!/usr/bin/env python3
"""projects/04-locale-convert/convert.py 的单元测试。

离线确定性用例走手写 mini 词典（不依赖网络与数据文件）；
真实数据用例（OpenCC 词典）在 data/opencc_pairs.json 缺失时跳过。
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from convert import (DictSet, LocaleConverter, convert_pass,  # noqa: E402
                     load_dicts, shadow_conflicts)
from seed_data import SEED_DICTS  # noqa: E402

PAIRS_JSON = os.path.join(PROJECT_DIR, "data", "opencc_pairs.json")

MINI = DictSet.from_raw({
    "st_phrases": {"软件": ["軟件"], "软件工程": ["軟件工程"], "头发": ["頭髮"]},
    "st_chars": {"软": ["軟"], "发": ["發", "髮"], "们": ["們"],
                 "头": ["頭"], "这": ["這"], "个": ["個"],
                 "错": ["錯"], "不": ["不"], "白": ["白"],
                 "义": ["義"], "须": ["須"]},
    "tw_phrases": {"軟件": ["軟體"], "軟件工程": ["軟體工程"],
                   "信息": ["資訊"], "程序正義": ["程序正義"],
                   "程序": ["程式"]},
    "tw_variants": {"裏": ["裡"]},
    "tw_phrases_rev": {"軟體": ["軟件"], "資訊": ["信息"]},
    "ts_phrases": {"乾坤": ["乾坤"]},
    "ts_chars": {"軟": ["软"], "體": ["体"], "資": ["资"], "訊": ["讯"],
                 "們": ["们"], "髮": ["发"]},
})


# ------------------------------------------------------------ 最长匹配

def test_longest_match_prefers_longer_phrase():
    conv = LocaleConverter(MINI, "cn2tw")
    # 软件工程 整体命中，不能被 软件 截断
    assert conv.convert("软件工程") == "軟體工程"
    assert conv.convert("软件很好") == "軟體很好"


def test_phrase_beats_char_map():
    conv = LocaleConverter(MINI, "cn2tw")
    # 头发 走短语表（頭髮），单字映射会把 发 转成首选 發
    assert conv.convert("头发") == "頭髮"
    assert conv.convert("白发") == "白發"      # 没有短语兜底时取字级首选（已知歧义）


def test_two_stage_chain():
    conv = LocaleConverter(MINI, "cn2tw")
    # 第一级 s2t 得 軟件，第二级 TWPhrases 得 軟體
    assert conv.convert("这个软件不错") == "這個軟體不錯"
    # identity 守卫词条：程序正義 不被 程序→程式 波及
    assert conv.convert("程序正义") == "程序正義"
    assert conv.convert("程序") == "程式"


def test_chars_only_mode():
    conv = LocaleConverter(MINI, "cn2tw", use_phrases=False)
    assert conv.convert("软件") == "軟件"       # 只做字级
    assert conv.convert("头发") == "頭發"


def test_reverse_direction():
    conv = LocaleConverter(MINI, "tw2cn")
    assert conv.convert("軟體") == "软件"
    assert conv.convert("資訊") == "信息"
    # ts_phrases 守卫：乾坤 不被 乾→干 拆坏
    assert conv.convert("乾坤") == "乾坤"


def test_bidirectional_round_trip():
    cn = LocaleConverter(MINI, "cn2tw")
    tw = LocaleConverter(MINI, "tw2cn")
    for word in ("软件", "信息"):
        assert tw.convert(cn.convert(word)) == word


def test_passthrough_non_cjk():
    conv = LocaleConverter(MINI, "cn2tw")
    text = "Hello, World! 123 https://a.b/c"
    assert conv.convert(text) == text
    assert conv.convert("CPU占用高") == "CPU占用高"   # 未知字原样保留


def test_convert_pass_takes_first_target():
    assert convert_pass("发", {}, {"发": ["發", "髮"]}) == "發"


def test_seed_fallback_load():
    dicts = load_dicts(os.path.join(PROJECT_DIR, "data", "__missing__.json"))
    assert dicts.meta.get("source") == "seed"
    conv = LocaleConverter(dicts, "cn2tw")
    assert conv.convert("信息") == "資訊"


# ------------------------------------------------------------ 歧义报告

def test_shadow_conflict_detected():
    # ABC→YZ 覆盖 AB→X，覆盖段没有 X：语义不一致，必须上报
    conflicts, consistent = shadow_conflicts({"AB": "X", "ABC": "YZ"})
    assert ("AB", "X", "ABC", "YZ") in conflicts
    assert consistent == 0


def test_shadow_consistent_counted():
    # ABC→XC 包含 AB→X：一致覆盖，只计数不上报
    conflicts, consistent = shadow_conflicts({"AB": "X", "ABC": "XC"})
    assert conflicts == []
    assert consistent == 1


def test_real_report_sections():
    dicts = load_dicts(PAIRS_JSON)
    from convert import build_report
    report = build_report(dicts)
    # 一对多：真实词典必然存在（st_chars 的 发/干 一类）
    total_multi = sum(len(v) for v in report["multi_target"].values())
    assert total_multi > 0
    # 长度冲突：TWPhrases 的 程序→程式 被 identity 守卫覆盖，必须出现
    l2 = report["shadow"]["cn2tw#L2"]
    assert any(sub == "程序" for sub, _, _, _ in l2["conflicts"])


# ------------------------------------------------------------ 真实数据集成

@pytest.mark.skipif(not os.path.exists(PAIRS_JSON),
                    reason="缺少 data/opencc_pairs.json（先跑 fetch_opencc.py）")
class TestRealData:
    @pytest.fixture(autouse=True)
    def _conv(self):
        self.dicts = load_dicts(PAIRS_JSON)
        self.cn2tw = LocaleConverter(self.dicts, "cn2tw")
        self.tw2cn = LocaleConverter(self.dicts, "tw2cn")

    def test_word_choice(self):
        assert self.cn2tw.convert("信息") == "資訊"
        assert self.cn2tw.convert("软件") == "軟體"
        assert self.cn2tw.convert("视频") == "影片"
        assert self.tw2cn.convert("軟體") == "软件"
        assert self.tw2cn.convert("資訊") == "信息"

    def test_guard_phrases(self):
        # OpenCC 的 identity 守卫：固定搭配不被 用词替换 波及
        assert self.cn2tw.convert("程序正义") == "程序正義"
        assert self.cn2tw.convert("视频会议") == "視訊會議"   # 而不是 影片會議

    def test_long_sentence(self):
        assert self.cn2tw.convert("把视频上传到网络硬盘。") == "把影片上傳到網路硬碟。"

    def test_round_trip_on_pairs(self):
        # 双向词对的交集应能完整往返（抽验核心用词）
        obj = __import__("json").load(open(PAIRS_JSON, encoding="utf-8"))
        checked = 0
        for cn, tw in obj["cn2tw_pairs"].items():
            if obj["tw2cn_pairs"].get(tw) == cn and len(cn) >= 2:
                assert self.tw2cn.convert(self.cn2tw.convert(cn)) == cn, cn
                checked += 1
                if checked >= 200:
                    break
        assert checked >= 50

    def test_cli_main(self, capsys):
        from convert import main
        assert main(["cn2tw", "--text", "我们需要更多信息。"]) == 0
        assert capsys.readouterr().out.strip() == "我們需要更多資訊。"
