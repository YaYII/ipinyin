"""projects/02-batch-convert 的测试：分句、字幕解析/重组、转换管线、CLI 集成。

跑法（隔离环境，有 onnxruntime）：
  /Users/yingyang/.workbuddy/binaries/python/envs/default/bin/python \
      -m pytest projects/02-batch-convert/test_convert.py -v

守卫行为用「小桩数据」测：直接构造 YueGuard 的小字典 + 假神经模型，
不依赖真模型文件；真引擎只在数据齐全时跑一条 parity 测试。
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import convert
import sentences
from ipinyin.yue import YueConverter
from ipinyin.yue_guard import YueGuard

ROOT = convert.ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# ---- 桩 ---------------------------------------------------------------------

class FakeNN:
    """查表式假神经模型：没收录的句子照抄原文。"""

    def __init__(self, table: dict[str, str]):
        self.table = table

    def translate(self, text: str) -> str:
        return self.table.get(text, text)


def stub_guard() -> YueGuard:
    """最小守卫数据：只认 不懂→唔識 这一个词级替换对。"""
    return YueGuard({
        "word_pairs": {"不懂": ["唔識"], "不知道": ["唔知道"]},
        "singles": {}, "char_pairs": {}, "left": {}, "right": {},
        "vocab": ["唔識", "唔知道"],
        "colloquial": "唔",
    })


def stub_conv(table: dict[str, str], mapping: dict[str, str] | None = None) -> YueConverter:
    conv = YueConverter(mapping or {})
    conv.nn = FakeNN(table)
    conv.guard = stub_guard()
    return conv


# ---- 分句 -------------------------------------------------------------------

def test_split_terminal():
    assert sentences.split_sentences("今天真好。明天更好！") == ["今天真好。", "明天更好！"]


def test_split_closer_attaches_to_sentence():
    assert sentences.split_sentences("他说「走吧」。她笑了。") == ["他说「走吧」。", "她笑了。"]


def test_split_decimal_not_cut():
    assert sentences.split_sentences("圆周率约3.14159没错。好的。") \
        == ["圆周率约3.14159没错。", "好的。"]


def test_split_lossless():
    text = "第一句。第二句！第三句？他说「好的」。”最后没有标点"
    assert "".join(sentences.split_sentences(text)) == text


def test_too_long_hard_split_under_limit():
    seg = "这一句没有任何标点" * 8
    pieces = sentences.split_sentences(seg, max_len=20)
    assert all(len(p) <= 20 for p in pieces)
    assert "".join(pieces) == seg


def test_too_long_prefers_secondary_punct():
    seg = "先去超市买东西，然后回家做饭，最后洗衣服。"
    pieces = sentences.split_sentences(seg, max_len=10)
    assert all(len(p) <= 10 for p in pieces)
    assert pieces[0] == "先去超市买东西，"
    assert "".join(pieces) == seg


def test_default_unit_len_is_guard_strict_len():
    assert convert.DEFAULT_UNIT_LEN == convert.GUARD_CTX_MAX_LEN == 7


def test_hard_cut_respects_lexicon_words():
    words = {"频道", "想法", "买东西"}
    pieces = sentences.split_sentences("欢迎来看我的频道好吗", max_len=7, words=words)
    assert pieces == ["欢迎来看我的", "频道好吗"]      # 频道没被切成 频|道
    pieces = sentences.split_sentences("我们不懂他的想法也好", max_len=7, words=words)
    assert any("想法" in p for p in pieces)
    assert "".join(pieces) == "我们不懂他的想法也好"


def test_no_cjk_segment_passes_through():
    conv = stub_conv({})
    pipe = convert.Pipeline(conv, words=set())
    assert pipe.convert_line("hello 123!") == "hello 123!"
    assert pipe.records == []


def test_hard_split_keeps_alnum_token_whole():
    seg = "编号是xyz123456789abc后面是中文。"
    pieces = sentences.split_sentences(seg, max_len=18)
    assert all(len(p) <= 18 for p in pieces)
    assert "".join(pieces) == seg
    assert any("xyz123456789abc" in p for p in pieces)   # 编号没被拦腰切断


def test_hard_split_pure_ascii_terminates():
    seg = "a1b2c3" * 10                                       # 60 字无标点英文数字
    pieces = sentences.split_sentences(seg, max_len=10)
    assert all(1 <= len(p) <= 10 for p in pieces)
    assert "".join(pieces) == seg


# ---- 字幕解析 / 重组 ----------------------------------------------------------

SRT_SAMPLE = (
    "1\n"
    "00:00:01,000 --> 00:00:04,000\n"
    "大家好。\n"
    "\n"
    "2\n"
    "00:00:05,000 --> 00:00:08,500\n"
    "第一行。\n"
    "第二行。\n"
    "\n"
)

VTT_SAMPLE = (
    "WEBVTT\n"
    "\n"
    "NOTE 这是一段注释\n"
    "\n"
    "intro\n"
    "00:00:01.000 --> 00:00:04.000\n"
    "大家好。\n"
    "\n"
    "00:00:05.000 --> 00:00:08.000 position:50%\n"
    "第二句。\n"
)


def test_parse_srt_cues():
    items = convert.parse_subtitle(SRT_SAMPLE)
    cues = [p for k, p in items if k == "cue"]
    assert len(cues) == 2
    assert cues[0].head == ["1"]
    assert cues[0].timing == "00:00:01,000 --> 00:00:04,000"
    assert cues[0].text == ["大家好。"]
    assert cues[1].text == ["第一行。", "第二行。"]      # 多行 cue


def test_srt_round_trip_verbatim():
    assert convert.render_subtitle(convert.parse_subtitle(SRT_SAMPLE)) == SRT_SAMPLE


def test_parse_vtt_keeps_header_and_note():
    items = convert.parse_subtitle(VTT_SAMPLE)
    cues = [p for k, p in items if k == "cue"]
    assert len(cues) == 2
    assert cues[0].head == ["intro"]                     # VTT 标识行
    assert cues[1].timing.endswith("position:50%")       # 布局参数随时间轴保留
    lines = convert.render_subtitle(items).splitlines()
    assert "WEBVTT" in lines
    assert "NOTE 这是一段注释" in lines


def test_srt_convert_timeline_untouched():
    # 桩守卫只认 不懂→唔識 这一个替换对：cue 文本用它，神经输出才过得了守卫
    conv = YueConverter({})
    conv.nn = FakeNN({"不懂。": "唔識。"})
    conv.guard = stub_guard()
    srt = SRT_SAMPLE.replace("大家好。", "不懂。")
    pipe = convert.Pipeline(conv)
    out = convert.convert_subtitle(srt, pipe)
    timings_in = [ln for ln in srt.splitlines() if "-->" in ln]
    timings_out = [ln for ln in out.splitlines() if "-->" in ln]
    assert timings_out == timings_in                     # 时间轴一个字符都不动
    assert "唔識。" in out.splitlines()
    assert out.splitlines()[0] == "1"                    # 序号原样


# ---- 转换管线（守卫 + 规则回退） -----------------------------------------------

def test_guard_allows_evidenced_change():
    out, source, fixes = convert.convert_one(stub_conv({"不懂": "唔識"}), "不懂")
    assert (out, source, fixes) == ("唔識", "neural", 0)


def test_guard_blocks_unevidenced_change():
    # 神经把 亡羊补牢 改成 亡羊補監（语料里只有 坐牢→坐監 的实证）→ 整块回退规则版
    out, source, fixes = convert.convert_one(stub_conv({"亡羊补牢": "亡羊補監"}), "亡羊补牢")
    assert out == "亡羊补牢"
    assert source == "neural"
    assert fixes == 1


def test_rule_fallback_without_nn():
    conv = YueConverter({"我们": "我哋"})
    conv.nn = None
    conv.guard = None
    out, source, fixes = convert.convert_one(conv, "我们不懂")
    assert (out, source, fixes) == ("我哋不懂", "rule", 0)


def test_change_ratio():
    assert convert.change_ratio("大家好。", "大家好。") == 0.0
    assert convert.change_ratio("AB", "CD") == 1.0
    assert convert.change_ratio("AXB", "AYB") == pytest.approx(1 / 3)


def test_pipeline_join_is_lossless_and_records_stats():
    # 两个替换对都在桩守卫的实证表里，神经输出原样放行
    table = {"不懂。": "唔識。", "不知道。": "唔知道。"}
    conv = stub_conv(table)
    pipe = convert.Pipeline(conv)
    line = "不懂。不知道。"
    out = pipe.convert_line(line)
    assert out == "唔識。唔知道。"
    assert [r.source for r in pipe.records] == ["neural", "neural"]
    assert pipe.summary().startswith("共 2 段")


# ---- 真引擎 parity（数据齐全才跑） ----------------------------------------------

HAS_DATA = all(os.path.exists(os.path.join(ROOT, "data", f))
               for f in ("yue.marshal", "yue.tsv", "yue_guard.marshal"))


@pytest.mark.skipif(not HAS_DATA, reason="仓库 data/ 不齐")
def test_parity_with_real_engine():
    conv = convert.load_engine()
    for text in ("我们不懂这个问题。", "亡羊补牢", "你现在在干什么？",
                 "亡羊补牢，为时未晚。", "今天天气真好。"):
        out, _, _ = convert.convert_one(conv, text)
        assert out == conv.convert(text), text


@pytest.mark.skipif(not HAS_DATA, reason="仓库 data/ 不齐")
def test_real_engine_guard_fixes_poem():
    conv = convert.load_engine()
    out, _, fixes = convert.convert_one(conv, "春眠不觉晓")
    assert out == "春眠不觉晓"        # 古诗不该被改：神经输出 春眠唔覺曉 被拦
    assert fixes == 1


@pytest.mark.skipif(not HAS_DATA, reason="仓库 data/ 不齐")
def test_default_chunking_blocks_idiom_inside_sentence():
    # 7 字单位下每个片段都走守卫严格路径：句中成语不再被静默改坏
    # （36 字一次性整句会漏拦成 亡羊補監，见 README 局限）
    pipe = convert.Pipeline(convert.load_engine())
    out = pipe.convert_line("亡羊补牢，为时未晚。")
    assert "監" not in out
    assert "亡羊补牢" in out
    assert sum(r.guard_fixes for r in pipe.records) >= 1


# ---- CLI 集成 ----------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "samples", "demo.srt")),
                    reason="样例缺失")
def test_cli_srt_end_to_end():
    src = os.path.join(HERE, "samples", "demo.srt")
    proc = subprocess.run([PY, os.path.join(HERE, "convert.py"), src],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    timings_in = [ln for ln in open(src, encoding="utf-8").read().splitlines()
                  if "-->" in ln]
    timings_out = [ln for ln in proc.stdout.splitlines() if "-->" in ln]
    assert timings_out == timings_in
    assert "歡迎嚟睇我嘅頻道" in proc.stdout      # 真引擎产出
