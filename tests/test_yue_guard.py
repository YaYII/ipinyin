"""P1 约束解码守卫：有实证的改动放行，无实证的回退规则。

测试用合成守卫数据，不依赖 data/yue_guard.marshal。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ipinyin.yue import YueConverter  # noqa: E402
from ipinyin.yue_guard import YueGuard  # noqa: E402


def make_guard() -> YueGuard:
    data = {
        "word_pairs": {
            "我们": ["我哋"],
            "早上好": ["早晨"],
            "什么": ["乜嘢", "咩"],
        },
        "singles": {"是": "係"},
        "char_pairs": {
            ("是", "係"): 100,
            ("不", "唔"): 100,
            ("懂", "識"): 50,
            ("牢", "監"): 10,
            ("补", "補"): 200,
            ("的", "嘅"): 300,
        },
        "left": {("坐", "牢", "監"): 9},
        "right": {("补", "補", "牢"): 5},
        "vocab": ["唔通", "真係", "入面", "巴打"],
        "colloquial": "係唔嘅咗",
    }
    return YueGuard(data, min_char=10, min_ctx=3, ctx_max_len=5)


def test_attested_conversion_passes():
    g = make_guard()
    assert g.check("我们", "我哋") == "我哋"
    assert g.check("我们去看戏", "我哋去看戏") == "我哋去看戏"


def test_multi_target_word_pair():
    g = make_guard()
    # 什么→咩 和 什么→乜嘢 都是有实证的写法
    assert g.check("你想什么", "你想咩") == "你想咩"
    assert g.check("你想什么", "你想乜嘢") == "你想乜嘢"


def test_unattested_idiom_change_repaired():
    g = make_guard()
    # 牢→監 只有 坐牢 语境的实证；补牢 语境没有 → 改回原文
    assert g.check("亡羊补牢", "亡羊補監") == "亡羊补牢"


def test_attested_context_allows_change():
    g = make_guard()
    # 坐牢→坐監 有语境实证，放行
    assert g.check("坐牢", "坐監") == "坐監"


def test_colloquial_insert_long_only():
    g = make_guard()
    # 长句：真→真係 是粤语核心语法，口语字插入放行
    assert g.check("我们真开心啊", "我哋真係开心啊") == "我哋真係开心啊"
    # 短输入不允许无实证插入
    assert g.check("真开心", "真係开心") == "真开心"


def test_shortening_deletion_long_only():
    g = make_guard()
    # 长句：知道→知 这类缩略放行
    assert g.check("我们不知道啊", "我哋唔知啊") == "我哋唔知啊"
    # 短输入不允许删除 → 还原原文
    assert g.check("不知道", "唔知") == "不知道"


def test_known_vocab_piece_long_only():
    g = make_guard()
    # 难道→唔通 凑不出替换对，但 唔通 是语料里常见的粤语词 → 长句放行
    assert g.check("难道是真的呀", "唔通係真嘅呀") == "唔通係真嘅呀"
    # 短输入没有词表兜底 → 还原原文
    assert g.check("难道真的", "唔通真嘅") == "难道真的"


def test_no_change_passthrough():
    g = make_guard()
    assert g.check("早上好", "早上好") == "早上好"


class _FakeNN:
    def __init__(self, table: dict[str, str]):
        self.table = table

    def translate(self, text: str) -> str:
        return self.table.get(text, "")


def test_converter_guard_falls_back_to_rules():
    conv = YueConverter({"是": "係"})
    conv.nn = _FakeNN({"亡羊补牢": "亡羊補監", "我们": "我哋"})
    conv.guard = make_guard()
    # 守卫否掉无实证改动 → 回退规则版（原样）
    assert conv.convert("亡羊补牢", neural=True) == "亡羊补牢"
    # 有实证的转换照常走神经
    assert conv.convert("我们", neural=True) == "我哋"
    # 没有神经输出的走规则
    assert conv.convert("是", neural=True) == "係"


def test_converter_without_guard_keeps_old_behavior():
    conv = YueConverter({"是": "係"})
    conv.nn = _FakeNN({"亡羊补牢": "亡羊補監"})
    assert conv.guard is None
    assert conv.convert("亡羊补牢", neural=True) == "亡羊補監"
