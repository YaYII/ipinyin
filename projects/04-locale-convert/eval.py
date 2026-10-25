#!/usr/bin/env python3
"""词表基线评测：held-out 整词命中率 + 语境依赖测试句。

两块内容：

1. held-out 切分（90/10，seed=0）：把合成词对洗牌切分，测试对的词条
   从转换器短语表里移除（单字映射保留 —— 衡量「词表增加值」正是目的），
   然后逐词转换打分。另报「纯单字映射」基线，两者之差即短语表的贡献。
   测试对仍可能靠别的词条组合出来（如 软件→軟體 在训练集，软件开发
   在测试集仍能拼对），这是合法的组合泛化，不算泄题。

2. 语境依赖测试句：16 句手工标注的句子，专门覆盖词表基线原理上做不
   到的场景（一词多义、同形异义、组合词缺失），每句标注预期失败模式。
   这就是本仓库「守卫」思路的地区版考卷：neural 版要先把这里做对。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from convert import (DIRECTIONS, LocaleConverter, convert_pass,  # noqa: E402
                     flatten_first, load_dicts)

PAIRS_JSON = os.path.join(PROJECT_DIR, "data", "opencc_pairs.json")
TRAIN_FRAC = 0.9
SEED = 0


# ---------------------------------------------------------------- 数据准备

def load_pairs(path: str = PAIRS_JSON) -> tuple[dict[str, str], dict[str, str], dict]:
    obj = json.load(open(path, encoding="utf-8"))
    return obj["cn2tw_pairs"], obj["tw2cn_pairs"], obj.get("meta", {})


def split_pairs(pairs: dict[str, str], train_frac: float = TRAIN_FRAC,
                seed: int = SEED) -> tuple[list, list]:
    """确定性切分：按键排序后 seed 洗牌，取后 10% 做测试。"""
    items = sorted(pairs.items())
    random.Random(seed).shuffle(items)
    n_test = max(1, round(len(items) * (1 - train_frac)))
    return items[n_test:], items[:n_test]


def stage1_convert(dicts, text: str) -> str:
    """cn→tw 的第一级（s2t），用于定位测试对在 TWPhrases 里的键形。"""
    return convert_pass(text,
                        flatten_first(dicts.dicts.get("st_phrases", {})),
                        dicts.dicts.get("st_chars", {}))


def held_out_converter(dicts, direction: str, test: list[tuple[str, str]]):
    """构造移除测试词条后的转换器。

    cn→tw：从 st_phrases 移除 cn；若 TWPhrases[t_form] 恰是测试目标，
           连 t_form 一起移除（t_form = 完整词典下的第一级输出）。
    tw→cn：从 tw_phrases_rev / ts_phrases 移除对应源词。
    """
    stripped = {role: dict(entries) for role, entries in dicts.dicts.items()}
    if direction == "cn2tw":
        for cn, tw in test:
            stripped["st_phrases"].pop(cn, None)
            t_form = stage1_convert(dicts, cn)
            targets = dicts.dicts.get("tw_phrases", {}).get(t_form)
            if targets and targets[0] == tw:
                stripped["tw_phrases"].pop(t_form, None)
    else:
        for tw, cn in test:
            stripped["tw_phrases_rev"].pop(tw, None)
            targets = dicts.dicts.get("ts_phrases", {}).get(tw)
            if targets and targets[0] == cn:
                stripped["ts_phrases"].pop(tw, None)
    from convert import DictSet
    return LocaleConverter(DictSet.from_raw(stripped), direction)


# ---------------------------------------------------------------- held-out

def eval_direction(dicts, pairs: dict[str, str], direction: str,
                   miss_limit: int = 15) -> dict:
    train, test = split_pairs(pairs)
    conv = held_out_converter(dicts, direction, test)
    chars_only = LocaleConverter(dicts, direction, use_phrases=False)

    full_hit = char_hit = 0
    word_total = word_hit = 0
    misses: list[tuple[str, str, str]] = []
    for src, want in test:
        got = conv.convert(src)
        char = chars_only.convert(src)
        is_word = char != want                      # 单字映射拼不出的才是「用词对」
        if got == want:
            full_hit += 1
        if char == want:
            char_hit += 1
        if is_word:
            word_total += 1
            if got == want:
                word_hit += 1
        elif got != want and len(misses) < miss_limit:
            misses.append((src, want, got))

    n = len(test)
    # 训练集上限：全量转换器（含测试词条）跑训练对抽样，衡量「词表内记忆」
    full_conv = LocaleConverter(dicts, direction)
    sample = train[:1000]
    train_hit = sum(1 for s, w in sample if full_conv.convert(s) == w)
    return {
        "direction": direction, "n_train": len(train), "n_test": n,
        "full_hit": full_hit, "full_rate": full_hit / n,
        "char_hit": char_hit, "char_rate": char_hit / n,
        "word_total": word_total, "word_hit": word_hit,
        "word_rate": word_hit / word_total if word_total else 0.0,
        "train_sample": len(sample), "train_rate": train_hit / len(sample) if sample else 0.0,
        "misses": misses,
    }


def print_heldout(result: dict) -> None:
    print(f"\n== held-out 评测（{result['direction']}，"
          f"train {result['n_train']} / test {result['n_test']}，seed={SEED}）==")
    print(f"  整词命中率     {result['full_hit']}/{result['n_test']}"
          f" = {result['full_rate']:.1%}")
    print(f"  纯单字映射     {result['char_hit']}/{result['n_test']}"
          f" = {result['char_rate']:.1%}")
    print(f"  词级对命中率   {result['word_hit']}/{result['word_total']}"
          f" = {result['word_rate']:.1%}"
          f"   （单字映射拼不出、必须靠短语表的 {result['word_total']} 对）")
    print(f"  训练集上限     {result['train_sample']} 对抽样 = "
          f"{result['train_rate']:.1%}（全量词典跑训练对，词表内应≈100%）")
    if result["misses"]:
        print(f"  失误示例（前 {len(result['misses'])} 条）：")
        for src, want, got in result["misses"]:
            print(f"    {src}  期望 {want}  实得 {got}")


# ------------------------------------------------------- 语境依赖测试句

# mode: baseline=应当通过（证明基线能力）| 失败模式标签（预期 FAIL 的原因）
CONTEXT_CASES = [
    # --- 应当通过（词表基线的能力证明）---
    ("cn2tw", "我们需要更多信息。", "我們需要更多資訊。", "baseline",
     "词级替换基本盘：信息→資訊"),
    ("cn2tw", "我们用视频会议开个会吧。", "我們用視訊會議開個會吧。", "baseline",
     "短语守卫：視頻會議→視訊會議 压过 视频→影片"),
    ("cn2tw", "激光打印机很贵。", "雷射印表機很貴。", "baseline",
     "相邻短语组合：激光→雷射 + 打印机→印表機"),
    ("cn2tw", "这个软件是免费的。", "這個軟體是免費的。", "baseline",
     "经典用词：软件→軟體"),
    ("cn2tw", "把视频上传到网络硬盘。", "把影片上傳到網路硬碟。", "baseline",
     "三词组合：影片/網路/硬碟 互不干扰"),
    ("cn2tw", "程序正义必须得到保障。", "程序正義必須得到保障。", "baseline",
     "OpenCC 自带 identity 守卫：程序正義 挡住 程序→程式"),
    ("cn2tw", "物体的质量越大，惯性越大。", "物體的質量越大，慣性越大。", "baseline",
     "物理义「质量」两岸同形，不动才是对的（对照漏转句）"),
    ("tw2cn", "他把檔案放在隨身碟裡。", "他把文件放在U盘里。", "baseline",
     "反向短语：檔案→文件、隨身碟→U盘"),
    ("tw2cn", "我在網路上查到這個資訊。", "我在网络上查到这个信息。", "baseline",
     "反向基本盘：網路→网络、資訊→信息"),
    # --- 预期失败（词表基线的原理性盲区，neural+守卫 要攻克的对象）---
    ("cn2tw", "把文件打印出来。", "把文件列印出來。", "over-convert",
     "过度转换：文件（纸质义）被 一律 转成 檔案（计算机义）；打印→列印 本身成功"),
    ("cn2tw", "投票程序很简单。", "投票程序很簡單。", "over-convert",
     "过度转换：程序（流程义）被 一律 转成 程式，对照守卫句只保了固定搭配"),
    ("cn2tw", "这款手机质量很好。", "這款手機品質很好。", "missing",
     "漏转：质量→品質 不在词表（OpenCC 未收），产品义只能靠语境"),
    ("cn2tw", "智能手机改变了生活。", "智慧型手機改變了生活。", "combo-miss",
     "组合词缺失：智能→智慧 逐词替换得 智慧手機，漏掉「智慧型」构词"),
    ("cn2tw", "地铁出口在哪里？", "捷運出口在哪裡？", "missing",
     "漏转：地铁→捷運 不在词表（同概念不同词，非字面差异）"),
    ("tw2cn", "這個軟體的品質不錯。", "这个软件的质量不错。", "missing-rev",
     "反向漏转：品質 逐字简化成 品质，回不到大陆的 质量"),
    ("tw2cn", "夜市賣的土豆很香。", "夜市卖的花生很香。", "homograph",
     "同形异义：台湾口语「土豆」指花生，词表没有语境判据（大陸土豆=馬鈴薯）"),
]


def eval_context(dicts) -> list[dict]:
    convs = {d: LocaleConverter(dicts, d) for d in DIRECTIONS}
    results = []
    for direction, text, expect, mode, note in CONTEXT_CASES:
        got = convs[direction].convert(text)
        results.append({"direction": direction, "text": text, "expect": expect,
                        "got": got, "mode": mode, "note": note,
                        "ok": got == expect})
    return results


def print_context(results: list[dict]) -> None:
    print("\n== 语境依赖测试句（人工标注，覆盖词表基线的盲区）==")
    for i, r in enumerate(results, 1):
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{i:2d}] {mark} {r['direction']} {r['text']}")
        if not r["ok"]:
            print(f"           期望: {r['expect']}")
            print(f"           实得: {r['got']}")
        print(f"           [{r['mode']}] {r['note']}")
    n_pass = sum(r["ok"] for r in results)
    print(f"\n  通过 {n_pass}/{len(results)}"
          f"（设计口径：baseline 全过、已知盲区全暴露）")
    fail_modes: dict[str, int] = {}
    for r in results:
        if not r["ok"]:
            fail_modes[r["mode"]] = fail_modes.get(r["mode"], 0) + 1
    if fail_modes:
        print("  失败模式分布：" +
              "，".join(f"{m}×{c}" for m, c in sorted(fail_modes.items())))


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=PAIRS_JSON)
    ap.add_argument("--skip-context", action="store_true")
    args = ap.parse_args()

    dicts = load_dicts(args.data)
    cn2tw, tw2cn, meta = load_pairs(args.data)

    seed_note = "（种子词典）" if meta.get("fallback_seed") else ""
    print(f"数据：cn→tw {len(cn2tw)} 对，tw→cn {len(tw2cn)} 对{seed_note}")

    for pairs, direction in ((cn2tw, "cn2tw"), (tw2cn, "tw2cn")):
        print_heldout(eval_direction(dicts, pairs, direction))

    if not args.skip_context:
        print_context(eval_context(dicts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
