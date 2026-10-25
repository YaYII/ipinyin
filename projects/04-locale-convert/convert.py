#!/usr/bin/env python3
"""中文地区用语双向转换 CLI（词表基线）：zh-CN ↔ zh-TW。

思路与仓库「普→粤」管线同构：**seq2seq 之外的实证守卫** —— 只做词表里有
实证的替换，替换来自 OpenCC 社区词典（人工审校），不猜。

管线（复刻 OpenCC 的字典链，两级最长匹配）：
  cn→tw: [STPhrases + STCharacters] → [TWPhrases + TWVariants]
  tw→cn: [TWPhrasesRev (+TWVariantsRevPhrases)] → [TSPhrases + TSCharacters]

每级内部：短语最长匹配优先，命不中再走单字映射，都没有就原样保留。

「守卫」雏形 —— 歧义报告（--report）：
  1. 一对多映射：同一源词有多个目标（OpenCC 行内多值），转换只取首选；
  2. 长度冲突：长短语覆盖短信语后，覆盖段的实际转换 ≠ 短语自身映射
     （如 程序→程式 被 程序正義→程序正義 覆盖 —— 这正是需要的守卫，
     但不被察觉的冲突就是隐患清单）；
  3. 往返不对称：cn→tw 与 tw→cn 不是互逆（一对多收敛/发散的直接后果）。

数据：data/opencc_pairs.json（fetch_opencc.py 产出）；缺失时退回
seed_data.py 的种子词典（离线可跑，规模小得多）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_JSON = os.path.join(PROJECT_DIR, "data", "opencc_pairs.json")

# 参与各级管线的角色 → (短语角色列表, 单字角色列表)。
# 短语角色按 OpenCC 配置的 short_circuit 顺序排列：同一源词冲突时，
# 列表靠后的角色（地区用词表）胜出，与官方字典组语义一致。
#   s2twp: [TWPhrases, TWVariantsPhrases, TWVariants]
#   tw2sp: [TWPhrasesRev, TWVariantsRevPhrases] → [TSPhrases, TSCharacters]
PIPELINE = {
    "cn2tw": [
        (("st_phrases",), ("st_chars",)),
        (("tw_variants_phrases", "tw_phrases"), ("tw_variants",)),
    ],
    "tw2cn": [
        (("tw_variants_rev_phrases", "tw_phrases_rev"), ()),
        (("ts_phrases",), ("ts_chars",)),
    ],
}
DIRECTIONS = ("cn2tw", "tw2cn")
ROLES = ("st_phrases", "st_chars", "tw_phrases", "tw_variants",
         "tw_phrases_rev", "ts_phrases", "ts_chars",
         "tw_variants_phrases", "tw_variants_rev_phrases")


# ---------------------------------------------------------------- 数据装载

@dataclass
class DictSet:
    """原始词典集：每个角色 {源: [首选目标, 备选...]}。"""
    dicts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, dict], meta: dict | None = None) -> "DictSet":
        dicts = {}
        for role in ROLES:
            entries = raw.get(role) or {}
            clean: dict[str, list[str]] = {}
            for k, v in entries.items():
                targets = [t for t in (v if isinstance(v, list) else [v]) if t]
                if k and targets:
                    clean[k] = targets
            dicts[role] = clean
        return cls(dicts, meta or {})


def load_dicts(path: str = PAIRS_JSON) -> DictSet:
    """加载 opencc_pairs.json；缺失/损坏时退回种子词典。"""
    if os.path.exists(path):
        try:
            obj = json.load(open(path, encoding="utf-8"))
            return DictSet.from_raw(obj.get("dicts", {}), obj.get("meta", {}))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] {path} 读取失败（{exc}），退回种子词典", file=sys.stderr)
    from seed_data import SEED_DICTS
    return DictSet.from_raw(SEED_DICTS, {"source": "seed", "reason": "no data file"})


# ---------------------------------------------------------------- 核心转换

def flatten_first(role_dict: dict[str, list[str]]) -> dict[str, str]:
    """取每个源词的首选目标（OpenCC 的约定：行内第一个值最优）。"""
    return {k: v[0] for k, v in role_dict.items()}


def convert_pass(text: str, phrases: dict[str, str], chars: dict[str, list[str]]) -> str:
    """单级转换：短语最长匹配 > 单字映射 > 原样保留。

    chars 的值保留列表（歧义报告要统计一对多），转换只取首选。
    """
    out: list[str] = []
    i, n = 0, len(text)
    max_len = max((len(k) for k in phrases), default=0)
    while i < n:
        hit = None
        for length in range(min(max_len, n - i), 1, -1):
            seg = text[i:i + length]
            if seg in phrases:
                hit = phrases[seg]
                i += length
                break
        if hit is None:
            targets = chars.get(text[i])
            out.append(targets[0] if targets else text[i])
            i += 1
        else:
            out.append(hit)
    return "".join(out)


class LocaleConverter:
    """两级字典链转换器。use_phrases=False 退化为纯单字映射（评测基线用）。"""

    def __init__(self, dicts: DictSet, direction: str, use_phrases: bool = True):
        if direction not in PIPELINE:
            raise ValueError(f"未知方向 {direction!r}，可选 {DIRECTIONS}")
        self.direction = direction
        self.stages: list[tuple[dict[str, str], dict[str, list[str]]]] = []
        for phrase_roles, char_roles in PIPELINE[direction]:
            phrases: dict[str, str] = {}
            if use_phrases:
                for role in phrase_roles:
                    phrases.update(flatten_first(dicts.dicts.get(role, {})))
            chars: dict[str, list[str]] = {}
            for role in char_roles:
                chars.update(dicts.dicts.get(role, {}))
            self.stages.append((phrases, chars))

    def convert(self, text: str) -> str:
        for phrases, chars in self.stages:
            if phrases or chars:
                text = convert_pass(text, phrases, chars)
        return text


# ---------------------------------------------------------------- 歧义报告

def collect_multi_target(dicts: DictSet) -> dict[str, list[tuple[str, list[str]]]]:
    """一对多映射：OpenCC 行内多值（第一个是首选，其余被转换器舍弃）。"""
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for role in ROLES:
        hits = [(k, v) for k, v in dicts.dicts.get(role, {}).items() if len(v) > 1]
        if hits:
            out[role] = hits
    return out


def shadow_conflicts(phrases: dict[str, str]) -> tuple[list[tuple[str, str, str, str]], int]:
    """长度冲突：短语 A 是短语 B 的子串，但 B 的整体转换结果里
    找不到 A 的映射 —— 长词条覆盖短词条且语义不一致。

    返回 (冲突清单, 一致的覆盖次数)。一致覆盖（軟件→軟體 被
    軟件工程→軟體工程 覆盖且包含 軟體）是词典设计使然，只计数。
    """
    keys = set(phrases)
    conflicts: list[tuple[str, str, str, str]] = []
    consistent = 0
    for long_key in phrases:
        L = len(long_key)
        if L < 3:
            continue
        converted: str | None = None
        for width in range(2, L):
            for start in range(0, L - width + 1):
                sub = long_key[start:start + width]
                if sub not in keys:
                    continue
                if converted is None:
                    converted = convert_pass(long_key, phrases, {})
                if phrases[sub] in converted:
                    consistent += 1
                else:
                    conflicts.append((sub, phrases[sub], long_key, converted))
    return conflicts, consistent


def round_trip_asymmetry(dicts: DictSet, cn2tw_pairs: dict[str, str],
                         limit: int = 200) -> list[tuple[str, str, str]]:
    """往返不对称：cn→tw 之后，用 tw→cn 转换器回不去原词。

    一对多收敛的代价：视频→影片 挤掉了 視訊 的反向路径，反向词典
    只能选一个赢家；另一些词（壁紙→桌布）反向根本没有词条。
    """
    back = LocaleConverter(dicts, "tw2cn")
    bad: list[tuple[str, str, str]] = []
    for cn, tw in cn2tw_pairs.items():
        got = back.convert(tw)
        if got != cn:
            bad.append((cn, tw, got))
            if len(bad) >= limit:
                break
    return bad


def build_report(dicts: DictSet, pairs: dict | None = None) -> dict:
    """汇总三类歧义证据，供 CLI 展示与 README 引用。"""
    multi = collect_multi_target(dicts)
    shadows = {}
    for direction, stages in PIPELINE.items():
        for level, (phrase_roles, _) in enumerate(stages, 1):
            merged: dict[str, str] = {}
            for role in phrase_roles:
                merged.update(flatten_first(dicts.dicts.get(role, {})))
            if len(merged) < 3:
                continue
            conflicts, consistent = shadow_conflicts(merged)
            shadows[f"{direction}#L{level}"] = {
                "keys": len(merged), "conflicts": conflicts, "consistent": consistent,
            }
    report: dict = {"multi_target": multi, "shadow": shadows, "round_trip": []}
    if pairs:
        report["round_trip"] = round_trip_asymmetry(
            dicts, pairs.get("cn2tw_pairs", {}))
    return report


def format_report(report: dict, example_limit: int = 12) -> str:
    lines: list[str] = []
    lines.append("== 歧义报告（词表基线的「守卫」清单）==")

    lines.append("\n[1] 一对多映射（行内多值，转换取首选、舍弃备选）")
    total_multi = sum(len(v) for v in report["multi_target"].values())
    if not total_multi:
        lines.append("    （无）")
    for role, hits in report["multi_target"].items():
        lines.append(f"    {role}: {len(hits)} 条")
        for k, v in hits[:example_limit]:
            lines.append(f"      {k} → {' | '.join(v)}")
        if len(hits) > example_limit:
            lines.append(f"      … 共 {len(hits)} 条")

    lines.append("\n[2] 长度冲突（长短语覆盖短信语，覆盖段转换 ≠ 短语映射）")
    found = False
    for name, info in report["shadow"].items():
        conflicts = info["conflicts"]
        lines.append(f"    {name}: 词条 {info['keys']}，冲突 {len(conflicts)}，"
                     f"一致覆盖 {info['consistent']}")
        for sub, sub_t, long_key, long_t in conflicts[:example_limit]:
            lines.append(f"      『{long_key}』→『{long_t}』覆盖『{sub}』→『{sub_t}』")
        if len(conflicts) > example_limit:
            lines.append(f"      … 共 {len(conflicts)} 条")
        found = found or bool(conflicts)
    if not found:
        lines.append("    （无冲突）")

    lines.append("\n[3] 往返不对称（cn→tw 后 tw→cn 回不到原词）")
    rt = report.get("round_trip", [])
    if not rt:
        lines.append("    （无）")
    else:
        lines.append(f"    共 {len(rt)} 例（示例前 {min(example_limit, len(rt))} 条）")
        for cn, tw, back in rt[:example_limit]:
            lines.append(f"      {cn} → {tw} → {back}")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="中文地区用语转换 zh-CN ↔ zh-TW（词表基线）")
    ap.add_argument("direction", nargs="?", choices=DIRECTIONS, help="转换方向")
    ap.add_argument("--text", help="待转换文本；缺省读 stdin")
    ap.add_argument("--report", action="store_true", help="输出歧义报告")
    ap.add_argument("--chars-only", action="store_true", help="只用单字映射（不要短语表）")
    ap.add_argument("--data", default=PAIRS_JSON, help="词典 JSON 路径")
    args = ap.parse_args(argv)

    dicts = load_dicts(args.data)
    pairs = None
    if os.path.exists(args.data):
        try:
            pairs = json.load(open(args.data, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pairs = None

    if args.report:
        print(format_report(build_report(dicts, pairs)))
        if not args.direction:
            return 0

    if not args.direction:
        if args.report:
            return 0
        ap.error("需要 direction（cn2tw / tw2cn）或 --report")

    conv = LocaleConverter(dicts, args.direction, use_phrases=not args.chars_only)
    text = args.text if args.text is not None else sys.stdin.read()
    print(conv.convert(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
