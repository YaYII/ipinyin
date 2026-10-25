#!/usr/bin/env python3
"""从 OpenCC 仓库拉取地区词典，解析成 (cn, tw) 词对 JSON。

数据源：github.com/BYVoid/OpenCC（raw.githubusercontent.com，经 curl 下载）。
目标文件（master 分支，data/dictionary/）：
  STCharacters / STPhrases      简→繁 字/短语
  TSCharacters / TSPhrases      繁→简 字/短语
  TWPhrases                     大陆用语→台湾用语（社区审校的用词差异）
  TWPhrasesRev                  台湾用语→大陆用语
  TWVariants / TWVariantsPhrases / TWVariantsRevPhrases  台湾正体用字

任务清单里点名的 TWPhrasesIT/Name/Other 是旧版（≤1.1.9）的拆分文件，
上游已合并为 TWPhrases.txt：脚本仍会尝试这三个文件（记录可达性），
master 拉取失败时回退拼接它们。上游从未发布过 TSPhrasesIT.txt，
与本任务「繁→简短语」对应的是 TSPhrases.txt + TWPhrasesRev.txt。

失败策略：每个下载最多重试 2 次；关键文件全部失败时退回 seed_data.py
的种子词典（meta.fallback_seed=true），README 记录。

产出 data/opencc_pairs.json：
  dicts         各角色 {源词: [首选目标, 备选...]}
  cn2tw_pairs   合成的简→台词对（最长匹配管线的直接输入口径）
  tw2cn_pairs   合成的台→简词对
  meta          来源、规模、冲突统计
"""
from __future__ import annotations

import argparse
import json
import marshal
import os
import subprocess
import sys
import time
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from convert import DictSet, LocaleConverter, flatten_first, convert_pass  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_DIR))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
OUT_JSON = os.path.join(PROJECT_DIR, "data", "opencc_pairs.json")
BASE = "https://raw.githubusercontent.com/BYVoid/OpenCC"

# 角色 → [(ref, 文件名)]，按序尝试，第一个成功者生效
CRITICAL_ROLES = {"st_chars", "st_phrases", "ts_chars", "ts_phrases",
                  "tw_phrases", "tw_phrases_rev", "tw_variants"}
FILE_PLAN: dict[str, list[tuple[str, str]]] = {
    "st_chars":    [("master", "STCharacters.txt")],
    "st_phrases":  [("master", "STPhrases.txt")],
    "ts_chars":    [("master", "TSCharacters.txt")],
    "ts_phrases":  [("master", "TSPhrases.txt")],
    "tw_phrases":  [("master", "TWPhrases.txt")],
    "tw_phrases_rev": [("master", "TWPhrasesRev.txt")],
    "tw_variants": [("master", "TWVariants.txt")],
    "tw_variants_phrases": [("master", "TWVariantsPhrases.txt")],
    "tw_variants_rev_phrases": [("master", "TWVariantsRevPhrases.txt")],
}
# 任务点名的旧版拆分文件：始终尝试（记录可达性）；master 的 TWPhrases.txt
# 失败时才并入 tw_phrases。TSPhrasesIT.txt 上游从未存在，不设下载项。
LEGACY_TW = [(f"ver.1.1.9", f"TWPhrases{name}.txt")
             for name in ("IT", "Name", "Other")]


def curl_fetch(url: str, dest: str, retries: int = 2, timeout: int = 60) -> bool:
    """curl 下载（本环境 urllib 出网受限，curl 可用）；失败重试至多 retries 次。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries + 1):
        proc = subprocess.run(
            ["curl", "-sSfL", "--max-time", str(timeout), "-o", dest, url],
            capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        time.sleep(1)
    if os.path.exists(dest):
        os.remove(dest)
    return False


def parse_dictionary(path: str) -> dict[str, list[str]]:
    """OpenCC 词典格式：key\tv1 v2 ...（# 开头为注释，行内第一个值是首选）。"""
    entries: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or "\t" not in line:
                continue
            key, _, rest = line.rstrip("\n").partition("\t")
            targets = rest.split()
            if key and targets:
                entries.setdefault(key, targets)
    return entries


def st_chars_from_marshal() -> dict[str, list[str]]:
    """回退：复用仓库 data/s2t.marshal（tools/build_s2t.py 的产物，只读）。"""
    path = os.path.join(REPO_ROOT, "data", "s2t.marshal")
    if not os.path.exists(path):
        return {}
    raw = marshal.load(open(path, "rb"))
    return {k: [v.replace("爲", "為")] for k, v in raw.items()}


def normalize_st_chars(entries: dict[str, list[str]]) -> dict[str, list[str]]:
    """与 tools/build_s2t.py 同口径：统一用「為」。"""
    return {k: [t.replace("爲", "為") for t in v] for k, v in entries.items()}


def fetch_all(retries: int) -> tuple[dict[str, dict[str, list[str]]], dict]:
    """下载并解析全部角色。返回 (角色词典, 来源元数据)。"""
    raw: dict[str, dict[str, list[str]]] = {}
    sources: dict[str, dict] = {}
    attempts: list[dict] = []

    for role, plan in FILE_PLAN.items():
        got = False
        for ref, name in plan:
            url = f"{BASE}/{ref}/data/dictionary/{name}"
            dest = os.path.join(RAW_DIR, f"{ref}_{name}")
            ok = curl_fetch(url, dest, retries=retries)
            attempts.append({"role": role, "ref": ref, "file": name,
                             "status": "ok" if ok else "failed"})
            if ok:
                raw[role] = parse_dictionary(dest)
                sources[role] = {"ref": ref, "file": name,
                                 "bytes": os.path.getsize(dest),
                                 "lines": sum(1 for _ in open(dest, encoding="utf-8"))}
                got = True
                break
        if not got:
            attempts.append({"role": role, "status": "all-failed"})

    # 任务点名的旧版拆分文件：尝试并记录；仅在 master TWPhrases 失败时并入
    legacy_ok = []
    for ref, name in LEGACY_TW:
        url = f"{BASE}/{ref}/data/dictionary/{name}"
        dest = os.path.join(RAW_DIR, f"{ref}_{name}")
        ok = curl_fetch(url, dest, retries=retries)
        attempts.append({"role": "tw_phrases(legacy)", "ref": ref, "file": name,
                         "status": "ok" if ok else "failed"})
        if ok:
            legacy_ok.append(parse_dictionary(dest))
    if "tw_phrases" not in raw and legacy_ok:
        merged: dict[str, list[str]] = {}
        for part in legacy_ok:
            for k, v in part.items():
                merged.setdefault(k, v)
        raw["tw_phrases"] = merged
        sources["tw_phrases"] = {"ref": "ver.1.1.9",
                                 "file": "TWPhrasesIT+Name+Other.txt",
                                 "bytes": sum(len(p) for p in legacy_ok)}

    # 单字映射兜底：仓库 s2t.marshal
    if "st_chars" not in raw:
        alt = st_chars_from_marshal()
        if alt:
            raw["st_chars"] = alt
            sources["st_chars"] = {"ref": "local", "file": "data/s2t.marshal",
                                   "bytes": os.path.getsize(os.path.join(REPO_ROOT, "data", "s2t.marshal"))}
            attempts.append({"role": "st_chars", "status": "fallback-marshal"})

    return raw, {"sources": sources, "attempts": attempts}


def derive_pairs(dicts: DictSet) -> tuple[dict[str, str], dict[str, str], dict]:
    """合成 (cn, tw) 双向词对 —— 与转换器运行时同口径（两级管线）。

    cn→tw：TWPhrases 词条 (t_src→tw) 的简体键 = 对 t_src 做 t→s 全程转换；
           再用 STPhrases 词条补空隙（目标 = 对繁体候选做 t2tw 级转换）。
    tw→cn：TWPhrasesRev 词条 (tw→t) 的简体目标 = 对 t 做 t→s 全程转换；
           再用 TSPhrases 词条补空隙。
    """
    d = dicts.dicts
    tw_phr = flatten_first(d.get("tw_phrases", {}))
    tw_var = d.get("tw_variants", {})
    ts_phr = flatten_first(d.get("ts_phrases", {}))
    ts_chr = d.get("ts_chars", {})

    def t2s(text: str) -> str:                       # 繁→简 全程
        return convert_pass(text, ts_phr, ts_chr)

    def t2tw(text: str) -> str:                      # 繁→台湾用语 级
        return convert_pass(text, tw_phr, tw_var)

    cn2tw: dict[str, str] = {}
    collisions = 0
    # 主源：TWPhrases（用词差异本体）。键的简体形 = t2s(繁体键)。
    for t_src, targets in d.get("tw_phrases", {}).items():
        cn2tw[t2s(t_src)] = targets[0]
    # 补源：STPhrases 简体键 → 其繁体候选再过 t2tw 级
    for s, targets in d.get("st_phrases", {}).items():
        target = t2tw(targets[0])
        if s in cn2tw:
            if cn2tw[s] != target:
                collisions += 1                      # 同键不同目标：留 TW 派生
        elif target != s:
            cn2tw[s] = target

    tw2cn: dict[str, str] = {}
    for tw_src, targets in d.get("tw_phrases_rev", {}).items():
        tw2cn[tw_src] = t2s(targets[0])
    for t, targets in d.get("ts_phrases", {}).items():
        if t not in tw2cn and targets[0] != t:
            tw2cn[t] = targets[0]

    stats = {"derived_collisions": collisions}
    return cn2tw, tw2cn, stats


def size_breakdown(dicts: DictSet, cn2tw: dict[str, str]) -> dict:
    """词对分类：目标可由纯单字映射得到的算「字级」，其余是「词级（用词）」。"""
    chars_only = LocaleConverter(dicts, "cn2tw", use_phrases=False)
    n_word = 0
    for cn, tw in cn2tw.items():
        if len(cn) >= 2 and chars_only.convert(cn) != tw:
            n_word += 1
    return {"word_level": n_word, "char_only": len(cn2tw) - n_word}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", type=int, default=2, help="每个下载的重试次数")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    t0 = time.time()
    raw, meta = fetch_all(args.retries)
    missing_critical = CRITICAL_ROLES - set(raw)

    if missing_critical:
        print(f"[warn] 关键文件缺失：{sorted(missing_critical)}，退回本地种子词典")
        from seed_data import SEED_DICTS
        raw = dict(SEED_DICTS)
        meta["sources"] = {"seed": {"file": "seed_data.py"}}
        meta["fallback_seed"] = True
    else:
        meta["fallback_seed"] = False

    if "st_chars" in raw:
        raw["st_chars"] = normalize_st_chars(raw["st_chars"])

    dicts = DictSet.from_raw(raw, meta)
    cn2tw, tw2cn, stats = derive_pairs(dicts)
    breakdown = size_breakdown(dicts, cn2tw)

    counts = {}
    for role, entries in dicts.dicts.items():
        multi = sum(1 for v in entries.values() if len(v) > 1)
        counts[role] = {"entries": len(entries), "multi_target": multi}
    meta.update({
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "counts": counts,
        "pairs": {"cn2tw": len(cn2tw), "tw2cn": len(tw2cn), **stats, **breakdown},
    })

    obj = {"meta": meta, "dicts": dicts.dicts,
           "cn2tw_pairs": cn2tw, "tw2cn_pairs": tw2cn}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)
    size_mb = os.path.getsize(args.out) / 1024 / 1024

    print(f"来源（{len(meta['sources'])} 个角色）：")
    for role, src in meta["sources"].items():
        extra = f", {src['bytes']}B" if "bytes" in src else ""
        print(f"  {role:26s} {src['ref']}/{src['file']}{extra}")
    failed = [a for a in meta["attempts"] if a["status"] != "ok"]
    if failed:
        print("下载失败（已按策略回退）：")
        for a in failed:
            print(f"  {a.get('ref', '')}/{a.get('file', a['role'])}: {a['status']}")
    print(f"\n词典规模：")
    for role, c in counts.items():
        flag = f"（含一对多 {c['multi_target']}）" if c["multi_target"] else ""
        print(f"  {role:26s} {c['entries']:6d} 条{flag}")
    print(f"\n合成词对：cn→tw {len(cn2tw)} 对（词级 {breakdown['word_level']} / "
          f"字级 {breakdown['char_only']}），tw→cn {len(tw2cn)} 对，"
          f"派生冲突 {stats['derived_collisions']}")
    print(f"写出 {args.out}（{size_mb:.2f} MB"
          + ("，超过 2MB，入库时需精简" if size_mb > 2 else "") + "）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
