#!/usr/bin/env python3
"""合并 pycorrector 的混淆集进项目⑤的 confusion.json。

pycorrector（github.com/shibing624/pycorrector）数据：
  vendor/pycorrector_data/same_pinyin.txt —— 每行「字<TAB>同音候选…」
  vendor/pycorrector_data/same_stroke.txt —— 每行「字<TAB>形近候选…」

合并规则：
  - 已有条目以本仓库为准（词库挖的同音 + 手写形近对质量更可控），
    pycorrector 只**补充**新字/新候选；
  - 候选列表封顶（同音 15 / 形近 8）：baseline.py 的 homophone 走 topk 截断，
    shape 不截断——不封顶会拖慢评测还引入脏候选；
  - 输出 data/confusion_merged.json，原 confusion.json 不动（A/B 对比用）。

用法：python3 merge_confusion.py [--pinyin 路径] [--stroke 路径]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONF = os.path.join(HERE, "data", "confusion.json")
OUT = os.path.join(HERE, "data", "confusion_merged.json")
DEFAULT_PINYIN = os.path.join(ROOT, "vendor", "pycorrector_data", "same_pinyin.txt")
DEFAULT_STROKE = os.path.join(ROOT, "vendor", "pycorrector_data", "same_stroke.txt")

TOP_HOMO = 15       # 同音候选封顶（baseline 的 topk 默认更小，这里是上限）
TOP_SHAPE = 8       # 形近候选封顶（baseline 不截断 shape，这里必须控）


def load_pycorrector(path: str, cap: int) -> dict[str, list[str]]:
    """解析 pycorrector 的「字<TAB>候选…」行格式（# 开头是注释）。"""
    table: dict[str, list[str]] = {}
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 2:
                continue
            char, cands = parts[0], parts[1:]
            # 只要基本区汉字（U+4E00–9FFF），扩展区生僻字对纠错是噪声
            if not ("\u4e00" <= char <= "\u9fff"):
                continue
            clean = [c for c in cands if "\u4e00" <= c <= "\u9fff"][:cap]
            if clean:
                table[char] = clean
    return table


def merge(base: dict[str, list[str]], extra: dict[str, list[str]], cap: int
          ) -> tuple[dict[str, list[str]], int, int]:
    """extra 补充进 base：已有键追加新候选并封顶；新键直接收；**最后对称化**。

    对称化是关键：pycorrector 的表是单向的（字→同音字），实测新增 6093 个
    同音对里只有 1.2% 带反向映射——而纠错器查的是「错字→候选」，
    缺反向的对照永远纠不回来。同音/形近关系本身对称（本仓库自建表
    从拼音键组挖出，天然对称），补上反向是修复数据而非造数据。
    """
    out = {k: list(v) for k, v in base.items()}
    new_keys = new_cands = 0
    for ch, cands in extra.items():
        if ch in out:
            seen = set(out[ch])
            for c in cands:
                if c not in seen and c != ch:
                    out[ch].append(c)
                    new_cands += 1
        else:
            out[ch] = [c for c in cands if c != ch]
            new_keys += 1
    # 对称化：X 的候选 W → W 的候选里也要有 X（封顶时旧候选优先，保稳定）
    for ch in list(out.keys()):
        for w in list(out[ch]):
            bucket = out.setdefault(w, [])
            if ch not in bucket and ch != w:
                bucket.append(ch)
                new_cands += 1
    for ch in out:
        out[ch] = out[ch][:cap]
    return out, new_keys, new_cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinyin", default=DEFAULT_PINYIN)
    ap.add_argument("--stroke", default=DEFAULT_STROKE)
    args = ap.parse_args()

    conf = json.load(open(CONF, encoding="utf-8"))
    pinyin = load_pycorrector(args.pinyin, TOP_HOMO)
    stroke = load_pycorrector(args.stroke, TOP_SHAPE)
    print(f"pycorrector 同音表 {len(pinyin)} 字，形近表 {len(stroke)} 字")

    homo, hk, hc = merge(conf["homophone"], pinyin, TOP_HOMO)
    shape, sk, sc = merge(conf["shape"], stroke, TOP_SHAPE)
    print(f"同音：新增 {hk} 字 / {hc} 候选 → {len(homo)} 字")
    print(f"形近：新增 {sk} 字 / {sc} 候选 → {len(shape)} 字")

    merged = {
        "homophone": homo,
        "shape": shape,
        "shape_pairs": conf.get("shape_pairs", []),
        "meta": {
            "merged_from": ["repo", "pycorrector:same_pinyin.txt", "pycorrector:same_stroke.txt"],
            "repo_homophone": len(conf["homophone"]),
            "repo_shape": len(conf["shape"]),
            "pycorrector_pinyin_chars": len(pinyin),
            "pycorrector_stroke_chars": len(stroke),
            "top_homo": TOP_HOMO,
            "top_shape": TOP_SHAPE,
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
