#!/usr/bin/env python3
"""把抓下来的个人文档去重、清洗，汇总成一份可训练的语料。

背景：从文档服务批量导出的正文会有重复（同一份文档导出多次、多版本并存），
也有含 API 密钥的文档（不能进语料）。这个脚本按内容哈希去重、清掉控制字符、
剔除太短的碎片，把干净文本写进 `_clean/`。

用法：
  python3 tools/merge_personal_corpus.py                 # 默认 vendor/personal
  python3 tools/merge_personal_corpus.py 目录 --min 300
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.t2s import Converter  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.t2s import Converter  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")
# 含这些内容的文档是密钥/请求体，不是语料
SECRET = re.compile(r"APPCODE|X-DSIWS-TOKEN|Authorization:|password|secret|api[_-]?key", re.I)


def clean(text: str) -> str:
    text = text.replace("\x0b", "\n").replace("\x0c", "\n")
    text = re.sub(r"[\x00-\x08\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=os.path.join(ROOT, "vendor", "personal"))
    ap.add_argument("--min", type=int, default=300, help="最少中文字数")
    ap.add_argument("--keep-traditional", action="store_true",
                    help="不做繁→简归一（默认会转：简中语言模型只该吃简体）")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out_dir = os.path.join(src, "_clean")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.txt")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(src, "*.txt")))
    if not files:
        print(f"{src} 里没有 .txt，先把文档导出来放这里", file=sys.stderr)
        return 1

    conv = None if args.keep_traditional else Converter.load(
        os.path.join(ROOT, "data", "s2t.marshal"))

    conv = None if args.keep_traditional else Converter.load(
        os.path.join(ROOT, "data", "s2t.marshal"))

    kept: dict[str, tuple[str, str, int]] = {}
    dropped: list[tuple[str, str]] = []
    for path in files:
        text = clean(open(path, encoding="utf-8", errors="ignore").read())
        if conv is not None:
            # 繁体只是显示层的映射，简中模型的语料必须是简体；
            # 否则 8.6% 的繁体字会占掉三元组容量，而引擎永远见不到它们。
            text = conv.to_simplified(text)
        if conv is not None:
            # 繁体只是显示层的映射，简中模型的语料必须是简体；
            # 否则 8.6% 的繁体字会占掉三元组容量，而引擎永远见不到它们。
            text = conv.to_simplified(text)
        cjk = len(CJK.findall(text))
        if cjk < args.min:
            dropped.append((path, f"中文 {cjk} < {args.min}"))
            continue
        if SECRET.search(text):
            dropped.append((path, "含密钥/凭据"))
            continue
        digest = hashlib.md5(re.sub(r"\s+", "", text).encode()).hexdigest()
        if digest in kept:
            dropped.append((path, f"与 {os.path.basename(kept[digest][0])} 重复"))
            continue
        kept[digest] = (path, text, cjk)

    total = 0
    for path, text, cjk in kept.values():
        with open(os.path.join(out_dir, os.path.basename(path)), "w", encoding="utf-8") as fh:
            fh.write(text)
        total += cjk

    print(f"{len(files)} 份 → 保留 {len(kept)} 份 / 中文 {total} 字 → {out_dir}")
    if dropped:
        print(f"丢弃 {len(dropped)} 份：")
        for path, why in dropped[:15]:
            print(f"  {os.path.basename(path)[:46]:<48} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
