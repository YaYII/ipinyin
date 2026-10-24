#!/usr/bin/env python3
"""简中整句基准：给一串拼音，第一候选是不是你想打的那句话。

这是输入法的核心指标 —— 词级 Top1 再高，用户整句打出来错一个字就要重打。

题面怎么造：取 held-out 的中文句子，用词库把每个词换成拼音，拼成一串。
  （多音字/未登录词会导致题面本身不合法，这些句子会被跳过）
判分：引擎第一候选 == 原句？出现在前 9 个里？

用法：python3 tools/eval_sentence.py [--n 300] [--lm path] [--pairs vendor/yue_pairs.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402

DATA = os.environ.get("IPINYIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".local/share/ipinyin/data")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")


def build_word_map(lexicon: Lexicon) -> dict[str, str]:
    """词 → 拼音键。多音字取最高频那个键。"""
    best: dict[str, tuple[int, str]] = {}
    for key in lexicon.iter_keys():
        for word, freq in lexicon.entries(key):
            if len(word) < 1:
                continue
            old = best.get(word)
            if old is None or freq > old[0]:
                best[word] = (freq, key)
    return {w: k for w, (_, k) in best.items()}


def to_pinyin(sentence: str, word_map: dict[str, str], max_word: int = 6,
              lexicon: Lexicon | None = None) -> tuple[str, list[str]] | None:
    """最长匹配切成词再拼拼音；切不出来返回 None。

    lexicon 非 None 时做「自洽」检查：切出来的每个词必须是它自己拼音键的首选词。
    否则题面本身不成立 —— 比如「不知道埞」会被切成「不知」+「道」，
    拼音串 buzhidao 的合法解就是「不知道」或「不知+到底」，判引擎错是冤枉它。
    """
    out: list[str] = []
    words: list[str] = []
    i = 0
    size = len(sentence)
    while i < size:
        for length in range(min(max_word, size - i), 0, -1):
            piece = sentence[i:i + length]
            key = word_map.get(piece)
            if key:
                out.append(key)
                words.append(piece)
                i += length
                break
        else:
            return None
    if lexicon is not None:
        for word, key in zip(words, out):
            entries = lexicon.entries(key)
            if not entries or entries[0][0] != word:
                return None
    return "".join(out), words


def load_sentences(pairs_path: str, limit: int = 20000) -> list[str]:
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    random.seed(0)
    random.shuffle(pairs)
    seen: set[str] = set()
    out: list[str] = []
    for p in pairs[:limit]:
        zh = (p.get("zh") or "").strip()
        if not zh or zh in seen:
            continue
        if not all("\u4e00" <= ch <= "\u9fff" for ch in zh):
            continue
        seen.add(zh)
        out.append(zh)
    return out


def load_sentences_from_text(path: str, min_len: int = 4, max_len: int = 40) -> list[str]:
    """从纯文本/目录里切句子：按标点断句，只保留 4~40 字的中文句子。"""
    import glob

    files: list[str] = []
    if os.path.isdir(path):
        for pattern in ("*.txt", "*.md"):
            files.extend(sorted(glob.glob(os.path.join(path, pattern))))
    else:
        files = [path]

    out: list[str] = []
    seen: set[str] = set()
    for name in files:
        with open(name, encoding="utf-8") as fh:
            text = fh.read()
        for piece in re.split(r"[^\u4e00-\u9fff]+", text):
            piece = piece.strip()
            if not piece or not (min_len <= len(piece) <= max_len):
                continue
            if piece in seen:
                continue
            seen.add(piece)
            out.append(piece)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--text", default="", help="改用纯文本（文件或目录）当题库：拿你自己的文档测")
    ap.add_argument("--n", type=int, default=300, help="每桶评测句数")
    ap.add_argument("--lm", default="", help="可选：字级语言模型（data/lm_char.marshal），用于重排")
    ap.add_argument("--lm-weight", type=float, default=0.4, help="上下文分权重 λ（通用最优 0.8，领域最优 0.2，0.4 是折中）")
    args = ap.parse_args()

    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    word_map = build_word_map(lexicon)
    ime = ImeState(lexicon)
    lm = None
    if args.lm:
        from ipinyin.lm import CharLM

        lm = CharLM.load(args.lm)
        if lm is None:
            print(f"语言模型加载失败：{args.lm}", file=sys.stderr)
            return 1
        ime.attach_lm(lm, args.lm_weight)
        print(f"已挂语言模型 {args.lm}（{len(lm)} 字表，λ={args.lm_weight}）")

    if args.text:
        sentences = load_sentences_from_text(args.text)
        print(f"题库：{args.text}（{len(sentences)} 句）")
    else:
        sentences = load_sentences(args.pairs)
    buckets: dict[str, list[str]] = {"2-4字": [], "5-7字": [], "8-15字": [], "16+字": []}
    for s in sentences:
        n = len(s)
        key = "2-4字" if n <= 4 else "5-7字" if n <= 7 else "8-15字" if n <= 15 else "16+字"
        if len(buckets[key]) < args.n:
            buckets[key].append(s)
        if all(len(v) >= args.n for v in buckets.values()):
            break

    def run(items, use_lm):
        """返回 {(长度, 句子): (top1, top9)}；use_lm=False 时临时摘掉模型。"""
        saved = ime.lm
        ime.attach_lm(lm if use_lm else None, args.lm_weight)
        out = {}
        for s in items:
            r = to_pinyin(s, word_map, lexicon=lexicon)
            if not r:
                continue
            key, _words = r
            ime.clear()
            for ch in key:
                ime.insert(ch)
            words = ime.view().words
            out[s] = (words[:1] == [s], s in words[:9])
        ime.attach_lm(saved, args.lm_weight)
        return out

    print(f"{'长度':<8}{'自洽题':>6}{'整句Top1':>10}{'前9':>8}{'平均键数':>9}")
    tot = ok1 = ok9 = 0
    fixed = broken = 0
    for name, items in buckets.items():
        base = run(items, use_lm=False) if lm is not None else {}
        cur = run(items, use_lm=True)
        n1 = n9 = valid = 0
        keys = 0
        for s in items:
            r = to_pinyin(s, word_map, lexicon=lexicon)
            if not r:
                continue
            key, _words = r
            valid += 1
            keys += len(key)
            got1, got9 = cur[s]
            if got1:
                n1 += 1
            if got9:
                n9 += 1
            if lm is not None:
                was1 = base[s][0]
                fixed += (not was1) and got1       # 本来错、重排救回来
                broken += was1 and (not got1)      # 本来对、重排弄坏了
        if not valid:
            print(f"{name:<8}{0:>6}  （没有合法题面）")
            continue
        print(f"{name:<8}{valid:>6}{n1 / valid:>10.1%}{n9 / valid:>8.1%}{keys / valid:>9.1f}")
        tot += valid
        ok1 += n1
        ok9 += n9
    if tot:
        print(f"{'合计':<8}{tot:>6}{ok1 / tot:>10.1%}{ok9 / tot:>8.1%}")
        if lm is not None:
            base_ok = ok1 - fixed + broken
            print(f"\n重排的两种作用必须分开看：救回 {fixed} 条，弄坏 {broken} 条，"
                  f"净 {fixed - broken:+d}（不重排是 {base_ok}/{tot} = {base_ok / tot:.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
