#!/usr/bin/env python3
"""逐词覆盖率：参考句里的词有多少能被引擎产出来 —— 评「补词」专用的指标。

为什么整句基准量不到补词：自洽题面要求切出来的每个词都是它拼音键的**首选词**，
词库缺的词没有拼音键，含缺词的句子从一开始就进不了测试集。
实测把挖出的词加进词库，整句 Top1 纹丝不动（9.9% → 9.9%）；
只有逐词覆盖率看得见补词有没有用。

做法：
  1. 补词表（默认 tools/user_words.txt，缺词带拼音）；已在词库里的跳过；
  2. 题库：个人文档 4~8 字句子，分成「含补词」和「对照（不含）」两组；
  3. 基线：现词库；对照：词库 + 补词（临时 TSV，不动原文件）；
  4. 每句用各自词库最长匹配切词、拼整句拼音（切不出的句子单独计数 ——
     那本身就是「打不出来」），逐键敲进 ImeState，看：
       整句 Top1 / 逐词覆盖（参考词出现在候选里：词本身或整句候选的一段）
两组句子两遍跑，加词应该只在「含补词」组涨分。

用法：
  python3 tools/eval_word_coverage.py
  python3 tools/eval_word_coverage.py --words tools/user_words.txt --limit 3000
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402
from eval_sentence import build_word_map  # noqa: E402

DATA = os.environ.get("IPINYIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".local/share/ipinyin/data")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")


def load_extra_words(path: str, lexicon_words: set[str]) -> list[tuple[str, str]]:
    """补词表：词 拼音。已在词库里的跳过。"""
    out: list[tuple[str, str]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in lexicon_words:
            out.append((parts[0], "".join(parts[1:]).lower()))
    return out


def lexicon_with_extra(base_path: str, extra: list[tuple[str, str]]) -> "Lexicon":
    """把补词并进词库副本（freq 给个不高不低的 500，不挤常用词）。"""
    fd, tmp = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(open(base_path, encoding="utf-8").read())
        for word, key in extra:
            fh.write(f"{key}\t{word}\t500\n")
    try:
        return Lexicon.load(tmp)
    finally:
        os.unlink(tmp)


def docs_sentences(src: str, conv: Converter, max_len: int = 8) -> list[str]:
    out: list[str] = []
    for path in sorted(glob.glob(os.path.join(src, "*.txt"))):
        text = conv.to_simplified(open(path, encoding="utf-8", errors="ignore").read())
        for piece in re.findall(r"[\u4e00-\u9fff]+", text):
            for sent in re.split(r"[，。；：！？、\s]", piece):
                sent = sent.strip()
                if 4 <= len(sent) <= max_len:
                    out.append(sent)
    return out


def segment(sentence: str, word_map: dict[str, str], max_word: int = 6):
    """最长匹配切词；切不出返回 None。返回 (拼音串, 词列表)。"""
    keys, words = [], []
    i, size = 0, len(sentence)
    while i < size:
        for length in range(min(max_word, size - i), 0, -1):
            piece = sentence[i:i + length]
            key = word_map.get(piece)
            if key:
                keys.append(key)
                words.append(piece)
                i += length
                break
        else:
            return None
    return "".join(keys), words


def run_group(lexicon: Lexicon, sents: list[str]) -> dict:
    word_map = build_word_map(lexicon)
    ime = ImeState(lexicon)
    n = top1 = 0
    cov_sum = 0.0
    cov_n = 0
    uncovered: list[str] = []
    for sent in sents:
        seg = segment(sent, word_map)
        if seg is None:
            continue
        pinyin, ref_words = seg
        ime.clear()
        for ch in pinyin:
            ime.insert(ch)
        cands = ime._candidates            # 全量候选（不只第一页）
        n += 1
        if cands[:1] == [sent]:
            top1 += 1
        hit = 0
        for w in ref_words:
            if any(c == w or w in c for c in cands):
                hit += 1
            elif len(uncovered) < 12 and w not in uncovered:
                uncovered.append(w)
        cov_sum += hit / len(ref_words)
        cov_n += 1
    return {
        "n": n,
        "top1": top1 / n if n else 0.0,
        "coverage": cov_sum / cov_n if cov_n else 0.0,
        "uncovered": uncovered,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "vendor", "personal", "_clean"))
    ap.add_argument("--words", default=os.path.join(ROOT, "tools", "user_words.txt"))
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="限制题数（0=不限）")
    args = ap.parse_args()

    conv = Converter.load(os.path.join(DATA, "s2t.marshal"))
    base_lex = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    lex_words = set()
    for line in open(os.path.join(DATA, "lexicon.tsv"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            lex_words.add(parts[1])

    extra = load_extra_words(args.words, lex_words)
    if not extra:
        print("补词表里的词都已在词库，没有可评的缺词", file=sys.stderr)
        return 1
    print(f"补词 {len(extra)} 个：{' '.join(w for w, _ in extra[:12])}"
          f"{' …' if len(extra) > 12 else ''}\n")

    sents = docs_sentences(args.src, conv, args.max_len)
    extra_set = {w for w, _ in extra}
    hit = [s for s in sents if any(w in s for w in extra_set)]
    ctrl = [s for s in sents if not any(w in s for w in extra_set)]
    if args.limit:
        hit, ctrl = hit[:args.limit], ctrl[:args.limit]
    if not hit:
        print("文档里没有含补词的句子", file=sys.stderr)
        return 1

    lex_new = lexicon_with_extra(os.path.join(DATA, "lexicon.tsv"), extra)

    print(f"{'组':<10}{'词库':<8}{'可切分':>8}{'整句Top1':>10}{'逐词覆盖':>10}")
    for name, group in (("含补词", hit), ("对照", ctrl)):
        for tag, lex in (("基线", base_lex), ("加词", lex_new)):
            r = run_group(lex, group)
            print(f"{name:<10}{tag:<8}{r['n']:>8}{r['top1']:>10.1%}{r['coverage']:>10.1%}")
            if name == "含补词" and tag == "加词" and r["uncovered"]:
                print(f"           仍未覆盖的词：{' '.join(r['uncovered'][:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
