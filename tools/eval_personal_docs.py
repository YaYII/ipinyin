#!/usr/bin/env python3
"""用你自己的简体文档做端到端评测：不重排 vs 重排，救回/弄坏分开报。

为什么单独一个脚本：`eval_sentence.py --text` 需要事先备好题库和一份**没见过的** LM。
这里把三件事串起来，一条命令出结果：
  1. 把 vendor/personal/_clean 按句切分，前 70% 训练、后 30% 测试（无泄漏）；
  2. 用「通用语料 + 前 70%」训一份临时 LM；
  3. 在同一批测试句上跑两遍（挂 LM / 不挂），报 Top1、前 9、救回、弄坏。

用法：
  python3 tools/eval_personal_docs.py                  # 默认 λ=0.4
  python3 tools/eval_personal_docs.py --weight 0.3     # 换权重
  python3 tools/eval_personal_docs.py --all            # 用全部文档当测试集（有泄漏，只看上限）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.lm import CharLM  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402

DATA = os.environ.get("IPINYIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".local/share/ipinyin/data")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")


def sentences(paths: list[str], conv: Converter, max_len: int = 8) -> list[str]:
    out: list[str] = []
    for path in paths:
        text = conv.to_simplified(open(path, encoding="utf-8", errors="ignore").read())
        for piece in re.findall(r"[\u4e00-\u9fff]+", text):
            for sent in piece.split():
                if 4 <= len(sent) <= max_len:
                    out.append(sent)
    return out


def bucket(sent: str) -> str:
    return "2-4字" if len(sent) <= 4 else "5-8字"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "vendor", "personal", "_clean"))
    ap.add_argument("--pairs", default=os.path.join(ROOT, "vendor", "yue_pairs.json"))
    ap.add_argument("--weight", type=float, default=0.4)
    ap.add_argument("--all", action="store_true", help="跳过 70/30 切分（会把测试句也训进去）")
    ap.add_argument("--user-seed", action="store_true",
                    help="挂上 data/user_seed.tsv（个人词库，含 user_words.txt 的补充词）")
    ap.add_argument("--max-len", type=int, default=8,
                    help="只测多长的句子（默认 8：再长就不是常见打字长度了）")
    args = ap.parse_args()

    conv = Converter.load(os.path.join(DATA, "s2t.marshal"))
    docs = sorted(glob.glob(os.path.join(args.src, "*.txt")))
    if not docs:
        print(f"{args.src} 没有 .txt，先导出文档", file=sys.stderr)
        return 1
    all_sents = sentences(docs, conv, args.max_len)
    cut = int(len(all_sents) * 0.7)
    train_doc, test_doc = all_sents[:cut], all_sents[cut:]
    if args.all:
        train_doc, test_doc = all_sents, all_sents

    base: list[str] = []
    if os.path.exists(args.pairs):
        base = [p["zh"] for p in json.load(open(args.pairs, encoding="utf-8")) if p.get("zh")]
    lm = CharLM(3).fit(base + train_doc, min_count=2)

    lexicon = Lexicon.load(os.path.join(DATA, "lexicon.tsv"))
    from eval_sentence import build_word_map, to_pinyin   # 复用同一套题面规则

    user_model = None
    if args.user_seed:
        from ipinyin.user_model import UserModel

        def valid(key: str, word: str) -> bool:
            if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
                return True
            return len(word) in lexicon.syllable_counts(key)

        user_model = UserModel(os.path.join(DATA, "user_seed.tsv"), validator=valid)

    word_map = build_word_map(lexicon)
    ime = ImeState(lexicon, user_model=user_model)

    def run(use_lm: bool) -> dict[str, tuple[bool, bool]]:
        ime.attach_lm(lm if use_lm else None, args.weight)
        out: dict[str, tuple[bool, bool]] = {}
        for sent in test_doc:
            r = to_pinyin(sent, word_map, lexicon=lexicon)
            if not r:
                continue
            ime.clear()
            for ch in r[0]:
                ime.insert(ch)
            words = ime.view().words
            out[sent] = (words[:1] == [sent], sent in words[:9])
        return out

    base_res, cur_res = run(False), run(True)
    if not base_res:
        print("没有可用的自洽题面（每个词都必须是它拼音键的首选词）", file=sys.stderr)
        return 1

    print(f"文档 {len(all_sents)} 句 → 训练 {len(train_doc)} / 测试 {len(test_doc)}；"
          f"自洽题 {len(base_res)} 条；λ={args.weight}\n")
    print(f"{'长度':<8}{'题数':>6}{'不重排':>9}{'重排':>9}{'变化':>8}{'救回':>6}{'弄坏':>6}")
    total = [0, 0, 0, 0, 0]
    for name in ("2-4字", "5-8字"):
        keys = [s for s in base_res if bucket(s) == name]
        if not keys:
            continue
        b = sum(base_res[s][0] for s in keys)
        c = sum(cur_res[s][0] for s in keys)
        fixed = sum(1 for s in keys if not base_res[s][0] and cur_res[s][0])
        broken = sum(1 for s in keys if base_res[s][0] and not cur_res[s][0])
        n = len(keys)
        print(f"{name:<8}{n:>6}{b / n:>9.1%}{c / n:>9.1%}{(c - b) / n * 100:>+7.1f}pt"
              f"{fixed:>6}{broken:>6}")
        total[0] += n
        total[1] += b
        total[2] += c
        total[3] += fixed
        total[4] += broken
    n, b, c, fixed, broken = total
    print(f"{'合计':<8}{n:>6}{b / n:>9.1%}{c / n:>9.1%}{(c - b) / n * 100:>+7.1f}pt"
          f"{fixed:>6}{broken:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
