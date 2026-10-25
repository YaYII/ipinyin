#!/usr/bin/env python3
"""合成中文错别字数据 + 构建本基准的评测资产。

数据来源决策（2026-10）：
  - 曾考虑联网拉 SIGHAN CSC 数据集（github 镜像），按任务约束最多试 2 次；
    实际未试：仓库内 vendor/yue_pairs.json（16.9 万句简体中文）+ data/lexicon.tsv
    （19.3 万条「拼音键→词」）已经足够合成同音/形近两类错别字，且完全离线、
    可复现、不引入许可证疑虑。README 里有更完整的记录。

合成策略（与 docs/nn-upgrade.md P4 的「编辑式模型」同构——只替换、不增删）：
  1. 同音混淆（字级）：lexicon.tsv 同一拼音键下的单字互为同音候选
     （多音字天然出现在多个键下，候选取并集）。
  2. 同音混淆（词级）：同一拼音键下的多字词互相整词替换，
     替换后逐字记录差异（两词共有的字不算错）。
  3. 形近混淆：手写 ~60 对常见形近字（己已巳/末未/拆折…），双向可用。

输出（全部落在本目录 data/ 下，不动仓库 data/）：
  confusion.json          同音/形近混淆表（baseline.py 也用它做候选源）
  train_typos.jsonl       训练用错句（给未来的编辑式模型，本轮不训）
  dev.jsonl / test.jsonl  评测集（含 ~15% 干净句，用于统计误报）
  lm_eval.marshal         只用 train 划分训的 3-gram CharLM（避免评测污染）

JSONL 行格式：{"sentence": <错句>, "errors": [{"char": 错字, "correct": 正字, "position": int}]}
position 是错句中的下标；sentence 始终是「待纠正的输入」，干净句 errors 为空。

用法：
  python3 data_gen.py                 # 全量：混淆表 + LM + 三个 JSONL
  python3 data_gen.py --only-tables   # 只生成混淆表
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.lm import CharLM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LEXICON_TSV = os.path.join(ROOT, "data", "lexicon.tsv")
YUE_PAIRS = os.path.join(ROOT, "vendor", "yue_pairs.json")
PRODUCT_LM = os.path.join(ROOT, "data", "lm_char.marshal")

SEED = 42
MIN_CHAR_FREQ = 50        # 同音候选字双方都要达到的词频门槛（滤掉生僻字）
MIN_WORD_FREQ = 200       # 词级替换双方都要达到的词频门槛
MAX_HOMOPHONE_CANDS = 10  # 混淆表里每个字最多保留几个同音候选（按词频降序）
CN = "\u4e00-\u9fff"

# ~60 对常见形近字（手写整理，双向）。故意不收互为同音的对（如 振/震、帐/账），
# 让「同音/形近」两个评测桶互斥。
SHAPE_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("末", "未"), ("日", "曰"), ("人", "入"), ("八", "入"),
    ("儿", "几"), ("刀", "力"), ("千", "干"), ("于", "干"), ("土", "士"),
    ("大", "太"), ("天", "夫"), ("王", "玉"), ("午", "牛"), ("牛", "年"),
    ("手", "毛"), ("乌", "鸟"), ("问", "间"), ("休", "体"), ("拆", "折"),
    ("抵", "低"), ("侯", "候"), ("睛", "晴"), ("请", "清"), ("清", "情"),
    ("辨", "辩"), ("辩", "辫"), ("喝", "渴"), ("拨", "泼"), ("治", "冶"),
    ("官", "宫"), ("栽", "载"), ("裁", "载"), ("换", "唤"), ("竞", "竟"),
    ("境", "镜"), ("崇", "祟"), ("贪", "贫"), ("味", "昧"), ("盲", "肓"),
    ("段", "断"), ("假", "暇"), ("模", "摸"), ("概", "慨"), ("即", "既"),
    ("象", "像"), ("材", "才"), ("闲", "闹"), ("扰", "拢"), ("忆", "亿"),
    ("纪", "记"), ("令", "今"), ("代", "伐"), ("蓝", "篮"), ("豪", "毫"),
    ("徒", "陡"), ("徒", "徙"), ("抄", "炒"), ("戊", "戍"), ("戍", "戌"),
]


# ---------------------------------------------------------------- 混淆表

def build_confusion(lexicon: Lexicon) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """返回 (homophone, shape)。

    homophone: 字 → 同音候选（词频降序）。字频取该字在所有键下 entries 的最大值。
    """
    freq: dict[str, int] = {}
    groups: dict[str, list[tuple[str, int]]] = {}
    for key in lexicon._keys:  # 只读用途；本脚本是离线数据工具，可接受
        entries = lexicon.entries(key)
        for word, f in entries:
            if len(word) == 1:
                if f > freq.get(word, 0):
                    freq[word] = f
                groups.setdefault(key, []).append((word, f))

    homophone: dict[str, set[str]] = {}
    for key, entries in groups.items():
        singles = [(w, f) for w, f in entries if f >= MIN_CHAR_FREQ]
        if len(singles) < 2:
            continue
        for word, _ in singles:
            if freq.get(word, 0) < MIN_CHAR_FREQ:
                continue
            cands = homophone.setdefault(word, set())
            for other, f in singles:
                if other != word and f >= MIN_CHAR_FREQ:
                    cands.add(other)

    # 常用字过滤：候选必须出现在评测语料里，滤掉繁体/生僻字（鍔、妸…）
    common = corpus_chars()
    homophone_out = {
        c: sorted(cands, key=lambda x: -freq.get(x, 0))[:MAX_HOMOPHONE_CANDS]
        for c, cands in homophone.items()
        if c in common and freq.get(c, 0) >= MIN_CHAR_FREQ
    }
    homophone_out = {c: [x for x in cands if x in common] for c, cands in homophone_out.items()}
    homophone_out = {c: cands for c, cands in homophone_out.items() if cands}

    shape: dict[str, set[str]] = {}
    for a, b in SHAPE_PAIRS:
        if a in homophone_out and b in homophone_out.get(a, []):
            continue  # 互为同音的对不进形近桶，保持桶互斥
        shape.setdefault(a, set()).add(b)
        shape.setdefault(b, set()).add(a)
    shape_out = {c: sorted(s) for c, s in shape.items() if c in common}
    return homophone_out, shape_out


def corpus_chars() -> set[str]:
    """评测语料（yue_pairs zh 侧）出现过的汉字集合，用来滤生僻候选。"""
    cache = os.path.join(DATA_DIR, ".corpus_chars.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return set(fh.read())
    chars: set[str] = set()
    with open(YUE_PAIRS, encoding="utf-8") as fh:
        for p in json.load(fh):
            for ch in p.get("zh", ""):
                if "\u4e00" <= ch <= "\u9fff":
                    chars.add(ch)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write("".join(sorted(chars)))
    return chars


def build_word_table(lexicon: Lexicon) -> dict[str, tuple[str, int]]:
    """词 → (拼音键, 词频)，只留 2~4 字、词频达标的词（给词级替换用）。"""
    table: dict[str, tuple[str, int]] = {}
    for key in lexicon._keys:
        for word, f in lexicon.entries(key):
            if 2 <= len(word) <= 4 and f >= MIN_WORD_FREQ and word not in table:
                table[word] = (key, f)
    return table


# ---------------------------------------------------------------- 造错

def _pick(rng: random.Random, seq: list[str]) -> str | None:
    return rng.choice(seq) if seq else None


def inject_errors(
    sentence: str,
    homophone: dict[str, list[str]],
    shape: dict[str, list[str]],
    word_table: dict[str, tuple[str, int]],
    lexicon: Lexicon,
    rng: random.Random,
    n_errors: int,
) -> tuple[str, list[dict]]:
    """向句子注入 n_errors 个错，返回 (错句, errors)。

    每个错的类型按 同音字 0.5 / 同音词 0.3 / 形近 0.2 抽签，不可行时回退。
    返回的 position 均为错句中的下标（替换不改长度）。
    """
    chars = list(sentence)
    n = len(chars)
    occupied: set[int] = set()          # 已被词级替换占用的位置
    errors: list[dict] = []
    cn_positions = [i for i, ch in enumerate(chars) if "\u4e00" <= ch <= "\u9fff"]

    for _ in range(n_errors):
        roll = rng.random()
        order = ["homophone_char", "homophone_word", "shape"] if roll < 0.5 else (
            ["homophone_word", "shape", "homophone_char"] if roll < 0.8 else
            ["shape", "homophone_char", "homophone_word"])

        done = False
        for kind in order:
            if kind == "homophone_word" and word_table:
                done = _inject_word(chars, word_table, lexicon, rng, occupied, errors)
            elif kind == "shape":
                done = _inject_char(chars, shape, rng, occupied, errors, cn_positions, "shape")
            else:
                done = _inject_char(chars, homophone, rng, occupied, errors, cn_positions, "homophone")
            if done:
                break
    return "".join(chars), errors


def _inject_word(chars, word_table, lexicon, rng, occupied, errors) -> bool:
    n = len(chars)
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts:
        for length in (2, 3, 4):
            end = start + length
            if end > n or any(i in occupied for i in range(start, end)):
                continue
            word = "".join(chars[start:end])
            hit = word_table.get(word)
            if not hit:
                continue
            key, _ = hit
            cands = [w for w, f in lexicon.entries(key)
                     if w != word and f >= MIN_WORD_FREQ and len(w) == length
                     and sum(1 for a, b in zip(w, word) if a != b) >= 1
                     and all("\u4e00" <= ch <= "\u9fff" for ch in w)]
            if not cands:
                continue
            repl = rng.choice(cands)
            for j, (orig, new) in enumerate(zip(word, repl)):
                if orig != new:
                    chars[start + j] = new
                    errors.append({"char": new, "correct": orig, "position": start + j,
                                   "type": "homophone"})
                    occupied.add(start + j)
            errors.sort(key=lambda e: e["position"])
            return True
    return False


def _inject_char(chars, table, rng, occupied, errors, cn_positions, etype) -> bool:
    positions = [i for i in cn_positions if i not in occupied and chars[i] in table]
    rng.shuffle(positions)
    for i in positions:
        cand = _pick(rng, table[chars[i]])
        if cand and cand != chars[i]:
            errors.append({"char": cand, "correct": chars[i], "position": i, "type": etype})
            chars[i] = cand
            occupied.add(i)
            return True
    return False


# ---------------------------------------------------------------- 划分与主流程

def _split_of(sentence: str) -> str:
    """按句子内容哈希稳定划分：train / dev / test，比例 88/4/8。"""
    h = int(hashlib.md5(sentence.encode("utf-8")).hexdigest(), 16) % 100
    if h < 88:
        return "train"
    if h < 92:
        return "dev"
    return "test"


def clean_sentences() -> list[str]:
    import re
    ok = re.compile(r"^[\u4e00-\u9fff\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\u2026\u2014,.!?;: 0-9a-zA-Z]+$")
    out = []
    with open(YUE_PAIRS, encoding="utf-8") as fh:
        for p in json.load(fh):
            s = p.get("zh", "").strip()
            if 8 <= len(s) <= 40 and ok.match(s) and sum("\u4e00" <= c <= "\u9fff" for c in s) >= 6:
                out.append(s)
    return out


def train_lm(sentences: list[str], out_path: str) -> str:
    lm = CharLM(3).fit(sentences, min_count=2)
    lm.save(out_path)
    return f"{os.path.getsize(out_path) / 1e6:.1f}MB, vocab {len(lm)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-tables", action="store_true", help="只生成混淆表")
    ap.add_argument("--dev", type=int, default=600)
    ap.add_argument("--test", type=int, default=1200)
    ap.add_argument("--train-typos", type=int, default=20000)
    ap.add_argument("--clean-ratio", type=float, default=0.15, help="评测集中无错句占比")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    lexicon = Lexicon.load(LEXICON_TSV)
    print(f"词库加载：{len(lexicon._keys)} 键")

    homophone, shape = build_confusion(lexicon)
    word_table = build_word_table(lexicon)
    with open(os.path.join(DATA_DIR, "confusion.json"), "w", encoding="utf-8") as fh:
        json.dump({"homophone": homophone, "shape": shape,
                   "shape_pairs": SHAPE_PAIRS}, fh, ensure_ascii=False, indent=1)
    print(f"混淆表：同音字 {len(homophone)} / 形近字 {len(shape)} / 可替换词 {len(word_table)}")
    if args.only_tables:
        return 0

    sentences = clean_sentences()
    print(f"评测候选句（8~40 字）：{len(sentences)}")
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for s in sentences:
        by_split[_split_of(s)].append(s)
    print(f"划分：train {len(by_split['train'])} / dev {len(by_split['dev'])} / test {len(by_split['test'])}")

    lm_path = os.path.join(DATA_DIR, "lm_eval.marshal")
    if os.path.exists(lm_path):
        print(f"评测 LM 已存在，跳过训练：{lm_path}")
    else:
        stats = train_lm(by_split["train"], lm_path)
        print(f"评测 LM（只用 train 划分，避免污染）：{stats}")

    rng = random.Random(SEED)
    pools = {"dev": by_split["dev"], "test": by_split["test"], "train": by_split["train"]}
    quotas = {"dev": args.dev, "test": args.test, "train": args.train_typos}

    for name, pool in pools.items():
        rng.shuffle(pool)
        made, idx = 0, 0
        out_path = os.path.join(DATA_DIR, f"{name}.jsonl" if name != "train" else "train_typos.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            while made < quotas[name] and idx < len(pool):
                s = pool[idx]
                idx += 1
                if rng.random() < args.clean_ratio:
                    record = {"sentence": s, "errors": []}
                else:
                    n_err = 1 if rng.random() < 0.7 else 2
                    bad, errors = inject_errors(s, homophone, shape, word_table, lexicon, rng, n_err)
                    if not errors:
                        continue
                    assert all(bad[e["position"]] == e["char"] for e in errors)
                    record = {"sentence": bad, "errors": errors}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                made += 1
        print(f"{out_path}: {made} 句")

    print(f"\n注意：本机产品 LM {PRODUCT_LM} "
          f"{'存在' if os.path.exists(PRODUCT_LM) else '不存在'}；"
          f"它由 yue_pairs 全量训练，本基准默认用 train 划分现训的 lm_eval.marshal，"
          f"避免评测句泄漏进 LM。对比实验可 --lm 指向产品 LM。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
