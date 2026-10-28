#!/usr/bin/env python3
"""把开源的拼音数据编译成本输入法使用的词库（data/lexicon.tsv）。

数据来源（tools/fetch_data.sh 下载到 vendor/）：
  pinyin.txt        单字读音（mozillazg/pinyin-data）
  large_pinyin.txt  词组读音（mozillazg/phrase-pinyin-data）
  jieba.txt         词频（fxsjy/jieba，格式「词 频次 词性」）

输出格式：每行「拼音键 \t 词 \t 频次」，拼音键是不带声调、不带分隔符的纯字母串。
"""
from __future__ import annotations

import gzip
import marshal
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor")
OUT = os.path.join(ROOT, "data", "lexicon.tsv")

# 带调字母 → 无调字母；ü 统一写成 v（输入习惯 lv → 绿）
_TONE_TABLE = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüńňǹḿ",
    "aaaaeeeeiiiioooouuuuvvvvvnnnm",
)
_VALID_KEY = re.compile(r"^[a-z]+$")

MAX_WORD_LEN = 8
MAX_KEY_LEN = 24
# 唔/嗯/呣 这类叹词在词库里会跟前面任意音节拼成"怎唔/我唔"这种垃圾候选
JUNK_KEYS = set(__import__('string').ascii_lowercase) | {"m", "n", "ng", "hm", "hng"}


# 每天都会用、但词频库里排名不高的词：给它们加权，保证第一候选
DAILY_BOOST = {
    "晚安": 40, "早上好": 30, "中午好": 25, "下午好": 25, "晚上好": 25,
    "你好": 12, "谢谢": 12, "再见": 12, "对不起": 12, "没关系": 12, "不客气": 12,
    "请问": 8, "麻烦": 6, "一会儿": 6, "怎么样": 6,
}


def strip_tones(text: str) -> str:
    """去掉声调与分隔符，只保留 a-z（ü 记作 v）。"""
    text = text.translate(_TONE_TABLE).lower()
    return re.sub(r"[^a-z]", "", text)


def load_freq(path: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            freq[parts[0]] = int(parts[1])
    return freq


def load_chars(path: str) -> list[tuple[str, str]]:
    """返回 [(汉字, 无调读音), ...]；多音字每个读音一条。"""
    out: list[tuple[str, str]] = []
    pattern = re.compile(r"^U\+([0-9A-Fa-f]+):\s*([^#]+?)\s*(?:#\s*(.*))?$")
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                continue
            try:
                code = int(m.group(1), 16)
                char = chr(code)
            except (ValueError, OverflowError):
                continue
            # 只收基本汉字区（含〇），扩展 A/B 区生僻字对日常输入是噪音
            if not (0x4E00 <= code <= 0x9FFF or code == 0x3007):
                continue
            readings = [r.strip() for r in m.group(2).split(",") if r.strip()]
            for index, reading in enumerate(readings):
                key = strip_tones(reading)
                if key and _VALID_KEY.match(key) and len(key) <= MAX_KEY_LEN:
                    # 第一个读音是该字的常用读音，权重略高于其他读音
                    out.append((char, key if index == 0 else key + "\x00" + str(index)))
    return out


def load_cedict(path: str) -> list[tuple[str, str]]:
    """CC-CEDICT（12.5 万条，中英词典）：补词库覆盖不到的词条。

    行格式：傳統 简体 [pin1 yin1] /释义/
    """
    out: list[tuple[str, str]] = []
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue
            _trad, simp, numbered = match.groups()
            if len(simp) > MAX_WORD_LEN:
                continue
            plain = numbered.replace("u:", "v").replace("U:", "v")
            plain = re.sub(r"[0-9]", "", plain)
            key = strip_tones(plain)
            if key and _VALID_KEY.match(key) and len(key) <= MAX_KEY_LEN:
                out.append((simp, key))
    return out


def load_phrases(path: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or ": " not in line:
                continue
            word, pinyin = line.split(": ", 1)
            word = word.strip()
            if not word or len(word) > MAX_WORD_LEN:
                continue
            key = strip_tones(pinyin)
            if key and _VALID_KEY.match(key) and len(key) <= MAX_KEY_LEN:
                out.append((word, key))
    return out


def main() -> int:
    for name in ("pinyin.txt", "large_pinyin.txt", "jieba.txt"):
        path = os.path.join(VENDOR, name)
        if not os.path.exists(path):
            print(f"缺少数据文件 {path}，请先运行 tools/fetch_data.sh", file=sys.stderr)
            return 2

    freq = load_freq(os.path.join(VENDOR, "jieba.txt"))
    print(f"词频条目：{len(freq)}")

    # key -> {word: weight}
    table: dict[str, dict[str, float]] = defaultdict(dict)

    def put(key: str, word: str, weight: float) -> None:
        if weight <= 0:
            return
        if key.split("\x00")[0] in JUNK_KEYS:
            return
        old = table[key].get(word, 0.0)
        if weight > old:
            table[key][word] = weight

    chars = load_chars(os.path.join(VENDOR, "pinyin.txt"))
    char_count = 0
    for char, key in chars:
        if key in JUNK_KEYS:
            continue
        primary = "\x00" not in key
        key = key.split("\x00")[0]
        base = freq.get(char, 0)
        if base <= 0:
            base = 5 if primary else 3  # 词频表没收录的生僻字，排到后面去
        put(key, char, base if primary else max(base * 0.02, 1))
        char_count += 1
    print(f"单字读音：{char_count}")

    phrases = load_phrases(os.path.join(VENDOR, "large_pinyin.txt"))
    kept = kept_unk = 0
    for word, key in phrases:
        f = freq.get(word, 0)
        if f <= 0:
            # 词频表没收录的词基本都是生僻词，收进来只会污染候选，直接丢弃
            continue
        # 长词天然更具体，给一点长度奖励，避免被高频短词淹没
        put(key, word, f * (1.0 + 0.15 * (len(word) - 1)))
        kept += 1
    print(f"词组：{kept}（其中无词频 {kept_unk}）")

    # CC-CEDICT：中英词典，补专有名词/技术词/书面语
    cedict = os.path.join(VENDOR, "cedict.gz")
    if os.path.exists(cedict):
        added = 0
        for word, key in load_cedict(cedict):
            f = freq.get(word, 0)
            # 词典里的词没有语料词频，权重压低，只保证"有词可选"
            put(key, word, f * 1.15 if f > 0 else 25)
            added += 1
        print(f"CC-CEDICT 补充词条：{added}")

    # 日常高频词加权：词频库里「万安」这类地名/书面语会压过「晚安」，这里矫正
    for bucket in table.values():
        for word in list(bucket):
            factor = DAILY_BOOST.get(word)
            if factor:
                bucket[word] *= factor

    # 现代常用词（品牌/职场），基础词典常缺；给高权重保证优先
    extra_path = os.path.join(ROOT, "data", "extra_words.txt") if "ROOT" in globals() else os.path.join("data", "extra_words.txt")
    if os.path.exists(extra_path):
        added = 0
        for line in open(extra_path, encoding="utf-8"):
            if line.startswith("#") or "\t" not in line:
                continue
            word, _, key = line.rstrip("\n").partition("\t")
            word, key = word.strip(), key.strip()
            if word and key:
                put(key, word, 1e9)      # 人工收录的现代词：明确标记，运行时能识别
                added += 1
        print(f"现代常用词补充：{added}")

    # 统计与裁剪
    pairs = sum(len(v) for v in table.values())
    print(f"拼音键 {len(table)} 个，候选词条 {pairs} 条")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for key in sorted(table):
            items = sorted(table[key].items(), key=lambda kv: -kv[1])
            for word, weight in items:
                fh.write(f"{key}\t{word}\t{int(round(weight))}\n")
    size = os.path.getsize(OUT)
    print(f"写出 {OUT}（{size/1024:.0f} KiB）")

    # 再压一份 marshal 快照：引擎启动直接读它，省掉解析文本的时间
    compact = {
        key: "|".join(f"{word}:{int(round(weight))}" for word, weight in sorted(items.items(), key=lambda kv: -kv[1]))
        for key, items in table.items()
    }
    total = sum(weight for items in table.values() for weight in items.values())
    snapshot = os.path.join(os.path.dirname(OUT), "lexicon.marshal")
    with open(snapshot, "wb") as fh:
        marshal.dump((compact, total), fh)
    print(f"写出 {snapshot}（{os.path.getsize(snapshot)/1024:.0f} KiB）")

    # 前缀候选表：1~3 个字母的前缀预先算好「最高频的 24 个词」。
    # 不预计算的话，输入 "s" 只能按字母序扫前 150 个键，候选会变成一堆 sa*。
    prefix_table: dict[str, list[tuple[int, str]]] = {}
    for length in (1, 2, 3):
        buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for key, payload in compact.items():
            head = payload.split("|", 1)[0]
            word, _, freq = head.rpartition(":")
            buckets[key[:length]].append((int(freq), word))
        for prefix, items in buckets.items():
            items.sort(reverse=True)
            prefix_table[prefix] = items[:24]
    prefixes = {
        prefix: "|".join(f"{word}:{freq}" for freq, word in items)
        for prefix, items in prefix_table.items()
    }
    prefix_file = os.path.join(os.path.dirname(OUT), "prefix.marshal")
    with open(prefix_file, "wb") as fh:
        marshal.dump(prefixes, fh)
    print(f"写出 {prefix_file}（{os.path.getsize(prefix_file)/1024:.0f} KiB，{len(prefixes)} 个前缀）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
