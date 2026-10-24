#!/usr/bin/env python3
"""抓取搜狗细胞词库（已转好的 txt），转成引擎能用的「拼音键 \\t 词 \\t 词频」。

来源：`maqibg/sougouscel` 镜像（把搜狗官方细胞词库的 .scel 预先转成了 txt）。
txt 每行格式是 `'拼音' 词 词频`，例如：
    'a'ba'ka'wei  阿巴卡韦  3

为什么要换：搜狗细胞词库覆盖了大量生僻领域词（医学、工程、城市、人文…），
正是本项目词库缺的那部分；而且它是**按拼音整串给键**，
跟引擎的「拼音键 → 候选词」结构天然对齐。

许可提醒：镜像仓库自己声明「来源于网络，不得用于商业用途」。
本脚本只作**本地评测**用；把这些词打进要发布的词库前，先确认许可。

用法：
  python3 tools/fetch_sogou_dict.py --out vendor/sogou           # 抓取
  python3 tools/fetch_sogou_dict.py --out vendor/sogou --list   # 只看有哪些
  python3 tools/fetch_sogou_dict.py --out vendor/sogou --category 医学医药
"""
from __future__ import annotations

import argparse
import html
import time
import json
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 走镜像：直连 GitHub 在大陆网络不通
API = "https://gh-proxy.com/https://api.github.com/repos/maqibg/sougouscel/contents/"
RAW = "https://gh-proxy.com/https://raw.githubusercontent.com/maqibg/sougouscel/main/"

# 默认抓这些类目：成语/俗语/历史/城市地名/常用语 —— 对日常输入最有价值。
# 刻意不抓物种名录、药品、ICD 编码、游戏 NPC 这类专业词库：
# 它们会把词库撑成几万条生僻词，反而挤掉常用词的候选位置。
DEFAULT_CATEGORIES = ("成语", "俗语", "谚语", "歇后语", "汉语", "词语",
                      "历史词汇", "中国历史", "城市", "地名", "省份", "区县",
                      "景点", "旅游", "民俗", "节日", "生肖", "百家姓",
                      "称谓", "常用语", "口语", "日常用语", "人文")

# 一行格式：`'音节1'音节2'音节3 词 [词频]`，例如
#   'an'ji'ji'suan 氨基己酸        ← 4 个引号，3 个音节
#   'di'fang 敌方                 ← 2 个引号，1 个音节
# 规律：**引号标在每个音节的开头，最后一个音节后面没有闭合引号**
# （`'an'ji'ji'suan` = 'an' + 'ji' + 'ji' + 'suan'）。
# 末段里「拼音」和「词」之间**没有分隔符**（`suan 氨基己酸` 之间是空格，
# 但 `fang 敌方` 里 `fang` 才是拼音、`敌方` 是词）——
# 所以靠「末段里哪一部分是 ASCII」来分：ASCII 部分是拼音，中文部分是词。
def split_line(line: str) -> tuple[str, str, int] | None:
    line = line.strip()
    if not line.startswith("'"):
        return None
    parts = line.split("'")
    if len(parts) < 3:
        return None
    tail = parts[-1]
    # 末段：前缀 ASCII = 最后一个音节，后面是中文词（可能跟词频）
    i = 0
    while i < len(tail) and (tail[i].isascii() and tail[i].isalpha()):
        i += 1
    last_syllable, rest = tail[:i], tail[i:].split()
    key = "".join(parts[1:-1]) + last_syllable
    if not key or not rest:
        return None
    word = rest[0]
    freq = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 0
    return key, word, freq


def fetch(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """带退避重试。gh-proxy 有限流，403 时多试几次通常能过。"""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (ipinyin eval)"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError("fetch failed")


def listing() -> list[dict]:
    """取「搜狗txt」目录下的文件清单。

    注意：GitHub API 会把中文文件名里的 HTML 实体原样返回（如 `&#13774;`），
    必须先 html.unescape，否则拼出来的 URL 取不到文件。
    """
    data = json.loads(fetch(API + urllib.parse.quote("搜狗txt")))
    out = []
    for x in data:
        if x.get("type") != "file":
            continue
        item = dict(x)
        item["name"] = html.unescape(x["name"])
        out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "vendor", "sogou"))
    ap.add_argument("--category", default="", help="只抓名字里含该词的类目")
    ap.add_argument("--max-size", type=int, default=700_000, help="单文件最大字节（超了跳过）")
    ap.add_argument("--list", action="store_true", help="只列类目，不下载")
    ap.add_argument("--limit", type=int, default=60, help="最多抓几个文件")
    args = ap.parse_args()

    try:
        files = listing()
    except Exception as exc:                       # 网络/镜像问题
        print(f"取目录失败：{exc}", file=sys.stderr)
        return 1
    print(f"目录 {len(files)} 个文件")

    if args.category:
        picked = [x for x in files if args.category in x["name"]]
    else:
        picked = [x for x in files
              if x.get("type") == "file"
              and any(c in x["name"] for c in DEFAULT_CATEGORIES)
              and x.get("size", 0) <= args.max_size]
    print(f"目录 {len(files)} 个文件，命中 {len(picked)} 个")
    if picked:
        print("样例:", picked[0]["name"], "→", RAW + urllib.parse.quote("搜狗txt") + "/" + urllib.parse.quote(picked[0]["name"])[:120])
    if args.list:
        for x in picked[:80]:
            print(f"  {x['name']}  ({x.get('size', 0)} B)")
        return 0

    os.makedirs(args.out, exist_ok=True)
    total_words = 0
    for x in picked[: args.limit]:
        name = x["name"]
        try:
            raw = fetch(RAW + urllib.parse.quote("搜狗txt") + "/" + urllib.parse.quote(name))
        except Exception as exc:
            print(f"  跳过 {name}：{exc}")
            continue
        rows = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            parts = split_line(line)
            if not parts:
                continue
            key, word, freq = parts
            if not key.isalpha() or not word:
                continue
            if not all("\u4e00" <= c <= "\u9fff" for c in word):
                continue
            rows.append((key, word, freq))
        if not rows:
            print(f"  {name}: 没解析出词条")
            continue
        path = os.path.join(args.out, f"{name}.tsv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"{k}\t{w}\t{f}" for k, w, f in rows) + "\n")
        total_words += len(rows)
        print(f"  {name}: {len(rows)} 词 → {os.path.basename(path)}")
    print(f"合计 {total_words} 词条 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
