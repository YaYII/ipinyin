"""把旧的单一 user_model.tsv 拆成 简 / 繁 / 粤 三份（一次性迁移）。

旧版把三种输出模式的命中混在一张表里：粤语选过的「嘅」会顶掉简体候选，
繁体选过的「聲音」会原样出现在简体模式（显示成「聲音」而不是「声音」）。
新版按模式各存一份文件，这个脚本把历史数据按字形分派过去。

用法：
  python3 tools/split_user_model.py            # 预演，只打印分派结果
  python3 tools/split_user_model.py --apply    # 真正写盘（原文件备份 .bak）

跑完记得重启输入法：ibus restart
"""
from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin import config  # noqa: E402
from ipinyin.t2s import Converter  # noqa: E402

# 只在粤语书面里出现的字：带这些字的词归粤语那份
CANTONESE = set("嘅哋唔咗佢咩睇瞓嗰啲喺攞係冇咁嚟啱嘢乜諗揀嗮")


def target_for(base: str, mode: str) -> str:
    """user_model.tsv / user_model.trad.tsv / user_bigram.tsv / user_bigram.yue.tsv。

    不能拿 config.user_model_path(mode) 去 replace：`user_model.trad.tsv` 里
    没有 `user_model.tsv` 这个子串，替换会静默失效，把词表写成词对表（踩过）。
    """
    stem = "user_bigram" if base.startswith("user_bigram") else "user_model"
    name = f"{stem}.tsv" if mode == "simp" else f"{stem}.{mode}.tsv"
    return os.path.join(os.path.dirname(config.user_model_path("simp")), name)


def classify(word: str, converter: Converter) -> str:
    """按字形判断这条命中属于哪个模式。

    先看繁简：`關係` 里虽然有「係」，但它是繁体（简化为「关系」），归繁。
    再看粤语字：`嘅`、`唔啱` 没有简体形，归粤语。
    其余归简体。
    """
    simplified = converter.to_simplified(word)
    if simplified != word:
        return "trad"
    if set(word) & CANTONESE:
        return "yue"
    return "simp"


def split_words(path: str, converter: Converter) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"simp": [], "trad": [], "yue": []}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # 归类看原字形（「聲音」是繁体、「關係」是繁体），落盘写简体形：
            # 繁/粤只是显示层转换，词表里存「聲音」会和候选基础形「声音」对不上，
            # 同一个词会在候选里出现两次
            mode = classify(parts[1], converter)
            parts[1] = converter.to_simplified(parts[1])
            buckets[mode].append("\t".join(parts) + "\n")
    return buckets


def split_bigrams(path: str, converter: Converter) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"simp": [], "trad": [], "yue": []}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # 一对里只要有一半是粤语/繁体，整对就跟着走
            kinds = {classify(parts[0], converter), classify(parts[1], converter)}  # 按原字形归类
            mode = "yue" if "yue" in kinds else ("trad" if "trad" in kinds else "simp")
            parts[0] = converter.to_simplified(parts[0])
            parts[1] = converter.to_simplified(parts[1])
            buckets[mode].append("\t".join(parts) + "\n")
    return buckets


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    converter = Converter.load(config.s2t_path())
    if not len(converter):
        print("缺 data/s2t.marshal，先跑 python3 tools/build_s2t.py", file=sys.stderr)
        return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for base, splitter in (("user_model.tsv", split_words), ("user_bigram.tsv", split_bigrams)):
        source = target_for(base, "simp")
        if not os.path.exists(source):
            print(f"跳过 {base}（不存在）")
            continue
        buckets = splitter(source, converter)
        counts = {mode: len(rows) for mode, rows in buckets.items()}
        print(f"{base}: 简 {counts['simp']} / 繁 {counts['trad']} / 粤 {counts['yue']}")
        if not apply:
            continue
        shutil.copy2(source, f"{source}.bak-{stamp}")
        for mode in ("simp", "trad", "yue"):
            target = target_for(base, mode)
            header = "# 拼音键\t词\t选中次数\t曝光次数\n" if base == "user_model.tsv" else ""
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(header)
                handle.writelines(buckets[mode])
            print(f"  写入 {os.path.basename(target)}（{counts[mode]} 条）")
    if not apply:
        print("\n预演完成。确认无误后加 --apply 真正写盘（原文件会备份成 .bak-<时间戳>）")
    else:
        print("\n迁移完成，重启输入法生效：ibus restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
