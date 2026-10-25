#!/usr/bin/env python
"""普通话 → 粤语批量转换 CLI：整篇文章、SRT/VTT 字幕。

复用输入法引擎（src/ipinyin/yue.py + yue_guard.py + 可选 yue_nn.py），
但服务的是批量场景。核心约束：**转换单位 ≤ 7 字**（默认，= 守卫严格路径
阈值）——长句不一次性喂给引擎，而是切成 ≤7 字的片段，每个片段都过守卫
的「有实证才改」校验，成语/古诗混在句中也不会被静默改坏（亡羊補監 类）。
切分顺序：终端标点 → 次级标点 → 硬切（词库护界，不切断词语）。
每片段走「神经 → 守卫 → 规则回退」管线，最后可选统一为繁体。

字幕只替换文本行，时间轴/序号/WEBVTT 头/NOTE 一律原样保留。

用法：
  python convert.py 文章.txt [--stats] [--out out.txt]
  python convert.py 字幕.srt [--stats] [--out out.srt]
  cat 文本 | python convert.py - --stats

用法详见同目录 README.md。
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ipinyin.t2s import Converter            # noqa: E402
from ipinyin.yue import YueConverter          # noqa: E402
from ipinyin.yue_guard import YueGuard        # noqa: E402
from ipinyin.yue_nn import NeuralYue          # noqa: E402

import sentences                              # noqa: E402

SUBTITLE_SUFFIXES = (".srt", ".vtt")
# 守卫的语境实证上限与 experiments/exp_yue_guard.py 的评测口径一致（7 字）。
# 转换单位默认就取这个值：单位 ≤ 7 字 → 全部走守卫严格路径（有实证才改），
# 长句宽松路径（known-yue 放行）不再被触发，句中成语不会被静默改坏。
GUARD_CTX_MAX_LEN = 7
DEFAULT_UNIT_LEN = GUARD_CTX_MAX_LEN

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ---- 引擎加载 -------------------------------------------------------------

def load_engine(root: str = ROOT, use_nn: bool = True) -> YueConverter:
    """按 exp_yue_guard.py 的方式装好 规则 + 守卫（+ 神经）。"""
    data = os.path.join(root, "data")
    conv = YueConverter.load_dict(
        os.path.join(data, "yue.marshal"),
        YueConverter.load(os.path.join(data, "yue.tsv"))._map)
    conv.guard = YueGuard.load(os.path.join(data, "yue_guard.marshal"))
    if conv.guard is not None:
        conv.guard.ctx_max_len = GUARD_CTX_MAX_LEN
    if use_nn:
        conv.nn = NeuralYue.load(data)
    return conv


# ---- 词库（切句护界用）------------------------------------------------------

_lexicon_words: set[str] | None = None


def load_mandarin_words(root: str = ROOT) -> set[str]:
    """data/lexicon.tsv 第二列（普通话词，2~8 字），硬切分句时避开词中间。

    找不到文件就返回空集，分句退化为纯标点/长度策略。进程内只读一次。
    """
    global _lexicon_words
    if _lexicon_words is None:
        words: set[str] = set()
        path = os.path.join(root, "data", "lexicon.tsv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for ln in fh:
                    parts = ln.rstrip("\n").split("\t")
                    if len(parts) >= 2 and 2 <= len(parts[1]) <= 8:
                        words.add(parts[1])
        _lexicon_words = words
    return _lexicon_words


# ---- 单句管线 ---------------------------------------------------------------

@dataclass
class Record:
    """一句的统计：来源、守卫是否修过、改动比例。"""
    where: str          # 来源位置（文本行号 / 字幕序号）
    src: str
    dst: str
    source: str         # "neural" | "rule"
    guard_fixes: int    # 守卫修复次数（0/1）
    ratio: float        # 改动字数比例（按繁体字形对齐后统计）


def convert_one(conv: YueConverter, text: str) -> tuple[str, str, int]:
    """单句转换，返回 (译文, 来源, 守卫修复次数)。

    与 YueConverter.convert() 的神经分支逐行对应（test_convert.py 有 parity
    测试盯住两者一致）；复刻是为了拿到引擎不外露的统计信息。
    """
    if conv.nn is not None:
        try:
            raw = conv.nn.translate(text)
        except Exception:
            raw = ""
        if raw:
            final = raw
            if conv.guard is not None:
                final = conv.guard.check(
                    text, raw, repair=lambda seg: conv.convert(seg, neural=False))
                if final != raw:
                    return final, "neural", 1
            if final:
                return final, "neural", 0
    return conv.convert(text, neural=False), "rule", 0


def change_ratio(src: str, dst: str) -> float:
    """改动字数比例：difflib 对齐后，原文中被替换/删除的字符占比。

    繁简字形差异也计入「改动」——它衡量的是「这句被改得多深、
    需不需要人工复核」，不是译文质量分。
    """
    if not src:
        return 0.0
    matcher = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    changed = sum(i2 - i1 for tag, i1, i2, _, _ in matcher.get_opcodes()
                  if tag != "equal")
    return changed / len(src)


class Pipeline:
    """逐段转换 + 统计收集。t2s 用于把输出统一成繁体。"""

    def __init__(self, conv: YueConverter, t2s: Converter | None = None,
                 max_len: int = DEFAULT_UNIT_LEN,
                 words: set[str] | None = None):
        self.conv = conv
        self.t2s = t2s
        self.max_len = max_len
        self.words = load_mandarin_words() if words is None else words
        self.records: list[Record] = []

    def convert_line(self, line: str, where: str = "") -> str:
        """转换一行：分句 → 逐段转换 → 拼回。空行原样返回。"""
        if not line.strip():
            return line
        out: list[str] = []
        for seg in sentences.split_sentences(line, self.max_len, self.words):
            if not _CJK_RE.search(seg):        # 纯标点/拉丁：引擎管不着，直通
                out.append(seg)
                continue
            final, source, fixes = convert_one(self.conv, seg)
            src_disp, dst_disp = seg, final
            if self.t2s is not None:
                src_disp = self.t2s.to_traditional(seg)
                dst_disp = self.t2s.to_traditional(final)
            self.records.append(Record(where, src_disp, dst_disp,
                                       source, fixes, change_ratio(src_disp, dst_disp)))
            out.append(dst_disp)
        return "".join(out)

    def summary(self) -> str:
        n = len(self.records)
        if not n:
            return "共 0 段"
        nn = sum(r.source == "neural" for r in self.records)
        fixes = sum(r.guard_fixes for r in self.records)
        avg = sum(r.ratio for r in self.records) / n
        return (f"共 {n} 段 | 神经 {nn} / 规则 {n - nn} | "
                f"守卫修复 {fixes} 段 | 平均改动 {avg:.0%}")


def print_stats(records: list[Record], width: int = 24, fh=None) -> None:
    fh = fh or sys.stderr
    print(f"{'#':>4}  来源   守卫  改动   原文 → 译文", file=fh)
    for i, r in enumerate(records, 1):
        if r.source == "neural":
            tag = "神经+修" if r.guard_fixes else "神经"
        else:
            tag = "规则"
        src = r.src if len(r.src) <= width else r.src[:width - 1] + "…"
        dst = r.dst if len(r.dst) <= width else r.dst[:width - 1] + "…"
        print(f"{i:>4}  {tag}    {r.guard_fixes:>2}   {r.ratio:>3.0%}   "
              f"{src} → {dst}", file=fh)
    print("-" * 60, file=fh)


# ---- 字幕解析（SRT / VTT） ---------------------------------------------------

@dataclass
class Cue:
    head: list[str]     # 序号行（SRT）/ 标识行（VTT），原样保留
    timing: str         # "-->" 时间轴行，原样保留
    text: list[str]     # 可翻译的文本行


_TIMING_RE = re.compile(r"\d{1,2}:\d{2}[.,]\d{1,3}\s*-->")


def parse_subtitle(text: str) -> list[tuple[str, object]]:
    """解析成条目序列：("cue", Cue) 或 ("line", str)。

    空行也作为 ("line", "") 记下，重组时逐条回放 —— 时间轴、序号、
    WEBVTT 头、NOTE 块、空行分布全部原样，只有 cue 的 text 行会被换掉。
    """
    items: list[tuple[str, object]] = []
    head: list[str] = []
    timing: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal head, timing, body
        if timing is None:
            items.extend(("line", ln) for ln in head)
        else:
            items.append(("cue", Cue(head, timing, body)))
        head, timing, body = [], None, []

    for ln in text.splitlines():
        if ln.strip() == "":
            flush()
            items.append(("line", ""))
        elif timing is None and _TIMING_RE.search(ln):
            timing = ln
        elif timing is None:
            head.append(ln)
        else:
            body.append(ln)
    flush()
    return items


def render_subtitle(items: list[tuple[str, object]], final_newline: bool = True) -> str:
    lines: list[str] = []
    for kind, payload in items:
        if kind == "cue":
            lines.extend(payload.head)
            lines.append(payload.timing)
            lines.extend(payload.text)
        else:
            lines.append(payload)
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if final_newline else "")


# ---- CLI -------------------------------------------------------------------

def detect_kind(path: str) -> str:
    return "subtitle" if path.lower().endswith(SUBTITLE_SUFFIXES) else "text"


def convert_text(text: str, pipe: Pipeline) -> str:
    lines = [pipe.convert_line(ln, where=f"行{i + 1}")
             for i, ln in enumerate(text.splitlines())]
    return "\n".join(lines) + ("\n" if lines else "")


def convert_subtitle(text: str, pipe: Pipeline) -> str:
    items = parse_subtitle(text)
    out: list[tuple[str, object]] = []
    cue_no = 0
    for kind, payload in items:
        if kind == "cue":
            cue_no += 1
            new_text = [pipe.convert_line(ln, where=f"字幕{cue_no}")
                        for ln in payload.text]
            out.append(("cue", Cue(payload.head, payload.timing, new_text)))
        else:
            out.append((kind, payload))
    return render_subtitle(out, final_newline=text.endswith("\n") or not text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="普通话 → 粤语批量转换（文章 / SRT / VTT）")
    ap.add_argument("input", help="输入文件（.txt / .srt / .vtt），或 - 表示 stdin 纯文本")
    ap.add_argument("--stats", action="store_true",
                    help="向 stderr 输出逐句统计（来源 / 守卫修复 / 改动比例）")
    ap.add_argument("--out", help="输出文件（默认写到 stdout）")
    ap.add_argument("--max-len", type=int, default=DEFAULT_UNIT_LEN,
                    help=f"转换单位长度上限（默认 {DEFAULT_UNIT_LEN} = 守卫严格路径阈值，"
                         f"每个单位都有实证校验；调大则回到一次性整句转换，物理上限 38）")
    ap.add_argument("--no-nn", action="store_true", help="禁用神经模型，纯规则转换")
    ap.add_argument("--asis", action="store_true",
                    help="不做繁体归一（默认把输出统一成繁体字形）")
    args = ap.parse_args(argv)

    if args.input == "-":
        text, kind = sys.stdin.read(), "text"
    else:
        if not os.path.exists(args.input):
            print(f"找不到输入文件：{args.input}", file=sys.stderr)
            return 2
        with open(args.input, encoding="utf-8-sig") as fh:
            text = fh.read()
        kind = detect_kind(args.input)

    conv = load_engine(use_nn=not args.no_nn)
    if conv.nn is None and not args.no_nn:
        print("提示：神经模型未加载（缺 onnxruntime 或 data/yue_nn.onnx），退回纯规则。",
              file=sys.stderr)
    t2s = None if args.asis else Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
    pipe = Pipeline(conv, t2s=t2s, max_len=args.max_len)

    result = convert_subtitle(text, pipe) if kind == "subtitle" else convert_text(text, pipe)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result)
    else:
        sys.stdout.write(result)
    if args.stats:
        print_stats(pipe.records)
        print(pipe.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
