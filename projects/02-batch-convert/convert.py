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

import particles                              # noqa: E402

SUBTITLE_SUFFIXES = (".srt", ".vtt")
# 守卫的语境实证上限与 experiments/exp_yue_guard.py 的评测口径一致（7 字）。
# 转换单位默认就取这个值：单位 ≤ 7 字 → 全部走守卫严格路径（有实证才改），
# 长句宽松路径（known-yue 放行）不再被触发，句中成语不会被静默改坏。
GUARD_CTX_MAX_LEN = 7
DEFAULT_UNIT_LEN = GUARD_CTX_MAX_LEN

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 保护片段：SRT/VTT 内嵌标签、ASS 覆写标签、HTML 实体、URL、邮箱。
# 这些东西喂给神经模型会被当正文翻坏（unknown token 直接丢字），切出来原样回填。
PROTECT_RE = re.compile(
    r"<[^<>]{1,60}>"                       # <i> </i> <v 名字>
    r"|\{[^{}]{0,60}}"                     # {\an8} 等 ASS 覆写标签
    r"|&[a-zA-Z]+;|&#\d+;"                 # HTML 实体
    r"|https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%\[\]-]+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# 高信号粤语专用字：片段里出现 ≥2 个就认为已是粤语，--yue-detect 时跳过转换。
YUE_ONLY_CHARS = set("唔嘅哋喺冇佢嘢咗啲睇咁攞畀嗰嚟啱乜冧諗")

# 字幕 QC 阈值（Netflix 中文 Timed Text 指南口径：CJK 阅读速度 ~9 字/秒、
# 行宽 ~16 个全角字符、单条时长 5/6 秒 ~ 7 秒）。
CPS_MAX_DEFAULT = 9.0
WIDTH_MAX_DEFAULT = 16
DUR_MIN = 5 / 6
DUR_MAX = 7.0


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


# ---- 术语表 / 保护片段 -------------------------------------------------------

def load_glossary(path: str) -> tuple[dict[str, str], list[str]]:
    """TSV 术语表：`原词<TAB>指定粤语写法`。译文留空 = 保护不改（人名/专名）。

    有译文的条目在转换前替换成指定写法并保护（引擎不会再碰它）。
    """
    force: dict[str, str] = {}
    protect: list[str] = []
    with open(path, encoding="utf-8-sig") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            if "\t" in ln:
                src, dst = ln.split("\t", 1)
                src, dst = src.strip(), dst.strip()
                if not src:
                    continue
                if dst:
                    force[src] = dst
                    if dst not in protect:
                        protect.append(dst)
                elif src not in protect:
                    protect.append(src)
            elif ln.strip() not in protect:      # 光秃秃一行 = 保护不改
                protect.append(ln.strip())
    return force, protect


def split_protected(line: str, protect_words: list[str]) -> list[tuple[str, bool]]:
    """把一行切成有序片段 (文本, 是否保护)。保护片段不送引擎、原样回填。

    匹配来源：PROTECT_RE（标签/URL/邮箱）+ 术语表保护词；重叠时先到先得、
    长的优先。分段顺序与原文一致，拼接无损。
    """
    marks: list[tuple[int, int]] = [(m.start(), m.end()) for m in PROTECT_RE.finditer(line)]
    for w in protect_words:
        start = 0
        while (i := line.find(w, start)) >= 0:
            marks.append((i, i + len(w)))
            start = i + len(w)
    marks.sort(key=lambda se: (se[0], -(se[1] - se[0])))
    spans: list[tuple[str, bool]] = []
    pos = 0
    for s, e in marks:
        if s < pos:
            continue
        if s > pos:
            spans.append((line[pos:s], False))
        spans.append((line[s:e], True))
        pos = e
    if pos < len(line):
        spans.append((line[pos:], False))
    return spans


def looks_cantonese(seg: str) -> bool:
    """高信号粤语专用字 ≥2 个 → 已是粤语，不必再转（防二次转换改坏）。"""
    return sum(ch in YUE_ONLY_CHARS for ch in seg) >= 2


# ---- 可疑片段打分（--flag） ---------------------------------------------------

@dataclass
class Flag:
    where: str
    score: int
    reasons: list[str]
    text: str


def _timing_span(ts: str) -> tuple[float, float] | None:
    """从 "-->" 行解析 (开始, 结束) 挂钟秒；起止不全或倒挂返回 None。"""
    m = list(re.finditer(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})", ts))
    if len(m) < 2:
        return None

    def abs_t(mm) -> float:
        h, mi, s, ms = mm.group(1, 2, 3, 4)
        return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000

    start, end = abs_t(m[0]), abs_t(m[-1])
    return (start, end) if end > start else None


def score_records(where: str, recs: list[Record], cfg: QCConfig,
                  text_lines: list[str] | None = None,
                  timing: str = "", preview: str = "") -> Flag | None:
    """把一段（字幕 cue / 文本行）的信号聚成可疑分。

    计分透明：守卫修复 ×2 + 语气词兜底 ×1 + CPS 超标 ×3 + 行宽/时长越界 ×1。
    text_lines 传字幕 cue 的行（含时间轴时才算 CPS/行宽/时长）；纯文本行
    只传 preview，不做排版 QC。
    """
    reasons: list[str] = []
    score = 0
    fixes = sum(r.guard_fixes for r in recs)
    parts = sum(r.particle for r in recs)
    if fixes:
        reasons.append(f"守卫修复{fixes}段")
        score += 2 * fixes
    if parts:
        reasons.append(f"语气词兜底{parts}处")
        score += parts
    if cfg.enabled and text_lines:
        visible = [PROTECT_RE.sub("", ln) for ln in text_lines]
        span = _timing_span(timing)
        if span is not None:
            dur = span[1] - span[0]
            cps = sum(len(v) for v in visible) / max(dur, 0.1)
            if cps > cfg.cps_max:
                reasons.append(f"CPS {cps:.1f}>{cfg.cps_max:g}")
                score += 3
            if not cfg.dur_min <= dur <= cfg.dur_max:
                reasons.append(f"时长{dur:.2f}s越界")
                score += 1
        width = max((len(v) for v in visible), default=0)
        if width > cfg.width_max:
            reasons.append(f"行宽{width}>{cfg.width_max}")
            score += 1
    if not reasons:
        return None
    preview = preview or " ".join(text_lines or [])
    return Flag(where, score, reasons, preview[:30])


@dataclass
class QCConfig:
    enabled: bool = False
    cps_max: float = CPS_MAX_DEFAULT
    width_max: int = WIDTH_MAX_DEFAULT
    dur_min: float = DUR_MIN
    dur_max: float = DUR_MAX


# ---- 单句管线 ---------------------------------------------------------------

@dataclass
class Record:
    """一段的统计：来源、守卫是否修过、语气词兜底、改动比例。"""
    where: str          # 来源位置（文本行号 / 字幕序号）
    src: str
    dst: str
    source: str         # "neural" | "rule"
    guard_fixes: int    # 守卫修复次数（0/1）
    ratio: float        # 改动字数比例（按繁体字形对齐后统计）
    particle: int = 0   # 句尾语气词兜底转换次数（0/1）


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
                 words: set[str] | None = None,
                 force: dict[str, str] | None = None,
                 protect_words: list[str] | None = None,
                 skip_yue: bool = False):
        self.conv = conv
        self.t2s = t2s
        self.max_len = max_len
        self.words = load_mandarin_words() if words is None else words
        self.force = force or {}
        self.protect_words = protect_words or []
        self.skip_yue = skip_yue
        self.skipped = 0            # --yue-detect 命中数
        self.records: list[Record] = []

    def convert_line(self, line: str, where: str = "") -> str:
        """转换一行：强制替换 → 切保护片段 → 逐段转换 → 拼回。空行原样返回。"""
        if not line.strip():
            return line
        for k in sorted(self.force, key=len, reverse=True):
            line = line.replace(k, self.force[k])
        out: list[str] = []
        for text, protected in split_protected(line, self.protect_words):
            out.append(text if protected else self._convert_span(text, where))
        return "".join(out)

    def _convert_span(self, text: str, where: str) -> str:
        if not _CJK_RE.search(text):
            return text
        out: list[str] = []
        for seg in sentences.split_sentences(text, self.max_len, self.words):
            if not _CJK_RE.search(seg):     # 纯标点：直通
                out.append(seg)
                continue
            if self.skip_yue and looks_cantonese(seg):
                self.skipped += 1
                out.append(seg)
                continue
            final, source, fixes = convert_one(self.conv, seg)
            final, pfix = particles.fix_final_particle(seg, final)
            src_disp, dst_disp = seg, final
            if self.t2s is not None:
                src_disp = self.t2s.to_traditional(seg)
                dst_disp = self.t2s.to_traditional(final)
            self.records.append(Record(where, src_disp, dst_disp,
                                       source, fixes, change_ratio(src_disp, dst_disp),
                                       int(pfix)))
            out.append(dst_disp)
        return "".join(out)

    def summary(self) -> str:
        n = len(self.records)
        if not n:
            return "共 0 段"
        nn = sum(r.source == "neural" for r in self.records)
        fixes = sum(r.guard_fixes for r in self.records)
        parts = sum(r.particle for r in self.records)
        avg = sum(r.ratio for r in self.records) / n
        extra = f" | 语气词兜底 {parts} 处" if parts else ""
        skip = f" | 已是粤语跳过 {self.skipped} 段" if self.skipped else ""
        return (f"共 {n} 段 | 神经 {nn} / 规则 {n - nn} | "
                f"守卫修复 {fixes} 段{extra}{skip} | 平均改动 {avg:.0%}")


def print_stats(records: list[Record], width: int = 24, fh=None) -> None:
    fh = fh or sys.stderr
    print(f"{'#':>4}  来源   守卫  语  改动   原文 → 译文", file=fh)
    for i, r in enumerate(records, 1):
        if r.source == "neural":
            tag = "神经+修" if r.guard_fixes else "神经"
        else:
            tag = "规则"
        src = r.src if len(r.src) <= width else r.src[:width - 1] + "…"
        dst = r.dst if len(r.dst) <= width else r.dst[:width - 1] + "…"
        print(f"{i:>4}  {tag}    {r.guard_fixes:>2}   {r.particle or ' ':1}   "
              f"{r.ratio:>3.0%}   {src} → {dst}", file=fh)
    print("-" * 60, file=fh)


def print_flag_report(flags: list[Flag], limit: int, fh=None) -> None:
    fh = fh or sys.stderr
    if not flags:
        print("无可疑片段。", file=fh)
        return
    flags = sorted(flags, key=lambda f: -f.score)
    print(f"可疑片段 Top {min(limit, len(flags))}/{len(flags)} "
          f"（守卫×2 + 语气词×1 + CPS×3 + 行宽/时长越界×1）：", file=fh)
    for f in flags[:limit]:
        print(f"  {f.where}  [{f.score}]  {'；'.join(f.reasons)}  {f.text}", file=fh)


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


def convert_text(text: str, pipe: Pipeline, cfg: QCConfig | None = None) -> tuple[str, list[Flag]]:
    lines = text.splitlines()
    out: list[str] = []
    flags: list[Flag] = []
    for i, ln in enumerate(lines):
        rec_start = len(pipe.records)
        out.append(pipe.convert_line(ln, where=f"行{i + 1}"))
        if cfg:
            f = score_records(f"行{i + 1}", pipe.records[rec_start:], cfg,
                              preview=out[-1])
            if f:
                flags.append(f)
    return "\n".join(out) + ("\n" if out else ""), flags


def convert_subtitle(text: str, pipe: Pipeline,
                     cfg: QCConfig | None = None) -> tuple[str, list[Flag]]:
    items = parse_subtitle(text)
    out: list[tuple[str, object]] = []
    flags: list[Flag] = []
    cue_no = 0
    for kind, payload in items:
        if kind == "cue":
            cue_no += 1
            rec_start = len(pipe.records)
            new_text = [pipe.convert_line(ln, where=f"字幕{cue_no}")
                        for ln in payload.text]
            if cfg:
                f = score_records(f"字幕{cue_no}", pipe.records[rec_start:], cfg,
                                  text_lines=new_text, timing=payload.timing)
                if f:
                    flags.append(f)
            out.append(("cue", Cue(payload.head, payload.timing, new_text)))
        else:
            out.append((kind, payload))
    result = render_subtitle(out, final_newline=text.endswith("\n") or not text)
    return result, flags


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
    ap.add_argument("--glossary", metavar="FILE",
                    help="术语表（TSV：原词<TAB>指定粤语写法；译文留空=保护不改）")
    ap.add_argument("--flag", nargs="?", const=10, type=int, default=0,
                    metavar="N",
                    help="输出可疑片段 Top N（默认 10）：守卫×2+语气词+CPS×3+越界×1")
    ap.add_argument("--cps", type=float, default=CPS_MAX_DEFAULT,
                    help="字幕阅读速度上限（字/秒，Netflix 中文指南约 9；0=不查）")
    ap.add_argument("--line-width", type=int, default=WIDTH_MAX_DEFAULT,
                    help="字幕行宽上限（全角字符数，0=不查）")
    ap.add_argument("--no-nn", action="store_true", help="禁用神经模型，纯规则转换")
    ap.add_argument("--asis", action="store_true",
                    help="不做繁体归一（默认把输出统一成繁体字形）")
    ap.add_argument("--yue-detect", action="store_true",
                    help="检测到已是粤语的片段直接跳过（防二次转换改坏）")
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

    force: dict[str, str] = {}
    protect: list[str] = []
    if args.glossary:
        if not os.path.exists(args.glossary):
            print(f"找不到术语表：{args.glossary}", file=sys.stderr)
            return 2
        force, protect = load_glossary(args.glossary)

    conv = load_engine(use_nn=not args.no_nn)
    if conv.nn is None and not args.no_nn:
        print("提示：神经模型未加载（缺 onnxruntime 或 data/yue_nn.onnx），退回纯规则。",
              file=sys.stderr)
    t2s = None if args.asis else Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
    pipe = Pipeline(conv, t2s=t2s, max_len=args.max_len, force=force,
                    protect_words=protect, skip_yue=args.yue_detect)
    cfg = QCConfig(enabled=bool(args.flag), cps_max=args.cps or float("inf"),
                   width_max=args.line_width or 10 ** 6)

    if kind == "subtitle":
        result, flags = convert_subtitle(text, pipe, cfg)
    else:
        result, flags = convert_text(text, pipe, cfg)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result)
    else:
        sys.stdout.write(result)
    if args.stats:
        print_stats(pipe.records)
        print(pipe.summary(), file=sys.stderr)
    if args.flag:
        print_flag_report(flags, args.flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
