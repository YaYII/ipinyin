#!/usr/bin/env python3
"""粤语 TTS 读音前端（G2P）：普通话文本 → 粤语书面文本 → 粤拼音节 → 音素序列。

管线（引擎加载方式对齐 experiments/exp_yue_guard.py 与 src/ipinyin/engine.py）：

  普通话输入
    │  ① normalize：全角→半角、数字→粤语读法（年份逐位 / 数量进位）、
    │     标点归一到 , . ! ?（TTS 停顿 token）
    ▼
  ② 普→粤转换：YueConverter（yue_rules 语序 + 词表；--nn 时神经模型 + YueGuard 守卫）
    ▼
  ③ 繁化：t2s.Converter.to_traditional —— 粤拼表 data/yue.marshal 以繁体字为键，
     转换产物里的简体字（迟/还/这…）不繁化会整字缺读（实测覆盖率差距巨大）
    ▼
  ④ 粤拼切分：词级最长匹配（data/yue.marshal 的 words，最长 20 字）优先，
     字级兜底（chars），无读法的字进缺字报告
    ▼
  ⑤ 音素化：phonemes.split_syllable 拆 声母+韵母+声调，
     再映射 Piper 风格 IPA token（占位方案，见 phonemes.py）

  输出：对齐 token 流 + 粤拼序列 + 音素序列

用法：
  python3 projects/03-tts-g2p/g2p.py "今天天气很好。"
  python3 projects/03-tts-g2p/g2p.py --file projects/03-tts-g2p/samples.txt
  python3 projects/03-tts-g2p/g2p.py "..." --json        # 机器可读
  python3 projects/03-tts-g2p/g2p.py "..." --nn          # 神经转换（需 onnxruntime）

已知的数字处理边界（TODO，见 README「已知局限」）：
  小数（3.5）、百分数（50%）、电话号码、负数、分数、超长编号一律保持原样不转。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin.t2s import Converter          # noqa: E402
from ipinyin.yue import YueConverter       # noqa: E402

from phonemes import split_syllable, syllable_to_tokens  # noqa: E402

# ---- 数字读法（简化汉字，后续过 convert + t2s 自然变繁）--------------------
CN_DIGITS = "零一二三四五六七八九"
MEASURE_WORDS = "个只只条张件位杯碗瓶盒支把岁点次遍块间层套双本堂餐杯份"


def int_to_cn(num: int, leading: bool = True) -> str:
    """整数 → 汉语数量读法（标准进位）。0<=n 常见范围；一万以上递归到亿。

    例：0→零 10→十 11→十一 20→二十 105→一百零五 110→一百一十
        1005→一千零五 10010→一万零一十 123456789→一亿二千三百四十五万六千七百八十九

    leading=False 用于进位单位之后的尾巴：10~19 要读「一十」
    （一百一十 / 一千零一十 / 一万零一十）， standalone 才是「十」。
    """
    if num < 10:
        return CN_DIGITS[num]
    if num < 20:
        return ("一" if not leading else "") + "十" + (CN_DIGITS[num - 10] if num > 10 else "")
    if num < 100:
        q, r = divmod(num, 10)
        return CN_DIGITS[q] + "十" + (CN_DIGITS[r] if r else "")
    if num < 1000:
        q, r = divmod(num, 100)
        tail = "零" + int_to_cn(r, False) if 0 < r < 10 else (int_to_cn(r, False) if r else "")
        return CN_DIGITS[q] + "百" + tail
    if num < 10000:
        q, r = divmod(num, 1000)
        tail = "零" + int_to_cn(r, False) if 0 < r < 100 else (int_to_cn(r, False) if r else "")
        return CN_DIGITS[q] + "千" + tail
    if num < 10 ** 8:
        q, r = divmod(num, 10000)
        tail = "零" + int_to_cn(r, False) if 0 < r < 1000 else (int_to_cn(r, False) if r else "")
        return int_to_cn(q) + "万" + tail
    q, r = divmod(num, 10 ** 8)
    tail = "零" + int_to_cn(r, False) if 0 < r < 10 ** 7 else (int_to_cn(r, False) if r else "")
    return int_to_cn(q) + "亿" + tail


def digits_to_cn(num_str: str) -> str:
    """数字串逐位读（年份/编号口径）：2026 → 二零二六。"""
    return "".join(CN_DIGITS[int(ch)] for ch in num_str)


def _quantity(num_str: str) -> str:
    """数字串 → 数量读法；带前导零的（编号/电话口径）逐位读。"""
    if len(num_str) > 1 and num_str.lstrip("0") != num_str:
        return digits_to_cn(num_str)          # 前导零当编号，逐位读
    return int_to_cn(int(num_str))


# ---- 规范化（数字 + 标点）--------------------------------------------------
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")

_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?=年)")
_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})(?=月)")
_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})(?=日|号)")
_CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2})(?=点)")
# 千分位组或普通整数；后面跟 . % 的不转（小数/百分数 TODO）
_INT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?![\d.%])")
# 单数量词前的「二」→「两」（第/老 开头、进位组合如十二 不动）
_TWO_RE = re.compile(rf"(?<![零一二三四五六七八九十百千万第老\d])二(?=[{MEASURE_WORDS}])")

def _int_sub(m: re.Match) -> str:
    """整数 token → 数量读法；11 位以上当电话/编号，保持原样（TODO：逐位读）。"""
    digits = m.group(1).replace(",", "")
    if len(digits) >= 11:
        return m.group(1)
    return _quantity(digits)


# 标点归一：一切停顿 → , ；句末 → . ! ?
_PUNCT_MAP = {"，": ",", "、": ",", "；": ",", "：": ",", ",": ",",
              "。": ".", "．": ".", ".": ".",
              "！": "!", "!": "!", "？": "?", "?": "?",
              "…": ",", "—": ",", "－": ",", "～": ",", "·": ""}
_STRIP_CHARS = "\"'“”‘’「」『』《》〈〉（）()【】[]{}<>|/\\\u3000"


def normalize(text: str) -> str:
    """① 全角数字→半角；数字→粤语读法（年份逐位/数量进位）；标点归一。"""
    text = text.translate(_FULLWIDTH).strip()

    # 年份：4 位数 + 年 → 逐位（2026年 → 二零二六年）
    text = _YEAR_RE.sub(lambda m: digits_to_cn(m.group(1)), text)
    # 月 / 日（号）/ 点钟：小整数 → 数量读法（10月25日 → 十月二十五日）
    text = _MONTH_RE.sub(lambda m: _quantity(m.group(1)), text)
    text = _DAY_RE.sub(lambda m: _quantity(m.group(1)), text)
    text = _CLOCK_RE.sub(lambda m: _quantity(m.group(1)), text)
    # 其余整数 → 数量读法（123 → 一百二十三；千分位 1,234 → 一千二百三十四）
    text = _INT_RE.sub(_int_sub, text)
    # 单数量词前的二 → 两（2个 → 两個）。TODO：更完整的变调/量词搭配
    text = _TWO_RE.sub("两", text)

    # 标点归一 + 去装饰符号
    out: list[str] = []
    for ch in text:
        if ch in _PUNCT_MAP:
            out.append(_PUNCT_MAP[ch])
        elif ch in _STRIP_CHARS:
            continue
        else:
            out.append(ch)
    return "".join(out)


# ---- 管线主体 ---------------------------------------------------------------
def _is_han(ch: str) -> bool:
    """汉字判定：基本区 + 扩展A（粤拼表有 6458 个）+ 扩展B（有 1981 个，如 𠝹）。"""
    return ("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"
            or "\U00020000" <= ch <= "\U0002A6DF")


@dataclass
class Token:
    """对齐 token：surface 在粤语书面文本上；syllables 是粤拼音节。"""
    kind: str                       # word | char | missing | punct | latin
    surface: str
    syllables: list[str] = field(default_factory=list)


@dataclass
class G2PResult:
    raw: str
    normalized: str
    yue: str                        # 粤语书面（繁体）
    tokens: list[Token] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # 无读法的字，按出现序去重
    jyutping: list[str] = field(default_factory=list)
    phonemes: list[str] = field(default_factory=list)  # Piper 风格占位 token


class G2P:
    """普通话 → 粤拼 → 音素 的前端引擎。线程不安全（neural 缓存在转换器里）。"""

    def __init__(self, neural: bool = False):
        self.conv = YueConverter.load_dict(
            os.path.join(ROOT, "data", "yue.marshal"),
            YueConverter.load(os.path.join(ROOT, "data", "yue.tsv"))._map)
        self.t2s = Converter.load(os.path.join(ROOT, "data", "s2t.marshal"))
        if neural:
            # 与 engine.py 同款加载：模型在 data/ 下才启用；守卫始终尝试加载
            from ipinyin.yue_nn import NeuralYue
            from ipinyin.yue_guard import YueGuard
            self.conv.nn = NeuralYue.load(os.path.join(ROOT, "data"))
            if self.conv.nn is None:
                print("警告：神经模型不可用（缺 onnxruntime 或 yue_nn.onnx），退回规则版",
                      file=sys.stderr)
            self.conv.guard = YueGuard.load(os.path.join(ROOT, "data", "yue_guard.marshal"))
        self._max_word = max((len(k) for k in self.conv._words), default=1)

    # ②③ 普→粤 → 繁化
    def to_yue(self, text: str) -> str:
        return self.t2s.to_traditional(self.conv.convert(text))

    # ④ 粤拼切分：词级最长匹配优先，字级兜底，缺字报告
    def segment(self, yue_text: str) -> list[Token]:
        tokens: list[Token] = []
        i, n = 0, len(yue_text)
        while i < n:
            ch = yue_text[i]
            if ch in _PUNCT_MAP.values() or ch in _STRIP_CHARS:
                if not ch.isspace():
                    tokens.append(Token("punct", ch))
                i += 1
                continue
            if ch.isspace():
                i += 1
                continue
            if _is_han(ch):
                matched = False
                for length in range(min(self._max_word, n - i), 1, -1):
                    reading = self.conv._words.get(yue_text[i:i + length])
                    if reading:
                        tokens.append(Token("word", yue_text[i:i + length], reading.split()))
                        i += length
                        matched = True
                        break
                if matched:
                    continue
                reading = self.conv._chars.get(ch)
                if reading:
                    tokens.append(Token("char", ch, [reading]))
                else:
                    tokens.append(Token("missing", ch))
                i += 1
                continue
            # 拉丁字母 / 未转成汉字的数字等：整段收一个 token（TODO：拉丁 G2P）
            j = i
            while j < n and not (_is_han(yue_text[j])
                                 or yue_text[j].isspace()
                                 or yue_text[j] in _PUNCT_MAP.values()):
                j += 1
            tokens.append(Token("latin", yue_text[i:j]))
            i = j
        return tokens

    # ⑤ 音素化
    def phonemize(self, tokens: list[Token]) -> tuple[list[str], list[str], list[str]]:
        """返回 (粤拼序列, 音素序列, 缺字列表)。"""
        jyutping: list[str] = []
        phonemes: list[str] = []
        missing: list[str] = []
        for tok in tokens:
            if tok.kind == "punct":
                phonemes.append(tok.surface)      # 停顿 token 保留
                continue
            for syl in tok.syllables:
                jyutping.append(syl)
                ph = syllable_to_tokens(syl)
                if ph is None:                    # 表里的坏音节（如 円→jen1，非标拼写）
                    missing.append(f"{tok.surface}:{syl}")
                    continue
                phonemes.extend(ph)
            if tok.kind == "missing":
                missing.append(tok.surface)
        return jyutping, phonemes, missing

    def __call__(self, text: str) -> G2PResult:
        normalized = normalize(text)
        yue = self.to_yue(normalized)
        tokens = self.segment(yue)
        jyutping, phonemes, missing = self.phonemize(tokens)
        return G2PResult(raw=text, normalized=normalized, yue=yue,
                         tokens=tokens, missing=missing,
                         jyutping=jyutping, phonemes=phonemes)


# ---- CLI --------------------------------------------------------------------
def _print_result(res: G2PResult) -> None:
    print(f"原文   : {res.raw}")
    print(f"规范   : {res.normalized}")
    print(f"粤语   : {res.yue}")
    print(f"粤拼   : {' '.join(res.jyutping) or '（无）'}")
    print(f"音素   : {' '.join(res.phonemes) or '（无）'}")
    if res.missing:
        print(f"缺读法 : {' '.join(res.missing)}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="普通话 → 粤语 → 粤拼 → 音素（G2P 前端）")
    ap.add_argument("text", nargs="*", help="要处理的文本；不给则读 --file 或 stdin")
    ap.add_argument("--file", help="按行处理一个文本文件")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--nn", action="store_true", help="启用神经转换（onnxruntime）")
    args = ap.parse_args()

    lines: list[str] = []
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    elif args.text:
        lines = [" ".join(args.text)]
    else:
        lines = [ln.strip() for ln in sys.stdin if ln.strip()]

    g2p = G2P(neural=args.nn)
    for line in lines:
        res = g2p(line)
        if args.json:
            print(json.dumps({
                "raw": res.raw, "normalized": res.normalized, "yue": res.yue,
                "jyutping": res.jyutping, "phonemes": res.phonemes,
                "missing": res.missing,
                "tokens": [{"kind": t.kind, "surface": t.surface,
                            "syllables": t.syllables} for t in res.tokens],
            }, ensure_ascii=False))
        else:
            _print_result(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
