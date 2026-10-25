"""粤语神经解码的约束守卫（P1）。

神经模型口语词转得好（不懂→唔識，+30pt），但会把成语/古诗改坏
（亡羊補牢→亡羊補監）。根因是模型没有「该不该改」的判据。

守卫的做法：神经输出与原文对齐后，逐段检查每个改动 ——
替换要么是**有实证的**（人工词表 / 短语表 / 平行语料对齐里反复出现），
要么整段否掉，回退规则版。语料里从没出现过的改法（牢→監）过不去。

数据由 tools/build_yue_guard.py 生成（data/yue_guard.marshal）；
没有数据文件时 check() 恒真，行为与无守卫完全一致。
"""
from __future__ import annotations

import difflib
import marshal
import os
import zlib
from typing import Callable


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


class YueGuard:
    __slots__ = ("word_pairs", "singles", "char_pairs", "left", "right", "vocab",
                 "colloquial", "min_char", "min_ctx", "ctx_max_len")

    def __init__(self, data: dict, min_char: int = 10, min_ctx: int = 3,
                 ctx_max_len: int = 5):
        self.word_pairs: dict[str, list[str]] = data.get("word_pairs", {})
        self.singles: dict[str, str] = data.get("singles", {})
        self.char_pairs: dict = data.get("char_pairs", {})
        self.left: dict = data.get("left", {})
        self.right: dict = data.get("right", {})
        self.vocab: set[str] = set(data.get("vocab", ()))
        self.colloquial: set[str] = set(data.get("colloquial", ""))
        self.min_char = min_char
        self.min_ctx = min_ctx
        # 语境实证只对短输入生效（≤5 字：词/成语/古诗这一档）。长句是口语流，
        # 逐字语境统计覆盖不住，会把神经在长句上的收益杀掉（实测 44.3%→25.3%）；
        # 语境要防的「古诗/成语里的高频字替换」（是→係、不→唔）只出现在短输入里。
        self.ctx_max_len = ctx_max_len

    @classmethod
    def load(cls, path: str) -> "YueGuard | None":
        if not path or not os.path.exists(path):
            return None
        try:
            raw = open(path, "rb").read()
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass
            data = marshal.loads(raw)
            if not isinstance(data, dict):
                return None
        except (OSError, ValueError, EOFError):
            return None
        return cls(data)

    # ---- 单个改动的判定 -------------------------------------------------

    def _char_ok(self, s: str, d: str) -> bool:
        return self.char_pairs.get((s, d), 0) >= self.min_char

    def _word_ok(self, s_seg: str, d_seg: str) -> bool:
        targets = self.word_pairs.get(s_seg)
        return targets is not None and (targets == d_seg or d_seg in targets)

    def _piece_ctx_ok(self, src: str, i: int, s: str, d: str) -> bool:
        if i > 0 and self.left.get((src[i - 1], s, d), 0) >= self.min_ctx:
            return True
        if i + 1 < len(src) and self.right.get((s, d, src[i + 1]), 0) >= self.min_ctx:
            return True
        return False

    def _decompose(self, src: str, i1: int, i2: int, d_seg: str, strict_ctx: bool) -> bool:
        """把一个改动块拆成「相等或各有实证」的小段（DP）。

        模型会把相邻转换合成一块（这样吗→咁咩 = 这样→咁 + 吗→咩），
        整块对不上实证表不等于改错了，逐段拆开验。
        """
        s_seg = src[i1:i2]
        n, m = len(s_seg), len(d_seg)
        seen = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            i, j = stack.pop()
            while i < n and j < m and s_seg[i] == d_seg[j]:
                i += 1
                j += 1
            if i == n and j == m:
                return True
            for k in range(1, min(5, n - i) + 1):
                sp = s_seg[i:i + k]
                for l in range(1, min(6, m - j) + 1):
                    dp_ = d_seg[j:j + l]
                    if (i + k, j + l) in seen:
                        continue
                    ok = False
                    if k == 1 and l == 1:
                        ok = self._char_ok(sp, dp_)
                        if ok and strict_ctx:
                            ok = self._piece_ctx_ok(src, i1 + i, sp, dp_)
                    else:
                        ok = self._word_ok(sp, dp_)
                    if ok:
                        seen.add((i + k, j + l))
                        stack.append((i + k, j + l))
        return False

    def _known_dst(self, d_seg: str) -> bool:
        """目标写法本身是语料里见过的粤语 n-gram（每片 ≥2 字）。"""
        m = len(d_seg)
        seen = {0}
        stack = [0]
        while stack:
            j = stack.pop()
            if j == m:
                return True
            for l in range(2, min(6, m - j) + 1):
                if d_seg[j:j + l] in self.vocab and (j + l) not in seen:
                    seen.add(j + l)
                    stack.append(j + l)
        return False

    def _covers(self, src: str, dst: str) -> bool:
        """整句分解：src→dst 能否拆成「相等 + 有实证片段」的覆盖。

        自己跑对齐 DP 而不用 difflib —— SequenceMatcher 会把
        多少钱→幾多錢 对齐成「插幾 + 多==多 + 少钱→錢」这种病态块，
        块级校验跟着全错。这里相等直通、片段各自验证，边界无关。
        长句额外允许：已知粤语片段（难道→唔通 这类凑不出替换对的合法
        转换）、口语单字插入（真→真係）、单字缩略删除（知道→知）。
        """
        n, m = len(src), len(dst)
        short = n <= self.ctx_max_len
        seen = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            i, j = stack.pop()
            if i == n and j == m:
                return True
            if i < n and j < m and src[i] == dst[j] and (i + 1, j + 1) not in seen:
                seen.add((i + 1, j + 1))
                stack.append((i + 1, j + 1))
            for k in range(1, min(5, n - i) + 1):
                sp = src[i:i + k]
                for l in range(1, min(6, m - j) + 1):
                    dp_ = dst[j:j + l]
                    if (i + k, j + l) in seen:
                        continue
                    if k == 1 and l == 1:
                        ok = self._char_ok(sp, dp_) and (
                            not short or self._piece_ctx_ok(src, i, sp, dp_))
                    else:
                        ok = self._word_ok(sp, dp_)
                    if ok:
                        seen.add((i + k, j + l))
                        stack.append((i + k, j + l))
            if not short:
                for l in range(2, min(6, m - j) + 1):
                    if dst[j:j + l] in self.vocab and (i, j + l) not in seen:
                        seen.add((i, j + l))
                        stack.append((i, j + l))
                if j < m and (dst[j] in self.colloquial or dst[j] == " ") \
                        and (i, j + 1) not in seen:
                    seen.add((i, j + 1))
                    stack.append((i, j + 1))
                if i < n and (i + 1, j) not in seen:
                    seen.add((i + 1, j))
                    stack.append((i + 1, j))
        return False

    def _allow(self, src: str, i1: int, i2: int, d_seg: str) -> bool:
        s_seg = src[i1:i2]
        if not s_seg or not d_seg:
            return False                      # 无实证的插入/删除不接受
        # 非汉字层（标点/空格/拉丁）：不归模型管，放行
        if not any(_is_cjk(ch) for ch in s_seg) and not any(_is_cjk(ch) for ch in d_seg):
            return True
        strict_ctx = len(src) <= self.ctx_max_len
        # 1) 片段级实证：词表 / 短语表 / 语料对齐挖出的整段替换
        if self._word_ok(s_seg, d_seg):
            return True
        # 2) 逐段拆解：相等 + 有实证的片段
        if self._decompose(src, i1, i2, d_seg, strict_ctx):
            return True
        # 3) 长句：输出是已知粤语写法即可（合法转换未必凑得出替换对：
        #    难道→唔通、兄弟→巴打；单字片段不算，防 牢→監 这类改写）
        if not strict_ctx and self._known_dst(d_seg):
            return True
        return False

    # ---- 整句判定 -------------------------------------------------------

    def check(self, src: str, dst: str, repair: Callable[[str], str] | None = None) -> str | None:
        """检查神经输出 dst。

        整句能分解成「相等 + 有实证片段」（_covers）→ 原样返回 dst；
        否则按 difflib 对齐找无实证的块，只把这些片段替换成
        repair(原文片段)（默认还原原文），其余保留神经输出 ——
        块级修复，不整句否决（整句否决实测会把长句从 44.3% 杀到 25.3%）。
        """
        if not dst or src == dst:
            return dst
        if self._covers(src, dst):
            return dst
        if repair is None:
            repair = lambda seg: seg          # noqa: E731
        matcher = difflib.SequenceMatcher(None, src, dst, autojunk=False)
        opcodes = matcher.get_opcodes()
        bad = [tag != "equal" and not self._allow(src, i1, i2, dst[j1:j2])
               for tag, i1, i2, j1, j2 in opcodes]
        if not any(bad):
            return dst
        if repair is None:
            repair = lambda seg: seg          # noqa: E731
        out: list[str] = []
        pos = 0                               # src 里已生成的部分
        for (tag, i1, i2, j1, j2), is_bad in zip(opcodes, bad):
            if tag == "equal":
                continue
            out.append(src[pos:i1])           # 与前面块之间的相同部分
            if is_bad:
                out.append(repair(src[i1:i2]))
            else:
                out.append(dst[j1:j2])        # 有实证的改动保留神经输出
            pos = i2
        out.append(src[pos:])
        return "".join(out)
