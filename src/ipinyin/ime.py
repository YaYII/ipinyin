"""输入法状态机（纯逻辑，不依赖 IBus，方便单测）。

职责：维护拼音缓冲、生成候选、处理翻页/选词，产出给 UI 用的视图。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import product

from .lexicon import Lexicon
from .segmenter import best_paths_scored
from .t2s import Converter
from .chaoshan import ChaoshanConverter
from .yue import YueConverter
from .user_model import ModeModels, UserModel

PAGE_SIZE = 9
ABBREV_COMBOS = 200      # 简拼展开最多试多少条完整拼音
ABBREV_OPTIONS = 12      # 每个首字母最多展开几个音节（kuang 这种排在 7~9 位，6 个不够）
MAX_BUFFER = 48
WORDS_PER_SYLLABLE = 2
MAX_COMPOSED = 9
SENTENCE_CANDIDATES = 3      # 整句给几条备选：只有一条时打错一个字就没有第二选择

# 快打时最常见的拼音笔误（zh/z、ch/c、sh/s、前后鼻音、ying/yin）
FUZZY_PAIRS = (
    ("zh", "z"),
    ("ch", "c"),
    ("sh", "s"),
    ("ang", "an"),
    ("eng", "en"),
    ("ing", "in"),
    ("ying", "yin"),
    ("iang", "ian"),
    ("uang", "uan"),
)


@dataclass
class View:
    visible: bool
    preedit: str
    candidates: list[str] = field(default_factory=list)   # 显示用（简/繁/粤/粤拼四列）
    words: list[str] = field(default_factory=list)        # 真正会上屏的词
    page: int = 0
    page_count: int = 0
    highlighted: int = 0
    cursor: int = 0          # 预编辑里的光标位置（已转成文字的部分）


def transposition_variants(text: str, limit: int = 8) -> list[str]:
    """相邻字母打反（zhogn→zhong），快打时最常见的手滑。"""
    out: list[str] = []
    for index in range(len(text) - 1):
        if text[index] == text[index + 1]:
            continue
        out.append(text[:index] + text[index + 1] + text[index] + text[index + 2:])
        if len(out) >= limit:
            break
    return out


def fuzzy_variants(text: str, limit: int = 16, depth: int = 2) -> list[str]:
    """模糊音变体；允许连改两处（pingying → pinying → pinyin）。"""
    seen = {text}
    frontier = [text]
    variants: list[str] = []
    for _ in range(depth):
        next_frontier: list[str] = []
        for base in frontier:
            for short, long_ in FUZZY_PAIRS:
                for source, target in ((short, long_), (long_, short)):
                    if source not in base:
                        continue
                    candidate = base.replace(source, target, 1)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    variants.append(candidate)
                    next_frontier.append(candidate)
                    if len(variants) >= limit:
                        return variants
        frontier = next_frontier
    return variants


class ImeState:
    def __init__(self, lexicon: Lexicon, page_size: int = PAGE_SIZE,
                 user_model: UserModel | ModeModels | None = None,
                 converter: Converter | None = None,
                 yue: YueConverter | None = None,
                 chao: ChaoshanConverter | None = None):
        self.lexicon = lexicon
        self.converter = converter
        self.yue = yue
        self.chao = chao
        self.lm = None             # 可选：字级语言模型（有 logp(text) 即可，可以是神经模型）
        self.lm_weight = 0.6
        self.mode = "simp"          # simp / trad / yue / chao / en
        self.shuangpin = False      # True = 小鹤双拼输入
        self.page_size = page_size
        # 词频按模式分家：粤语的选择不参与简体排序
        if isinstance(user_model, ModeModels):
            self._models: ModeModels | None = user_model
        elif user_model is not None:
            self._models = ModeModels({"simp": user_model})
        else:
            self._models = None
        self.buffer = ""
        self.context: str | None = None          # 上一次上屏的词，用来猜下一个词
        self.last_build_ms = 0.0
        self.last_commit_key = ""                # 最近一次上屏消耗掉的拼音
        self.last_commit_full = ""               # 候选原本对应的完整拼音键
        self.last_commit_typed = ""              # 你实际敲的（简拼/前缀补全时更短）
        self.plain_text = False                  # 当前串不像拼音（英文直出用）
        self._yue_cache: dict[str, str] = {}     # 首选候选的粤语神经结果（切模式不重算）
        self._candidates: list[str] = []         # 实际上屏用（按繁体开关转换）
        self._display: list[str] = []            # 候选窗显示用（简/繁并列）
        self._keys: list[str] = []               # 每个候选消耗的拼音前缀
        self._cache: dict[str, tuple[int, bool, list[str], list[str], list[str]]] = {}
        self._page = 0
        self._highlighted = 0
        self._segment = -1   # -1 = 全部音节都参与；>=1 表示只取前 n 个音节

    # ---------- 输入 ----------
    def insert(self, char: str) -> bool:
        if not char or len(self.buffer) >= MAX_BUFFER:
            return False
        self.buffer += char
        self.plain_text = False
        self._segment = -1
        self._rebuild()
        return True

    def backspace(self) -> bool:
        if not self.buffer:
            return False
        self.buffer = self.buffer[:-1]
        self._segment = -1
        self._rebuild()
        return True

    def clear(self) -> bool:
        if not self.buffer:
            return False
        self.buffer = ""
        self.plain_text = False
        self._candidates = []
        self._keys = []
        self._page = 0
        self._highlighted = 0
        self._segment = -1
        return True

    # ---------- 选择 ----------
    def commit_current(self) -> str | None:
        if not self.buffer:
            return None
        index = self._page * self.page_size + self._highlighted
        return self._commit_index(index)

    def commit_raw(self) -> str | None:
        if not self.buffer:
            return None
        text = self.buffer
        self.last_commit_key = ""
        self.clear()
        return text

    def select(self, digit: int) -> str | None:
        if not self.buffer or digit < 1:
            return None
        return self._commit_index(self._page * self.page_size + digit - 1)

    def select_page_index(self, index: int) -> str | None:
        if not self.buffer or index < 0:
            return None
        return self._commit_index(self._page * self.page_size + index)

    def _commit_index(self, index: int) -> str | None:
        if index < 0 or index >= len(self._candidates):
            return None
        word = self._candidates[index]
        full = self._keys[index] if index < len(self._keys) else self.buffer
        key = full
        # 前缀补全：候选的完整拼音比输入的还长（shif → 是否/shifou），
        # 学习要记在「你实际打的 shif」上，否则选多少次都记不到 shif 名下
        if key and len(key) > len(self.buffer):
            key = self.buffer
        self.last_commit_key = key
        self.last_commit_full = full
        self.last_commit_typed = self.buffer
        # 只吃掉了部分拼音时，剩下的继续组词（长拼音里选单字不用删字母）
        if key and len(key) < len(self.buffer) and self.buffer.startswith(key):
            self.buffer = self.buffer[len(key):]
            self._segment = -1
            self._rebuild()
        else:
            self.buffer = ""
            self._candidates = []
            self._keys = []
            self._page = 0
            self._highlighted = 0
            self._segment = -1
        return word

    def move_segment(self, delta: int) -> bool:
        """左右键：在「你 / nihao / nihaoma」之间切换要转换的拼音段。"""
        syllables = self.lexicon.split_syllables(self.buffer)
        if len(syllables) < 2:
            return False
        total = len(syllables)
        current = total if self._segment < 0 else min(self._segment, total)
        target = min(max(current + delta, 1), total)
        if target == current:
            return False
        self._segment = -1 if target == total else target
        self._rebuild()
        return True

    @property
    def user_model(self) -> UserModel | None:
        """当前输出模式对应的那份个人词频。"""
        if self._models is None:
            return None
        return self._models.for_mode(self.mode)

    def _lm_score(self, text: str, fallback: float) -> float:
        """展开候选的排序分：有语言模型就用它，没有就用词频。"""
        if self.lm is None:
            return fallback
        score = self.lm.logp(text)
        return score if score == score and score > -1e9 else fallback

    def _abbrev_segments(self, text: str):
        """把 qingkx / qkx 切成「已知音节 + 待展开首字母」的序列。

        返回 None 表示不像简拼：能整段切成音节（nihao），或首字母段少于 2 个
        （nih 这种正常打一半的交给前缀补全，不要在这里展开）。
        """
        segments: list[tuple[str, list[tuple[str, int]] | None]] = []
        syllables = self.lexicon.syllables()
        initials = 0
        index = 0
        while index < len(text):
            matched = ""
            for length in range(min(6, len(text) - index), 0, -1):
                piece = text[index:index + length]
                if piece in syllables:
                    matched = piece
                    break
            if matched:
                segments.append((matched, None))
                index += len(matched)
                continue
            options = self.lexicon.syllable_options(text[index], ABBREV_OPTIONS)
            if not options:
                return None
            segments.append((text[index], options))
            initials += 1
            index += 1
        # 只展开「首音节全拼 + 1 个首字母」这种两段式（qingkx / banbl）：
        # 三段以上（qkx）组合爆炸，交给静态简拼表 + 一次全拼学习即可
        return segments if initials == 2 else None

    def _expand_abbreviation(self, text: str) -> list[str]:
        """简拼 → 若干条候选完整拼音串（按音节词频剪枝）。

        `qingkx` 展开成 qing+kuang+xia、qing+ke+xi… 等 40 条以内，
        再交给整句切分去认「情况下」——这类短语不在词库里，
        是 情况 + 下 拼出来的，静态简拼表永远查不到。
        """
        segments = self._abbrev_segments(text)
        if not segments:
            return []
        combos: list[tuple[float, str]] = [(0.0, "")]
        for piece, options in segments:
            if options is None:
                combos = [(score, partial + piece) for score, partial in combos]
                continue
            expanded: list[tuple[float, str]] = []
            for score, partial in combos:
                for syllable, freq in options:
                    expanded.append((score + math.log1p(freq), partial + syllable))
            expanded.sort(key=lambda item: -item[0])
            combos = expanded[:ABBREV_COMBOS]
        return [full for _score, full in combos]

    def _abbrev_candidates(self, text: str) -> list[tuple[float, str]]:
        """qingkx → 情况下、banbl → 办不了。

        做法是「按词组合」而不是「展开整串再切分」：候选由 词+词 或 字+字+字 拼出来，
        天然过滤掉「请科学」这种不是词的东西，最后交给字级 LM 排序
        （LM 分得清 情况下 -12.7 与 请科学 -19.5）。

        每条路径只取词频前几名再送 LM——全量送进去会被「版本(1000万)」这类
        超高频词淹没，LM 根本没机会看到正确答案。
        """
        segments = self._abbrev_segments(text)
        if not segments:
            return []
        fixed = "".join(piece for piece, options in segments if options is None)
        tail = [piece for piece, options in segments if options is not None]
        options1 = [s for s, _ in self.lexicon.syllable_options(tail[0], 24)]
        options2 = [s for s, _ in self.lexicon.syllable_options(tail[1], 24)]

        def words_of(key: str, limit: int = 2) -> list[tuple[str, int]]:
            return self.lexicon.entries(key)[:limit]

        def rank(pairs: list[tuple[float, str]], limit: int = 8) -> list[tuple[float, str]]:
            best: dict[str, float] = {}
            for score, word in pairs:
                if len(word) >= 2 and score > best.get(word, -1e9):
                    best[word] = score
            return sorted(((s, w) for w, s in best.items()), reverse=True)[:limit]

        # ① 词 + 词：情况 + 下
        heads = [(w, f) for first in options1 for w, f in words_of(fixed + first)]
        tails = [(w, f) for second in options2 for w, f in words_of(second)]
        path1 = rank([(math.log1p(hf) + math.log1p(tf), h + t)
                      for h, hf in heads for t, tf in tails])

        # ② 词 + 词：办 + 不了
        path2 = rank([(math.log1p(hf) + math.log1p(tf), h + t)
                      for h, hf in words_of(fixed)
                      for first in options1[:8]
                      for second in options2[:8]
                      for t, tf in words_of(first + second)])

        # ③ 字 + 字 + 字：办 + 不 + 了
        path3 = rank([(math.log1p(hf) + math.log1p(mf) + math.log1p(lf), h + m + last)
                      for h, hf in words_of(fixed)
                      for first in options1[:5]
                      for m, mf in words_of(first)
                      for second in options2[:5]
                      for last, lf in words_of(second, 1)])

        scored = [(self._lm_score(word, score), word)
                  for score, word in path1 + path2 + path3]
        scored.sort(key=lambda item: -item[0])
        return scored

    def _abbrev_keys(self, key: str) -> list[str]:
        """一个完整拼音键的缩写形式：全首字母（qkx）+ 首音节全拼（qingkx）。"""
        syllables = self.lexicon.split_syllables(key)
        if len(syllables) < 2:
            return []
        initials = "".join(syllable[0] for syllable in syllables)
        head = syllables[0] + "".join(syllable[0] for syllable in syllables[1:])
        return [initials] if head == initials else [initials, head]

    def learn_keys(self, typed: str | None, consumed: str,
                   full: str | None = None) -> list[str]:
        """上屏后要记进词频的键（可能不止一个）。

        少记一个就会出现「明明选过却永远不命中」：

        1. 实际敲的键（jianp / shif）——前缀补全时它比完整键短；
        2. 完整拼音键（jianpin）——下次老实打全拼也要能命中；
        3. 它的简拼（jianpin → jp）——这条是「简拼命不中」的根因：
           你打全拼选过「简拼」，却从来没教过输入法 jp 也等于它，
           而简拼候选表按词频排、只给前几十个，冷门词根本翻不到，
           于是永远选不到、永远学不会。选中一次就把简拼一起教了。
        """
        keys: list[str] = []
        for candidate in (consumed, typed):
            if candidate and candidate not in keys:
                keys.append(candidate)
        for abbrev in self._abbrev_keys(full or consumed):
            if abbrev not in keys:
                keys.append(abbrev)
        return keys

    def base_form(self, text: str) -> str:
        """上屏学习一律记简体形。

        繁体/粤语只是显示层的转换：选了「聲音」要记成「声音」，
        否则下次简体模式会把它当成另一个词的命中。
        """
        return self.converter.to_simplified(text) if self.converter else text

    def active_text(self) -> str:
        syllables = self.lexicon.split_syllables(self.buffer)
        if not syllables or self._segment < 0:
            return self.buffer
        return "".join(syllables[: self._segment])

    def current_word(self) -> str | None:
        """当前高亮的候选——就是按空格会上屏的那个（已按模式转好简/繁/粤）。"""
        index = self._page * self.page_size + self._highlighted
        if 0 <= index < len(self._candidates):
            return self._candidates[index]
        return None

    # ---------- 浏览 ----------
    def move(self, delta: int) -> bool:
        total = len(self._candidates)
        if total == 0:
            return False
        index = self._page * self.page_size + self._highlighted + delta
        if index < 0 or index >= total:
            return False
        self._page, self._highlighted = divmod(index, self.page_size)
        return True

    def cycle_mode(self) -> str:
        """Tab：简体 → 繁体 → 粤语 → 英文 → 简体。

        英文是「固定的一档」，不是 Shift 随手按出来的：
        Shift 只负责强制回到中文（见 engine 的 Shift 处理）。
        """
        self.mode = {"simp": "trad", "trad": "yue", "yue": "chao",
                     "chao": "en", "en": "simp"}[self.mode]
        if self.mode == "en":
            self.buffer = ""        # 进英文档不留半截拼音
        self._rebuild()
        return self.mode

    def page_by(self, delta: int) -> bool:
        if not self._candidates:
            return False
        target = min(max(self._page + delta, 0), self._page_count() - 1)
        if target == self._page:
            return False
        self._page = target
        self._highlighted = 0
        return True

    # ---------- 视图 ----------
    def attach_lm(self, lm, weight: float = 0.6) -> None:
        """挂上上下文语言模型：整句候选会按「词库先验 + λ·上下文分」重排。

        lm 只需要实现 logp(text) -> float（对数概率）。神经 LM 用同样的接口直接替换。
        """
        self.lm = lm
        self.lm_weight = weight
        self._cache.clear()
        if self.buffer:                 # 候选是上次按键算的，挂上模型后要重算一遍
            self._rebuild()

    def rebuild_neural(self) -> bool:
        """把首选候选升级成神经粤语（引擎在按键空闲 200ms 后调用）。

        为什么不在按键路径上做：一次推理 8~45ms，每键都做就是「打字很卡」。
        先给规则版（0.02ms），停手了再换成神经版，用户看到的顺序不变、手感不卡。
        """
        if self.mode != "yue" or self.yue is None or not self.buffer:
            return False
        text = self.active_text()
        cached = self._cache.get(text)
        if cached and cached[0] == (self.user_model.version if self.user_model else 0):
            self._cache.pop(text, None)
        self._rebuild(neural=True)
        return True

    def view(self) -> View:
        if not self.buffer:
            return View(visible=False, preedit="")
        start = self._page * self.page_size
        return View(
            visible=True,
            preedit=self.buffer,
            cursor=len(self.active_text()),
            candidates=self._display[start : start + self.page_size],
            words=self._candidates[start : start + self.page_size],
            page=self._page,
            page_count=self._page_count(),
            highlighted=self._highlighted,
        )

    # ---------- 内部 ----------
    def _page_count(self) -> int:
        if not self._candidates:
            return 0
        return (len(self._candidates) + self.page_size - 1) // self.page_size

    def _plausible(self, key: str, word: str) -> bool:
        if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
            return True
        return len(word) in self.lexicon.syllable_counts(key)

    def _compose(self, text: str) -> list[str]:
        """按音节把单字拼起来，补出词库没有的常见组合（只+会→只会）。"""
        syllables = self.lexicon.split_syllables(text)
        if not syllables or not (2 <= len(syllables) <= 3):
            return []
        pools: list[list[tuple[str, int]]] = []
        for syllable in syllables:
            words = self.lexicon.entries(syllable)[:WORDS_PER_SYLLABLE]
            if not words:
                return []
            pools.append(list(words))
        combos: list[tuple[float, str]] = []
        for picked in product(*pools):
            word = "".join(item[0] for item in picked)
            score = sum(self.lexicon.score(item[1]) for item in picked)
            if self.user_model and len(picked) > 1:
                score += 3.0 * sum(
                    1 for left, right in zip(picked, picked[1:])
                    if right[0] in self.user_model.after(left[0])
                )
            combos.append((score, word))
        combos.sort(key=lambda item: -item[0])
        return [word for _, word in combos[:MAX_COMPOSED]]

    def _fallback(self, text: str) -> list[str]:
        """词库兜底：任何输入都不能是空白。"""
        out: list[str] = []
        for syllable in (self.lexicon.split_syllables(text) or [])[-2:]:
            out.extend(word for word, _ in self.lexicon.entries(syllable)[:3])
        for cut in range(len(text) - 1, 0, -1):
            head, tail = text[:cut], text[cut:]
            if not self.lexicon.is_syllable_prefix(tail):
                break
            if not self.lexicon.split_syllables(head):
                continue
            found = [word for word, _ in self.lexicon.entries(head)[:4]]
            found += [word for word, _ in self.lexicon.prefix(head)[:4]]
            if found:
                out.extend(found)
                break
        seen: set[str] = set()
        result: list[str] = []
        for word in out:
            if word not in seen:
                seen.add(word)
                result.append(word)
        return result[:9]

    def _rebuild(self, neural: bool = False) -> None:
        """重建候选。neural=True 时首选候选走神经粤语（慢，留给空闲时做）。"""
        # 英文档：不产生任何候选，按键原样交给应用
        if self.mode == "en":
            self._candidates, self._display, self._keys = [], [], []
            return
        neural_pass = neural
        started = time.perf_counter()
        text = self.active_text()
        if self.shuangpin and text:
            from . import shuangpin

            text = shuangpin.to_full(text)
        if not self.buffer:
            self._candidates, self._keys = [], []
            return

        version = self.user_model.version if self.user_model else 0
        cached = self._cache.get(text) if self._segment < 0 else None
        if cached and cached[0] == version and cached[1] == self.mode:
            _, _, self._candidates, self._display, self._keys = cached
            self.last_build_ms = (time.perf_counter() - started) * 1000
            return

        # 快路径：不像拼音的串（hello / word / nice 这类英文）只要没有可补的前缀，
        # 就直接把原文当唯一候选——跳过整句切分、语言模型、粤语转换。
        # 英文打字本来就不该走中文那套（实测每键 20~30ms，卡得明显）。
        if not self.lexicon.split_syllables(text):
            learned = self.user_model.preferred(text) if self.user_model else []
            if (not learned and self._abbrev_segments(text) is None
                    and not self.lexicon.prefix_entries(text)[:1]
                    and not self.lexicon.initials_entries(text)):
                self.plain_text = True
                self._candidates = [text]
                self._display = [text]
                self._keys = [text]
                self._page = 0
                self._highlighted = 0
                self.last_build_ms = (time.perf_counter() - started) * 1000
                return

        # 词库里偶有「w → w」这种字母当词的垃圾条目，会顶掉真正的候选
        exact = [(w, f) for w, f in self.lexicon.entries(text)
                 if not (len(w) <= 2 and w.isascii())]
        keyed = self.lexicon.prefix_entries(text)
        paths = best_paths_scored(self.lexicon, text, k=SENTENCE_CANDIDATES)
        if self.lm is not None and len(paths) > 1:
            # 词库先验 + 上下文分插值，按字数归一。λ=0.6 是基准全量扫描（0.3~1.0）的
            # 甜点：重排 83.2%→85.5%（救回 17→23、弄坏 2 不变）；λ=1.0 开始弄坏增多。
            # 领域语料仍按 nn-chinese.md §八 的结论单独调 λ（attach_lm 传参覆盖）：
            # 不归一的话对数概率求和会让最短的候选永远排第一（与质量无关）——
            # 这是 chinese-ime-lm 的 docs/measurements.md 点名的两个测量陷阱之一。
            weight = self.lm_weight

            def _rank(item):
                score, seg = item
                text_len = max(len("".join(seg)), 1)
                return -(score + weight * self.lm.logp("".join(seg))) / text_len

            paths = sorted(paths, key=_rank)
        sentences = ["".join(p) for _score, p in paths if len(p) > 1]
        composed = self._compose(text)

        close = [(k, w, f) for k, w, f in keyed if len(k) <= len(text) + 3]
        far = [(k, w, f) for k, w, f in keyed if len(k) > len(text) + 3]

        items: list[tuple[str, str, int]] = []   # (词, 消耗的拼音, 词频)
        if exact:
            items += [(w, text, f) for w, f in exact]
            items += [(s, text, 0) for s in sentences]
            items += [(w, k, f) for k, w, f in close]
            items += [(w, k, f) for k, w, f in far]
            items += [(w, text, 0) for w in composed]
        elif close:
            items += [(w, k, f) for k, w, f in close]
            items += [(s, text, 0) for s in sentences]
            items += [(w, text, 0) for w in composed]
            items += [(w, k, f) for k, w, f in far]
        else:
            items += [(s, text, 0) for s in sentences]
            items += [(w, text, 0) for w in composed]
            items += [(w, k, f) for k, w, f in far]

        # 简拼：qingkx → 情况下、banbl → 办不了。
        # 这些短语不在词库里（是 情况 + 下 拼出来的），静态简拼表查不到，只能现场展开，
        # 再用语言模型挑：展开出来的同音组合很多，词频分不清「请科学」和「情况下」，
        # 字级 LM 分得清（-12.7 vs -19.5）。
        # 静态简拼表已经能给出候选（rg → 如果）就别展开，否则会把老行为顶掉
        if not exact and not close and not self.lexicon.initials_entries(text):
            scored = self._abbrev_candidates(text)
            seen_abbrev: set[str] = set()
            scored.sort(key=lambda item: -item[0])
            rank = 0
            for _score, word in scored:
                if word in seen_abbrev:
                    continue
                seen_abbrev.add(word)
                items.append((word, text, 100000 - rank * 100))
                rank += 1
                if rank >= 9:
                    break

        # 表情：wx → 😊（整串命中才出，避免干扰正常输入）
        emoji = self.lexicon.emoji_for(text)

        # 简拼：rg → 如果（每个音节取首字母，搜狗/Rime 同一原理）
        # 去重交给下面的统一去重环节
        if text.isalpha() and 2 <= len(text) <= 6:
            items.extend((word, text, freq) for word, freq in self.lexicon.initials_entries(text))
        # 混拼：首音节首字母 + 其余全拼（wmen→我们、rguo→如果）
        if text.isalpha() and 3 <= len(text) <= 10:
            items.extend((word, text, freq) for word, freq in self.lexicon.mixed_entries(text))

        # 单字候选：长拼音里也能直接选字，选完剩余拼音继续组词
        # 末音节必须有：diyiye 想选「页」时，只给前两个音节就永远选不到
        syllables = self.lexicon.split_syllables(text) or []
        wanted = syllables[:2] + ([syllables[-1]] if syllables and syllables[-1] not in syllables[:2] else [])
        for syllable in wanted:
            for word, freq in self.lexicon.entries(syllable)[:5]:
                if len(word) == 1:
                    items.append((word, syllable, freq))

        # 去重（同一个词只留第一次出现的）
        seen: set[str] = set()
        unique: list[tuple[str, str, int]] = []
        for word, key, freq in items:
            if word in seen:
                continue
            seen.add(word)
            unique.append((word, key, freq))

        # 排序完全按「你选过的」：选中一次 +1，次数多的在前。
        # 分数只在同等条件下当补充依据（整串匹配 > 上下文 > 词频），
        # 任何规则都不许压过你的选择。
        syllables = self.lexicon.split_syllables(text) or []
        context_words: set[str] = set()
        if self.user_model and self.context:
            context_words = set(self.user_model.after(self.context))

        def rank_key(item: tuple[str, str, int]) -> tuple:
            word, _key, freq = item
            picks = self.user_model.picks(text, word) if self.user_model else 0
            full = 1 if (syllables and len(word) == len(syllables)) else 0
            near = 1 if word in context_words else 0
            return (-picks, -full, -near, -freq)

        # 学过的词即使候选里没有，也要补进来（照样按选中次数排）
        if self.user_model:
            for word in self.user_model.preferred(text):
                if word in seen or (len(word) <= 2 and word.isascii()):
                    continue
                seen.add(word)
                unique.append((word, text, 0))

        # 表情也是普通候选：没选过就是 0 次，排在选过的和整串词之后
        emoji_word = self.lexicon.emoji_for(text)
        if emoji_word and emoji_word not in seen:
            seen.add(emoji_word)
            unique.append((emoji_word, text, 0))

        unique.sort(key=rank_key)
        picked = {word for word, _k, _f in unique
                  if self.user_model and self.user_model.picks(text, word) > 0}

        # 模糊音：只在首选很弱时启用，且只收比首选更高频的
        # 但是——用户手动选过的词绝不能被模糊音挤下去
        top_freq = unique[0][2] if unique else 0
        weak = (not unique) or (len(text) <= 8 and top_freq < 1000)
        if len(text) >= 2 and not exact and weak and picked.isdisjoint([unique[0][0]] if unique else []):
            strong: list[tuple[str, str, int]] = []
            for variant in fuzzy_variants(text, limit=8) + transposition_variants(text):
                for word, freq in self.lexicon.entries(variant)[:5]:
                    if word not in seen and freq > top_freq:
                        strong.append((word, text, freq))
                if len(strong) >= 12:
                    break
            if strong:
                strong.sort(key=lambda item: -item[2])
                head: list[tuple[str, str, int]] = []
                for word, key, freq in strong:
                    if word in seen or word in picked:
                        continue
                    seen.add(word)
                    head.append((word, key, freq))
                # 插在"用户选过的词"之后，绝不覆盖它们
                insert_at = 0
                while insert_at < len(unique) and unique[insert_at][0] in picked:
                    insert_at += 1
                unique[insert_at:insert_at] = head

        # 只要有中文候选，就不要把字母本身列在候选里（w / abc 这类只配在纯英文输入时出现）
        if any(any("\u4e00" <= ch <= "\u9fff" for ch in w) for w, _, _ in unique):
            # 有中文候选时，只丢掉「字母本身」这类候选；表情/符号要留着
            unique = [item for item in unique
                      if not (item[0].isascii() and item[0].isalnum() and len(item[0]) <= 3)]
        if not unique:
            unique = [(word, text, 0) for word in self._fallback(text)]
        if not unique:
            unique = [(text, text, 0)]

        converter = self.converter
        words: list[str] = []
        displays: list[str] = []
        keys: list[str] = []
        seen_out: set[str] = set()   # 同一个词的不同字形（聲音/声音）只留第一条
        for rank, (word, key, _freq) in enumerate(unique):
            simp = word
            trad = converter.to_traditional(simp) if converter else simp
            # 神经模型只在粤语模式、且只对首选候选推理：
            # convert() 在候选重建路径上，2~4 字词单次 7~10ms，9 个候选全算就是 60ms+/键。
            # 非首选候选走规则（0.02ms）够用，用户真正要看的粤语写法就是首选那条。
            neural = neural_pass and rank == 0
            if self.yue is None:
                yue = trad
            elif neural:
                cached = self._yue_cache.get(simp)
                if cached is None:
                    cached = self.yue.convert(simp, neural=True)
                    if len(self._yue_cache) > 512:
                        self._yue_cache.clear()
                    self._yue_cache[simp] = cached
                yue = cached
            else:
                yue = self.yue.convert(simp, neural=False)
            if converter:
                yue = converter.to_traditional(yue)   # 粤语书面一律用繁体字
            # 潮州话（汕头音）：词级改写（吃饭→食饭、我们→阮），语序仍是普通话的
            chao_text = self.chao.rewrite(simp) if self.chao is not None else ""
            forms = {"simp": simp, "trad": trad, "yue": yue, "chao": chao_text or simp}
            output = forms.get(self.mode, simp)
            if output in seen_out:
                continue
            seen_out.add(output)
            words.append(output)
            keys.append(key)
            # 固定四列：普通话 / 繁体 / 粤语 / 粤拼，方便对照学习
            if not any("\u4e00" <= ch <= "\u9fff" for ch in simp):
                displays.append(simp)
                continue
            if self.mode == "chao" and self.chao is not None:
                # 潮汕档：只留三列——原文 / 潮汕文字 / 潮拼，方便对照
                pengim = self.chao.pengim(chao_text)
                columns = [simp, chao_text or simp]
                if pengim:
                    columns.append(pengim)
                displays.append("  ".join(columns))
                continue
            columns = [simp, trad if trad != simp else "—", yue if yue != simp else "—"]
            if self.yue is not None:
                # 粤拼词表是繁体字，先把粤语写法转繁体再查（呢個/聽日/邊度），查不到再用原形
                yue_trad = converter.to_traditional(yue) if converter else yue
                reading = self.yue.jyutping(yue_trad) or self.yue.jyutping(yue)
                if reading:
                    columns.append(reading)
            if chao_text:
                columns.append(chao_text if chao_text != simp else "—")   # 潮汕文字放最后一列
            displays.append("  ".join(columns))
        self._candidates = words
        self._display = displays
        self._keys = keys
        self.last_build_ms = (time.perf_counter() - started) * 1000
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[text] = (version, self.mode, self._candidates, self._display, self._keys)
        if self.user_model:
            self.user_model.observe(text, self._candidates[: self.page_size])
