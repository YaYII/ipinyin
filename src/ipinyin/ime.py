"""输入法状态机（纯逻辑，不依赖 IBus，方便单测）。

职责：维护拼音缓冲、生成候选、处理翻页/选词，产出给 UI 用的视图。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

from .lexicon import Lexicon
from .segmenter import best_paths_scored
from .t2s import Converter
from .yue import YueConverter
from .user_model import UserModel

PAGE_SIZE = 9
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
                 user_model: UserModel | None = None, converter: Converter | None = None,
                 yue: YueConverter | None = None):
        self.lexicon = lexicon
        self.converter = converter
        self.yue = yue
        self.lm = None             # 可选：字级语言模型（有 logp(text) 即可，可以是神经模型）
        self.lm_weight = 0.6
        self.mode = "simp"          # simp / trad / yue
        self.page_size = page_size
        self.user_model = user_model
        self.buffer = ""
        self.context: str | None = None          # 上一次上屏的词，用来猜下一个词
        self.last_build_ms = 0.0
        self.last_commit_key = ""                # 最近一次上屏消耗掉的拼音
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
        key = self._keys[index] if index < len(self._keys) else self.buffer
        self.last_commit_key = key
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

    def active_text(self) -> str:
        syllables = self.lexicon.split_syllables(self.buffer)
        if not syllables or self._segment < 0:
            return self.buffer
        return "".join(syllables[: self._segment])

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
        """Tab：简体 → 繁体 → 粤语 → 简体"""
        self.mode = {"simp": "trad", "trad": "yue", "yue": "simp"}[self.mode]
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

    def _rebuild(self) -> None:
        started = time.perf_counter()
        text = self.active_text()
        if not self.buffer:
            self._candidates, self._keys = [], []
            return

        version = self.user_model.version if self.user_model else 0
        cached = self._cache.get(text) if self._segment < 0 else None
        if cached and cached[0] == version and cached[1] == self.mode:
            _, _, self._candidates, self._display, self._keys = cached
            self.last_build_ms = (time.perf_counter() - started) * 1000
            return

        exact = self.lexicon.entries(text)
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

        # 简拼：rg → 如果（每个音节取首字母，搜狗/Rime 同一原理）
        # 去重交给下面的统一去重环节
        if text.isalpha() and 2 <= len(text) <= 6:
            items.extend((word, text, freq) for word, freq in self.lexicon.initials_entries(text))

        # 单字候选：长拼音里也能直接选字，选完剩余拼音继续组词
        for syllable in (self.lexicon.split_syllables(text) or [])[:2]:
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

        # 上下文预测 + 个人词频：不光调顺序，学过的词没在候选里也要补进来
        # （饿了吗 不在词库、组合也没拼出来，只靠"排序"永远看不到它）
        priority: dict[str, int] = {}
        if self.user_model and self.context:
            for word in self.user_model.after(self.context):
                if word in seen:
                    priority.setdefault(word, len(priority))
        if self.user_model:
            for rank, word in enumerate(self.user_model.preferred(text)):
                if word not in seen:
                    seen.add(word)
                    unique.insert(0, (word, text, 0))     # 学过的词补进候选
                priority.setdefault(word, rank)
        if priority:
            unique.sort(key=lambda item: priority.get(item[0], len(priority)))

        # 模糊音：只在首选很弱时启用，且只收比首选更高频的
        # 但是——用户手动选过的词绝不能被模糊音挤下去
        picked = set(priority)
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

        if not unique:
            unique = [(word, text, 0) for word in self._fallback(text)]
        if not unique:
            unique = [(text, text, 0)]

        converter = self.converter
        words: list[str] = []
        displays: list[str] = []
        for rank, (word, key, _freq) in enumerate(unique):
            simp = word
            trad = converter.to_traditional(simp) if converter else simp
            # 神经模型只在粤语模式、且只对首选候选推理：
            # convert() 在候选重建路径上，2~4 字词单次 7~10ms，9 个候选全算就是 60ms+/键。
            # 非首选候选走规则（0.02ms）够用，用户真正要看的粤语写法就是首选那条。
            neural = rank == 0
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
            forms = {"simp": simp, "trad": trad, "yue": yue}
            output = forms.get(self.mode, simp)
            words.append(output)
            # 固定四列：普通话 / 繁体 / 粤语 / 粤拼，方便对照学习
            if not any("\u4e00" <= ch <= "\u9fff" for ch in simp):
                displays.append(simp)
                continue
            columns = [simp, trad if trad != simp else "—", yue if yue != simp else "—"]
            if self.yue is not None:
                # 粤拼词表是繁体字，先把粤语写法转繁体再查（呢個/聽日/邊度），查不到再用原形
                yue_trad = converter.to_traditional(yue) if converter else yue
                reading = self.yue.jyutping(yue_trad) or self.yue.jyutping(yue)
                if reading:
                    columns.append(reading)
            displays.append("  ".join(columns))
        self._candidates = words
        self._display = displays
        self._keys = [key for _, key, _ in unique]
        self.last_build_ms = (time.perf_counter() - started) * 1000
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[text] = (version, self.mode, self._candidates, self._display, self._keys)
        if self.user_model:
            self.user_model.observe(text, self._candidates[: self.page_size])
