"""IBus 引擎适配层：把按键事件翻译成状态机动作，再画到 IBus 的 preedit / 候选表上。

这一层只做「翻译与显示」，所有输入逻辑都在 ime.py（纯 Python，可单测）。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import gi

gi.require_version("IBus", "1.0")
from gi.repository import Gio, GLib, GObject, IBus  # noqa: E402

from . import config, memory  # noqa: E402
from .ime import ImeState  # noqa: E402
from .keys import Action, translate  # noqa: E402
from .lexicon import Lexicon  # noqa: E402
from .t2s import Converter  # noqa: E402
from .yue import YueConverter  # noqa: E402
from .user_model import ModeModels, UserModel  # noqa: E402

LOG = logging.getLogger("ipinyin")

_LEXICON: Lexicon | None = None
# 打字很快时，候选窗没必要每键都重画：把 UI 刷新合并到 15ms 一次（约 60fps）
SYNC_INTERVAL = 0.015
_MODELS: ModeModels | None = None
_CONVERTER: Converter | None = None
_YUE: YueConverter | None = None
_LM = None            # 字级语言模型；False 表示明确没有/已关闭
# 词库加载完成前，按键一律放行（当英文打），绝不阻塞 IBus 主循环
_LEXICON_READY = threading.Event()


def get_lexicon() -> Lexicon:
    global _LEXICON
    if _LEXICON is None:
        path = config.lexicon_path()
        LOG.info("加载词库 %s", path)
        _LEXICON = Lexicon.load(path)
        _LEXICON.load_initials(config.initials_path())
        _LEXICON.warm_options()
        _LEXICON.load_emoji(config.emoji_path())
        LOG.info("词库就绪：%d 个拼音键", len(_LEXICON))
    return _LEXICON


def get_yue() -> YueConverter:
    global _YUE
    if _YUE is None:
        _YUE = YueConverter.load_dict(config.yue_dict_path(), YueConverter.load(config.yue_path())._map)
        if os.environ.get("IPINYIN_YUE_NN", "1") != "0":
            from .yue_nn import NeuralYue

            # 神经模型：优先用户目录（可单独换新模型），deb 装机退回系统目录
            for nn_dir in (os.path.join(config.data_home(), "data"), config.SYSTEM_DATA):
                _YUE.nn = NeuralYue.load(nn_dir)
                if _YUE.nn is not None:
                    break
            if _YUE.nn is not None:
                LOG.info("粤语神经模型已加载（%d 字表）", len(_YUE.nn))
            elif any(os.path.exists(os.path.join(d, "yue_nn.onnx"))
                     for d in (os.path.join(config.data_home(), "data"), config.SYSTEM_DATA)):
                # 模型在但运行时缺：告诉用户怎么启用，而不是悄悄退回规则版
                LOG.warning("找到 yue_nn.onnx 但缺 onnxruntime —— pip3 install --user onnxruntime "
                            "后重启引擎即可启用神经粤语转换")
        from .yue_guard import YueGuard

        _YUE.guard = YueGuard.load(config.yue_guard_path())
        if _YUE.guard is not None:
            LOG.info("粤语解码守卫已加载")
        LOG.info("粤语对照 %d 条", len(_YUE))
    return _YUE


def get_lm():
    """字级语言模型（可选）：给整句候选做上下文重排。

    放在 ~/.local/share/ipinyin/data/lm_char.marshal，没有就退回纯词库排序。
    设 IPINYIN_LM=0 可强制关闭。
    """
    global _LM
    if _LM is None:
        if os.environ.get("IPINYIN_LM", "1") == "0":
            _LM = False
            return None
        from .lm import CharLM

        path = os.path.join(config.data_home(), "data", "lm_char.marshal")
        _LM = CharLM.load(path) or False
        if _LM:
            LOG.info("语言模型已加载（%d 字表，%s）", len(_LM), path)
        else:
            LOG.info("没找到语言模型 %s，整句候选按词库排序", path)
    return _LM or None


def _read_input_mode() -> bool:
    """输入方案：full（全拼，默认）/ flypy（小鹤双拼）。"""
    try:
        with open(config.input_mode_path(), encoding="utf-8") as fh:
            return fh.read().strip() == "flypy"
    except OSError:
        return False


def _read_mode() -> str:
    """记住上次的简/繁选择。"""
    try:
        with open(config.mode_path(), encoding="utf-8") as fh:
            mode = fh.read().strip()
        # 历史数据里可能有 en：那是临时档，启动时一律回中文
        return mode if mode in ("simp", "trad", "yue") else "simp"
    except OSError:
        return "simp"


def _write_mode(mode: str) -> None:
    try:
        path = config.mode_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(mode)
    except OSError:
        pass


def _valid_pair(key: str, word: str) -> bool:
    """拼音键能切成几个音节，就该对应几个汉字；对不上的是脏数据（如 zenm→怎唔）。

    但简拼键（w / wm / rg）音节数少于字数，属于正常输入，必须放行——
    否则「w → 我们」永远学不进来。
    """
    if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
        return True
    if key.isascii() and key.isalpha():
        return True
    return len(word) in get_lexicon().syllable_counts(key)


def _tts_auto_path() -> str:
    return os.path.join(config.data_home(), "tts_auto")


def _read_tts_auto() -> bool:
    try:
        with open(_tts_auto_path(), encoding="utf-8") as handle:
            return handle.read().strip() == "1"
    except OSError:
        return False


def _write_tts_auto(on: bool) -> None:
    try:
        os.makedirs(os.path.dirname(_tts_auto_path()), exist_ok=True)
        with open(_tts_auto_path(), "w", encoding="utf-8") as handle:
            handle.write("1" if on else "0")
    except OSError:
        pass


def get_models() -> ModeModels:
    """三份个人词频（简/繁/粤各一份），互不干扰。"""
    global _MODELS
    if _MODELS is None:
        _MODELS = ModeModels({
            mode: UserModel(
                config.user_model_path(mode),
                config.user_seed_path(),
                config.user_bigram_path(mode),
                config.user_bigram_seed_path(),
                validator=_valid_pair,
            )
            for mode in ModeModels.MODES
        })
        LOG.info("个人词频 %s 条（简 %d / 繁 %d / 粤 %d）", len(_MODELS),
                 len(_MODELS.for_mode("simp")), len(_MODELS.for_mode("trad")),
                 len(_MODELS.for_mode("yue")))
    return _MODELS


def get_user_model() -> UserModel:
    """兼容旧调用：返回简体那份。新代码请用 get_models() / ime.user_model。"""
    return get_models().for_mode("simp")


def get_converter() -> Converter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = Converter.load(config.s2t_path())
        LOG.info("简繁映射 %d 字", len(_CONVERTER))
    return _CONVERTER


_ENGINE_CHECK_AT = 0.0


def _ensure_global_engine(reason: str) -> None:
    """全局引擎为空时自荐（后台线程里做）。

    必须离开主线程：在主线程同步调 set_global_engine，ibus-daemon 收到后会
    反过来让本引擎 Enable，而主线程正卡在这个同步调用上——死锁，
    最后 IBus 超时返回「操作被取消」，输入法彻底起不来（真实踩过）。
    """
    threading.Thread(target=_ensure_global_engine_worker, args=(reason,),
                     name="engine-ensure", daemon=True).start()


def _ensure_global_engine_worker(reason: str) -> None:
    global _ENGINE_CHECK_AT
    _ENGINE_CHECK_AT = time.monotonic()
    try:
        bus = IBus.Bus()
        if not bus.is_connected():
            return
        current = bus.get_global_engine()
        name = current.get_name() if current is not None else ""
        if name:
            return
        LOG.info("全局引擎为空（%s），自荐回 ipinyin", reason)
        if not bus.set_global_engine("ipinyin"):
            LOG.warning("自荐被拒绝（set_global_engine 返回 False）")
    except GLib.Error as exc:
        LOG.warning("自动设置全局引擎失败：%s", exc.message)


def setup_logging() -> None:
    if LOG.handlers:
        return
    level = logging.DEBUG if os.environ.get("IPINYIN_DEBUG") else logging.INFO
    LOG.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        path = config.log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path, maxBytes=512 * 1024, backupCount=2, encoding="utf-8"
        )
    except OSError:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    LOG.addHandler(handler)


class PinyinEngine(IBus.Engine):
    """一个最小的拼音引擎：整串匹配 + 整句切分 + 前缀补全。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ime: ImeState | None = None
        self._last_commit: str | None = None
        self._last_base: str | None = None   # 上屏词的简体形（学习/上下文用）
        self._last_sync = 0.0
        self._sync_source = 0
        self._last_commit_info: tuple[str, str, float] | None = None
        # 上屏即读：F9 切换，持久化到 ~/.local/share/ipinyin/tts_auto
        self._tts_auto = os.environ.get("IPINYIN_TTS_AUTO") == "1" or _read_tts_auto()
        self._hint_source = 0   # 切换后短暂显示状态用的 GLib 定时器
        self._neural_source = 0  # 粤语档：空闲后升级神经写法的定时器
        self._last_hint: tuple[str, bool] = ("", False)
        # 记忆体：上屏采集 + 空闲 60 秒后台整理
        self._last_key_at = time.monotonic()
        self._last_commit_at = 0.0
        self._last_focus_at = 0.0
        # AI 感知状态：读 screencast 心跳，3 秒缓存（别每次按键都读文件）
        self._ai_on = False
        self._ai_checked = 0.0
        self._ai_want: bool | None = None   # F10 的目标状态（None = 跟随心跳）
        self._boost_mtime = 0.0             # AI 提权队列的 mtime
        # 一次输入的「手感」：敲了几个键、退了几次、翻了几页、用了多久
        self._comp_start = 0.0
        self._comp_keys = 0
        self._comp_backs = 0
        self._comp_flips = 0
        GLib.timeout_add_seconds(10, self._memory_tick)
        self._last_cn_mode = "simp"  # 上一次的中文档（Shift 强制回来用）
        # 系统按键连发（auto-repeat）防护：同键 50ms 内反复到达，只认前两次
        self._repeat_key = -1
        self._repeat_count = 0
        self._repeat_at = 0.0
        self._shift_down = False
        self._shift_clean = True

    # ---------- 状态 ----------
    @property
    def ready(self) -> bool:
        return _LEXICON_READY.is_set()

    @property
    def ime(self) -> ImeState:
        if self._ime is None:
            self._ime = ImeState(get_lexicon(), user_model=get_models(),
                                 converter=get_converter(), yue=get_yue())
            lm = get_lm()
            if lm is not None:
                self._ime.attach_lm(lm)
            self._ime.mode = _read_mode()
            self._last_cn_mode = self._ime.mode if self._ime.mode != "en" else "simp"
            self._ime.shuangpin = _read_input_mode()
        return self._ime

    # ---------- IBus 虚函数 ----------
    def do_process_key_event(self, keyval, keycode, state):
        try:
            started = time.perf_counter()
            self._last_key_at = time.monotonic()
            handled = self._handle(keyval, state)
            cost = (time.perf_counter() - started) * 1000
            if cost > 20:  # 超过 20ms 就是肉眼可感的卡顿，单独记一条
                LOG.warning("按键处理慢 总 %.1f ms（算候选 %.1f ms）keyval=%#x",
                            cost, self.ime.last_build_ms, keyval)
            return handled
        except Exception:  # 引擎绝不能因为一次异常把输入法整挂
            LOG.exception("处理按键失败 keyval=%#x", keyval)
            return False

    def do_focus_out(self):
        # 切窗口时不丢正在输入的拼音：buffer 留着，回来还能接着选词。
        # （以前这里直接清空，用户体感就是"打了一半的字突然没了"）
        if self._ime is not None and self._ime.buffer:
            LOG.info("focus_out 保留未上屏拼音 %r", self._ime.buffer)
            self.hide_preedit_text()
            self.hide_lookup_table()
        get_user_model().flush_if_dirty()

    def do_set_cursor_location(self, x, y, w, h):
        """输入框光标的屏幕坐标：屏幕感知靠它判断「你在哪块屏幕打字」。"""
        try:
            from . import screencast

            screencast.write_cursor(x, y, w, h)
        except Exception:
            pass

    def do_focus_in(self):
        # 引擎位空了就自荐（节流 15s）：升级/重启后 ibus-daemon 会把当前引擎清掉，
        # 表现就是「输入法不见了，打出来全是拼音」
        if time.monotonic() - _ENGINE_CHECK_AT > 15:
            _ensure_global_engine("焦点切换")
        # 焦点切换也是一种触角：切得越频繁，说明越在多个任务间跳
        gap = int((time.monotonic() - self._last_focus_at) * 1000) if self._last_focus_at else 0
        self._last_focus_at = time.monotonic()
        memory.record(None, "", self._ime.mode if self._ime is not None else "simp",
                      kind="focus", gap_ms=gap)
        self._sync()

    def do_reset(self):
        self._reset()

    def do_disable(self):
        self._reset()
        get_user_model().flush_if_dirty()

    def do_candidate_clicked(self, index, button, state):
        try:
            key = self.ime.buffer
            self._finish(self.ime.select_page_index(int(index)), key, weight=3, rank=int(index) + 1)
        except Exception:
            LOG.exception("候选词点击失败 index=%s", index)
        return True

    # ---------- 内部 ----------
    def _reset(self) -> None:
        try:
            if self._ime is not None:
                self._ime.clear()
        finally:
            self._sync_now()

    def _handle(self, keyval: int, state: int) -> bool:
        if not self.ready:
            return False  # 词库还在后台加载：放行，不卡输入

        # 轻按一下 Shift（中间没按别的键）：中/英切换。英文模式下所有键原样放行。
        if keyval in (0xFFE1, 0xFFE2):
            if state & (1 << 30):
                if self._shift_down and self._shift_clean:
                    # Shift = 强制回到中文（不是切英文）。英文只能靠 Tab 转到「英文」档，
                    # 这样 Shift 永远不会把你带进英文，按下去一定是拼音。
                    if self._ime is not None:
                        if self._ime.buffer:
                            self._finish(self._ime.commit_current(), self._ime.buffer)
                        if self._ime.mode == "en":
                            self._ime.mode = self._last_cn_mode
                            _write_mode(self._ime.mode)
                            self._ime._rebuild()
                        LOG.info("Shift → 强制中文（%s）", self._ime.mode)
                    self._push_mode_hint(force=True)
                self._shift_down = False
            else:
                self._shift_down = True
                self._shift_clean = True
            return False
        if self._shift_down:
            self._shift_clean = False
        action, value = translate(keyval, state)
        LOG.debug("key %#x state=%#x -> %s %r buffer=%r", keyval, state, action.name, value, self.ime.buffer)
        if action is Action.IGNORE:
            return False

        # 切档键必须在任何状态下都生效：英文档里、空输入时都要能 Tab 回中文。
        # 以前这段写在 elif 链里（在 `if not composing: return False` 之后），
        # 结果英文档 buffer 永远为空 → Tab 被吞 → 再也回不来（真实 bug）。
        if action is Action.TOGGLE_TRAD:
            mode = self.ime.cycle_mode()
            if mode != "en":
                _write_mode(mode)      # 英文档不落盘：重启必须回中文
                self._last_cn_mode = mode
            LOG.info("切换到%s", {"simp": "简体", "trad": "繁体", "yue": "粤语", "en": "英文"}[mode])
            self._sync_now()
            self._flash_mode_hint()
            return True

        # 英文档：除了上面的全局键，其余按键原样交给应用
        if self._ime is not None and self._ime.mode == "en":
            return False

        composing = bool(self.ime.buffer)

        # 朗读不依赖是否有候选：没在打拼音时念刚上屏的那句
        if action is Action.SPEAK:
            return self._speak()

        # F10：AI 屏幕感知 开/关（随时可按）
        if action is Action.TOGGLE_AI:
            self._toggle_ai()
            return True

        # F9：上屏即读 开/关（随时可按，不用在输入中）
        if action is Action.TOGGLE_TTS:
            self._tts_auto = not self._tts_auto
            _write_tts_auto(self._tts_auto)
            LOG.info("上屏即读 %s", "开启" if self._tts_auto else "关闭")
            self._flash_mode_hint()
            return True

        if action is Action.LETTER:
            now = time.monotonic()
            if keyval == self._repeat_key and (now - self._repeat_at) < 0.05:
                self._repeat_count += 1
                self._repeat_at = now
                if self._repeat_count >= 1:  # 只保留 2 个，正常双字母够用
                    LOG.info("忽略系统按键连发 key=%r x%d", value, self._repeat_count + 1)
                    return True
            else:
                self._repeat_key = keyval
                self._repeat_count = 0
                self._repeat_at = now
            if value == "'" and not composing:
                return False
            if not composing:
                self._comp_start = time.monotonic()
            self._comp_keys += 1
            self.ime.insert(value)
            self._push_preedit()   # 先把字画出来，候选慢一点也不会觉得"卡"
            self._sync()
            self._schedule_neural()
            return True

        if not composing:
            return False

        # 上屏前先把挂起的神经升级补上：200ms 内就选词的话，
        # 否则会把规则版的粤语写法定稿（质量差一截）
        if action in (Action.COMMIT, Action.DIGIT) and self._neural_source:
            GLib.source_remove(self._neural_source)
            self._neural_source = 0
            if self._ime is not None and self._ime.rebuild_neural():
                self._sync_now()

        key = self.ime.buffer
        if action is Action.DIGIT:
            self._finish(self.ime.select(int(value)), key, weight=3, rank=int(value))
        elif action is Action.BACKSPACE:
            self._comp_backs += 1
            self.ime.backspace()
            self._sync()
        elif action is Action.COMMIT:
            text = self.ime.commit_current()
            # 中英混输：原样上屏的英文后面补一个空格，避免「你好world」黏在一起
            if text and text.isascii() and text.isalpha():
                text += " "
            self._finish(text, key)
        elif action is Action.COMMIT_RAW:
            self._finish(self.ime.commit_raw(), key)
        elif action is Action.CLEAR:
            self._record_abandon()
            self._reset()
        elif action is Action.PAGE_PREV:
            self._comp_flips += 1
            self.ime.page_by(-1)
            self._sync()
        elif action is Action.PAGE_NEXT:
            self._comp_flips += 1
            self.ime.page_by(1)
            self._sync()
        elif action is Action.SEG_PREV:
            if self.ime.move_segment(-1):
                self._sync_now()
        elif action is Action.SEG_NEXT:
            if self.ime.move_segment(1):
                self._sync_now()
        elif action is Action.PREV:
            self.ime.move(-1)
            self._sync()
        elif action is Action.NEXT:
            self.ime.move(1)
            self._sync()
        elif action is Action.PASSTHROUGH:
            # 输入中遇到标点：先把首选词上屏，标点交给应用自己处理
            self._finish(self.ime.commit_current(), key)
            return False
        return True

    def _reset_comp(self) -> None:
        self._comp_start = 0.0
        self._comp_keys = 0
        self._comp_backs = 0
        self._comp_flips = 0

    def _record_abandon(self) -> None:
        """打到一半按 Esc：这是最有价值的负样本——说明候选里没有你想要的。"""
        if not self._comp_keys:
            return
        ms = int((time.monotonic() - self._comp_start) * 1000) if self._comp_start else 0
        memory.record(None, "", self._ime.mode if self._ime is not None else "simp",
                      kind="abandon", keys=self._comp_keys, backs=self._comp_backs,
                      flips=self._comp_flips, ms=ms)
        self._reset_comp()

    def _apply_boosts(self) -> None:
        """应用 AI 校对提出的提权：读队列 → 学进个人词频 → 删队列。"""
        path = config.boost_queue_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime == self._boost_mtime:
            return
        self._boost_mtime = mtime
        model = self._ime.user_model if self._ime is not None else None
        if model is None:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        applied = 0
        for line in lines:
            try:
                item = json.loads(line)
                model.learn(str(item["key"]), str(item["word"]), weight=int(item.get("weight", 3)))
                applied += 1
            except (ValueError, KeyError, TypeError):
                continue
        try:
            os.remove(path)
        except OSError:
            pass
        if applied:
            LOG.info("应用 AI 提权 %d 条", applied)

    def _memory_tick(self) -> bool:
        """空闲 60 秒且有待整理数据时，后台聚合一次（不占按键路径）。"""
        try:
            self._apply_boosts()
            idle = time.monotonic() - self._last_key_at
            if idle >= 60 and memory.pending() > 0:
                threading.Thread(target=memory.consolidate, daemon=True).start()
                self._last_key_at = time.monotonic()   # 避免重复触发
        except Exception:
            LOG.exception("记忆整理失败")
        return True

    def _finish(self, text: str | None, key: str | None = None, learn: bool = True,
                weight: int = 1, rank: int = 0) -> None:
        if text:
            if self._ime is not None and self._ime.last_commit_key:
                key = self._ime.last_commit_key
            # 学的是「简体形」：繁体/粤语只是显示层转换，选了「聲音」要记成「声音」，
            # 否则下次简体模式会把它当成另一个词的命中。
            base = self._ime.base_form(text) if self._ime is not None else text
            gap_ms = int((time.monotonic() - self._last_commit_at) * 1000) if self._last_commit_at else 0
            self._last_commit_at = time.monotonic()
            decision_ms = int((time.monotonic() - self._comp_start) * 1000) if self._comp_start else 0
            memory.record(key, text, self._ime.mode if self._ime is not None else "simp",
                          rank, gap_ms, keys=self._comp_keys, backs=self._comp_backs,
                          flips=self._comp_flips, ms=decision_ms)
            self._reset_comp()
            # 只在"你主动选了某个候选"时学习；空格直接收首选不算选择，
            # 否则输入法会把自己猜错的词学进去，越学越歪。
            typed = key
            if key and learn:
                model = self._ime.user_model if self._ime is not None else get_user_model()
                keys = (self._ime.learn_keys(typed, key, self._ime.last_commit_full)
                        if self._ime is not None else [key])
                for learn_key in keys:
                    model.learn(learn_key, base, weight)
                exposures = model.exposures(key, base)
                rate = f"{model.hit_rate(key, base):.0%}" if exposures else "-"
                LOG.info("上屏 %r → %r（记 %s；选中 %d 次 / 展示 %d 次 / 命中率 %s）",
                         key, text, "、".join(repr(k) for k in keys),
                         model.picks(key, base), exposures, rate)
                # 记住"上个词 → 这个词"，下次打完上一个词，下一个词会被提前
                model.learn_bigram(self._last_base, base)
                self._learn_phrase(key, base, model)
                self._last_base = base
                if self._ime is not None:
                    self._ime.context = base
            self._last_commit = text
            self.commit_text(IBus.Text.new_from_string(text))
            self._speak_auto(text)
        self._sync_now()

    def _refresh_tts_auto(self) -> None:
        """状态文件可能被 bin/ipinyin-tts 改过：上屏前重读，不用重启引擎。

        节流 1 秒：这个函数现在挂在每次同步上，每键都 open() 一次是浪费。
        """
        now = time.monotonic()
        if now - getattr(self, "_tts_checked", 0.0) < 1.0:
            return
        self._tts_checked = now
        if os.environ.get("IPINYIN_TTS_AUTO") == "1":
            self._tts_auto = True
            return
        try:
            with open(_tts_auto_path(), encoding="utf-8") as handle:
                self._tts_auto = handle.read().strip() == "1"
        except OSError:
            self._tts_auto = False

    def _speak_auto(self, text: str) -> None:
        """上屏即读：开关由 F9 或 ipinyin-tts on/off 控制（默认关，打字时不吵）。"""
        self._refresh_tts_auto()
        if not config.tts_enabled() or not self._tts_auto:
            return
        from .speak import get_speaker, voice_for

        mode = self._ime.mode if self._ime is not None else "simp"
        get_speaker().speak(text, voice_for(mode))

    def _schedule_neural(self) -> None:
        """粤语档：停手 200ms 后再把首选换成神经写法。

        神经推理 8~45ms，放在按键路径上就是「打字很卡」；先给规则版（0.02ms），
        停手再升级，看到的顺序不变、手感不卡。
        """
        if self._ime is None or self._ime.mode != "yue":
            return
        if self._neural_source:
            GLib.source_remove(self._neural_source)
        self._neural_source = GLib.timeout_add(200, self._neural_pass)

    def _neural_pass(self) -> bool:
        self._neural_source = 0
        if self._ime is not None and self._ime.rebuild_neural():
            self._sync_now()
        return False

    def _speak(self) -> bool:
        """Ctrl+回车：念当前候选；没候选就念刚上屏的那句。

        声线只由当前模式决定（粤语模式念粤语，简繁念普通话），
        念上屏句时也一样——否则粤语模式会拿普通话音去读「嘅 / 唔啱」。
        合成与播放都在后台线程，这里只是发一条指令，绝不阻塞主循环。
        """
        if not config.tts_enabled():
            return False
        from .speak import get_speaker, target

        word = self._ime.current_word() if self._ime is not None else None
        mode = self._ime.mode if self._ime is not None else "simp"
        picked = target(mode, word, self._last_commit)
        if picked is None:
            return False
        text, voice = picked
        if not get_speaker().speak(text, voice):
            return False
        LOG.info("朗读 %r（%s）", text, voice)
        return True

    def _learn_phrase(self, key: str, text: str, model) -> None:
        """连着打的两次上屏，合成一个词组记下来。

        比如先打「只」再打「会」，下次输入 zhihui 就能整词出「只会」——
        单个字都有、合起来没有的词，用一次就自动进库。
        """
        now = time.monotonic()
        previous = self._last_commit_info
        self._last_commit_info = (key, text, now)
        if not previous:
            return
        prev_key, prev_text, prev_time = previous
        if now - prev_time > 6.0:
            return
        joined = prev_text + text
        joined_key = prev_key + key
        if len(joined) > 6 or len(joined_key) > 24:
            return
        # 只合成「拼音能切成音节、且音节数=字数」的组合，
        # 否则会把 ranh+huigun 拼成「然回滚」这种垃圾词，还越点越靠前
        syllables = self.ime.lexicon.split_syllables(joined_key) if self.ime else []
        if not syllables or len(syllables) != len(joined):
            return
        model.learn(joined_key, joined, weight=2)
        LOG.info("合成词组 %r → %r", joined_key, joined)

    def _flash_mode_hint(self) -> None:
        """切换后先显示 2 秒——没在打字也能确认开关生效。"""
        self._push_mode_hint(force=True)
        if self._hint_source:
            GLib.source_remove(self._hint_source)
        self._hint_source = GLib.timeout_add(2000, self._clear_mode_hint)

    def _clear_mode_hint(self) -> bool:
        self._hint_source = 0
        self.update_auxiliary_text(IBus.Text.new_from_string(""), False)
        return False

    def _toggle_ai(self) -> None:
        """F10：启停屏幕感知（抓帧 + 理解两个 systemd 用户服务）。"""
        current = self._ai_want if self._ai_want is not None else self._ai_active()
        want_on = not current
        self._ai_want = want_on
        threading.Thread(target=self._set_ai, args=(want_on,), daemon=True).start()
        LOG.info("AI 屏幕感知 %s", "开启" if want_on else "关闭")
        self._flash_mode_hint()          # 立刻给反馈，别等 3 秒
        # 服务启停有延迟，3 秒后再刷新一次
        GLib.timeout_add(3000, self._refresh_hint_soon)

    def _refresh_hint_soon(self) -> bool:
        self._ai_checked = 0.0        # 强制重读心跳
        self._flash_mode_hint()
        return False

    def _set_ai(self, on: bool) -> None:
        units = ["ipinyin-screencast.service", "ipinyin-vision-watch.service"]
        try:
            subprocess.run(["systemctl", "--user", "start" if on else "stop", *units],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if not on:
                # 关掉后心跳还在 90 秒有效期内，界面会误显示「开」——直接删掉心跳
                try:
                    from . import screencast

                    os.remove(screencast.heartbeat_path())
                except OSError:
                    pass
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("切换 AI 感知失败：%s", exc)

    def _ai_active(self) -> bool:
        """AI 感知（ScreenCast 抓帧）是否在跑：看心跳文件，3 秒缓存。"""
        if self._ai_want is not None:
            return self._ai_want        # F10 按过之后以目标状态为准
        now = time.monotonic()
        if now - self._ai_checked < 3:
            return self._ai_on
        self._ai_checked = now
        try:
            from . import screencast

            self._ai_on = screencast.is_active()
        except Exception:
            self._ai_on = False
        return self._ai_on

    def _send_hint(self, text: str, visible: bool, force: bool = False) -> None:
        """发状态条，内容没变就不发（每次同步都发会多一次 D-Bus 往返）。"""
        if not force and (text, visible) == self._last_hint:
            return
        self._last_hint = (text, visible)
        self.update_auxiliary_text(IBus.Text.new_from_string(text), visible)

    def _push_mode_hint(self, force: bool = False) -> None:
        """在预编辑上方显示当前模式与朗读开关，避免忘了自己在哪个状态。"""
        if self._ime is None:
            return
        if self.ime.mode == "en":
            # 英文档必须显示：否则用户以为输入法坏了
            self._send_hint("英文 · Tab 回中文", True, force)
            return
        self._refresh_tts_auto()   # CLI 改了状态也要立刻反映到提示上
        label = {"simp": "简体", "trad": "繁体", "yue": "粤语"}.get(self.ime.mode, "")
        # 状态可见：音:开 = 上屏即读；音:关 = 只按 Ctrl+回车 才念
        label = (f"{label} · 音:{'开' if self._tts_auto else '关'}"
                 f" · AI:{'开' if self._ai_active() else '关'}")
        visible = bool(self.ime.buffer) or force
        self._send_hint(label if visible else "", visible, force)

    def _push_preedit(self) -> None:
        if self._ime is None:
            return
        view = self.ime.view()
        if not view.visible:
            return
        self.update_preedit_text(IBus.Text.new_from_string(view.preedit), view.cursor, True)
        self._push_mode_hint()

    def _sync(self) -> None:
        """节流版：距离上次刷新不足 15ms 就合并，避免每键都打一次 D-Bus。"""
        if self._ime is None:
            return
        now = time.monotonic()
        wait = SYNC_INTERVAL - (now - self._last_sync)
        if wait <= 0:
            self._sync_now()
        elif not self._sync_source:
            self._sync_source = GLib.timeout_add(int(wait * 1000) + 1, self._flush_sync)

    def _flush_sync(self) -> bool:
        self._sync_source = 0
        self._sync_now()
        return False

    def _sync_now(self) -> None:
        if self._sync_source:
            GLib.source_remove(self._sync_source)
            self._sync_source = 0
        self._last_sync = time.monotonic()
        if self._ime is None:
            return
        view = self.ime.view()
        if not view.visible:
            self.hide_preedit_text()
            self.hide_lookup_table()
            # 空输入时也要刷新状态栏：否则按 Tab 切了模式，提示还停在旧的那行
            self._push_mode_hint()
            return

        preedit = IBus.Text.new_from_string(view.preedit)
        # visible=True 已经等于"显示"，再调 show_preedit_text 只是多一次 D-Bus 往返
        self.update_preedit_text(preedit, view.cursor, True)

        table = IBus.LookupTable.new(self.ime.page_size, 0, True, True)
        for index, candidate in enumerate(view.candidates):
            table.append_candidate(IBus.Text.new_from_string(candidate))
            # 必须显式给序号：GNOME Shell 的候选窗按 label 逐条取，没有 label 就一条都不显示
            table.set_label(index, IBus.Text.new_from_string(str(index + 1)))
        table.set_cursor_pos(view.highlighted)
        self.update_lookup_table(table, True)
        # 状态栏（简体/繁体/粤语 · 音 · AI）必须跟着候选一起刷新——
        # 以前只有 _push_preedit()（敲字母）会刷它，按 Tab 切模式时它不动，
        # 于是候选变了、提示还是旧的
        self._push_mode_hint()


def _acquire_singleton() -> object | None:
    """单实例锁：同一时刻只允许一个引擎注册。

    没有它就会出现「两个引擎抢同一个名字」——IBus 只认第一个，
    按鍵全部送进已经半死的那个，输入法等于坏了（真实故障）。
    拿不到锁时先等 5 秒（给旧进程退出时间），还不行就退出。
    """
    import fcntl

    path = os.path.join(config.cache_home(), "engine.lock")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "w")
    except OSError:
        return object()          # 拿不到锁文件也不能让输入法挂掉
    for attempt in range(11):
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(str(os.getpid()))
            handle.flush()
            return handle
        except OSError:
            if attempt >= 10:
                return None
            time.sleep(0.5)
    return None


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argv = list(sys.argv if argv is None else argv)
    LOG.info("启动 ipinyin 引擎 pid=%d args=%s", os.getpid(), argv[1:])
    lock = _acquire_singleton()
    if lock is None:
        LOG.warning("已有另一个引擎在跑（锁被占用），本进程退出，避免重复注册")
        return 0

    # 词库放后台线程加载：主循环一启动就能应答按键，绝不出现「卡住 → 飙一堆字」。
    def _preload() -> None:
        started = time.perf_counter()
        try:
            get_lexicon()
            get_user_model().maybe_compact()
            _LEXICON_READY.set()
            # 词库是几十万个长寿命对象，冻结后不再参与 GC 扫描，省掉偶发的回收停顿
            gc.freeze()
            gc.set_threshold(50000, 100, 100)  # 减少全量回收，避免偶发几十毫秒停顿
            LOG.info("词库预加载完成 %.0f ms", (time.perf_counter() - started) * 1000)
        except Exception:  # 词库坏了也要能启动，最多是打不出字
            LOG.exception("词库加载失败")

    threading.Thread(target=_preload, name="lexicon-preload", daemon=True).start()

    IBus.init()
    bus = IBus.Bus()
    if not bus.is_connected():
        LOG.error("连不上 IBus 总线")
        return 1

    # 先导出 factory 再抢名字：ibus-daemon 是盯着名字变化来 CreateEngine 的，
    # 顺序反了就会撞上「Factory 还不存在」的竞态（第一次切换必失败）。
    factory = IBus.Factory.new(bus.get_connection())
    factory.add_engine("ipinyin", PinyinEngine.__gtype__)
    if not bus.request_name("org.freedesktop.IBus.IPinyin", 0):
        # 名字被占说明已有实例在服务，这个进程直接退出，避免每次切换多留一个吃内存的僵尸
        LOG.info("已有 ipinyin 引擎在跑，本进程退出")
        return 0
    # 常驻模式下 ibus-daemon 重启会让本进程的连接失效：发现总线消失就退出，
    # 交给 systemd 拉起一个新进程重新连上，避免留着一个"活着但没连接"的僵尸。
    quit_on_vanish = threading.Event()

    def _on_vanished(_conn, _name):
        LOG.warning("IBus 总线消失，退出等待重启")
        quit_on_vanish.set()
        IBus.quit()

    Gio.bus_watch_name_on_connection(
        bus.get_connection(), "org.freedesktop.IBus", Gio.BusNameWatcherFlags.NONE, None, _on_vanished
    )

    # 引擎被杀后重启时，ibus-daemon 会把"当前引擎"清空（表现为完全打不了字）。
    # 这里发现引擎位空缺就自荐，恢复可用状态。
    # 注意：这里不能同步自荐（会死锁），一律走后台线程 + 定时复查
    for delay in (3000, 6000, 15000, 30000, 60000):
        GLib.timeout_add(delay, lambda: (_ensure_global_engine("启动后复查"), False)[1])

    LOG.info("引擎已注册，进入主循环")
    IBus.main()
    return 1 if quit_on_vanish.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
