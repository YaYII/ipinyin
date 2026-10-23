"""IBus 引擎适配层：把按键事件翻译成状态机动作，再画到 IBus 的 preedit / 候选表上。

这一层只做「翻译与显示」，所有输入逻辑都在 ime.py（纯 Python，可单测）。
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import gi

gi.require_version("IBus", "1.0")
from gi.repository import Gio, GLib, GObject, IBus  # noqa: E402

from . import config  # noqa: E402
from .ime import ImeState  # noqa: E402
from .keys import Action, translate  # noqa: E402
from .lexicon import Lexicon  # noqa: E402
from .t2s import Converter  # noqa: E402
from .yue import YueConverter  # noqa: E402
from .user_model import UserModel  # noqa: E402

LOG = logging.getLogger("ipinyin")

_LEXICON: Lexicon | None = None
# 打字很快时，候选窗没必要每键都重画：把 UI 刷新合并到 15ms 一次（约 60fps）
SYNC_INTERVAL = 0.015
_USER_MODEL: UserModel | None = None
_CONVERTER: Converter | None = None
_YUE: YueConverter | None = None
# 词库加载完成前，按键一律放行（当英文打），绝不阻塞 IBus 主循环
_LEXICON_READY = threading.Event()


def get_lexicon() -> Lexicon:
    global _LEXICON
    if _LEXICON is None:
        path = config.lexicon_path()
        LOG.info("加载词库 %s", path)
        _LEXICON = Lexicon.load(path)
        LOG.info("词库就绪：%d 个拼音键", len(_LEXICON))
    return _LEXICON


def get_yue() -> YueConverter:
    global _YUE
    if _YUE is None:
        _YUE = YueConverter.load_dict(config.yue_dict_path(), YueConverter.load(config.yue_path())._map)
        LOG.info("粤语对照 %d 条", len(_YUE))
    return _YUE


def _read_mode() -> str:
    """记住上次的简/繁选择。"""
    try:
        with open(config.mode_path(), encoding="utf-8") as fh:
            mode = fh.read().strip()
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
    """拼音键能切成几个音节，就该对应几个汉字；对不上的是脏数据（如 zenm→怎唔）。"""
    if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
        return True
    return len(word) in get_lexicon().syllable_counts(key)


def get_user_model() -> UserModel:
    global _USER_MODEL
    if _USER_MODEL is None:
        _USER_MODEL = UserModel(
            config.user_model_path(),
            config.user_seed_path(),
            config.user_bigram_path(),
            config.user_bigram_seed_path(),
            validator=_valid_pair,
        )
        LOG.info("个人词频 %d 条", len(_USER_MODEL))
    return _USER_MODEL


def get_converter() -> Converter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = Converter.load(config.s2t_path())
        LOG.info("简繁映射 %d 字", len(_CONVERTER))
    return _CONVERTER


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
        self._last_sync = 0.0
        self._sync_source = 0
        self._last_commit_info: tuple[str, str, float] | None = None
        self._english = False       # 轻按 Shift 在中/英之间切换
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
            self._ime = ImeState(get_lexicon(), user_model=get_user_model(),
                                 converter=get_converter(), yue=get_yue())
            self._ime.mode = _read_mode()
        return self._ime

    # ---------- IBus 虚函数 ----------
    def do_process_key_event(self, keyval, keycode, state):
        try:
            started = time.perf_counter()
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

    def do_focus_in(self):
        self._sync()

    def do_reset(self):
        self._reset()

    def do_disable(self):
        self._reset()
        get_user_model().flush_if_dirty()

    def do_candidate_clicked(self, index, button, state):
        try:
            key = self.ime.buffer
            self._finish(self.ime.select_page_index(int(index)), key, weight=3)
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
                    self._english = not self._english
                    self._reset()
                    LOG.info("切换到%s输入", "英文" if self._english else "中文")
                self._shift_down = False
            else:
                self._shift_down = True
                self._shift_clean = True
            return False
        if self._shift_down:
            self._shift_clean = False
        if self._english:
            return False

        action, value = translate(keyval, state)
        LOG.debug("key %#x state=%#x -> %s %r buffer=%r", keyval, state, action.name, value, self.ime.buffer)
        if action is Action.IGNORE:
            return False

        composing = bool(self.ime.buffer)

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
            self.ime.insert(value)
            self._push_preedit()   # 先把字画出来，候选慢一点也不会觉得"卡"
            self._sync()
            return True

        if not composing:
            return False

        key = self.ime.buffer
        if action is Action.DIGIT:
            self._finish(self.ime.select(int(value)), key, weight=3)
        elif action is Action.BACKSPACE:
            self.ime.backspace()
            self._sync()
        elif action is Action.COMMIT:
            self._finish(self.ime.commit_current(), key)
        elif action is Action.COMMIT_RAW:
            self._finish(self.ime.commit_raw(), key)
        elif action is Action.CLEAR:
            self._reset()
        elif action is Action.PAGE_PREV:
            self.ime.page_by(-1)
            self._sync()
        elif action is Action.PAGE_NEXT:
            self.ime.page_by(1)
            self._sync()
        elif action is Action.TOGGLE_TRAD:
            mode = self.ime.cycle_mode()
            _write_mode(mode)
            LOG.info("切换到%s输出", {"simp": "简体", "trad": "繁体", "yue": "粤语"}[mode])
            self._sync_now()
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

    def _finish(self, text: str | None, key: str | None = None, learn: bool = True, weight: int = 1) -> None:
        if text:
            if self._ime is not None and self._ime.last_commit_key:
                key = self._ime.last_commit_key
            # 只在"你主动选了某个候选"时学习；空格直接收首选不算选择，
            # 否则输入法会把自己猜错的词学进去，越学越歪。
            if key and learn:
                model = get_user_model()
                model.learn(key, text, weight)
                exposures = model.exposures(key, text)
                rate = f"{model.hit_rate(key, text):.0%}" if exposures else "-"
                LOG.info("上屏 %r → %r（选中 %d 次 / 展示 %d 次 / 命中率 %s）",
                         key, text, model.picks(key, text), exposures, rate)
                # 记住"上个词 → 这个词"，下次打完上一个词，下一个词会被提前
                model.learn_bigram(self._last_commit, text)
                self._learn_phrase(key, text, model)
                self._last_commit = text
                if self._ime is not None:
                    self._ime.context = text
            self.commit_text(IBus.Text.new_from_string(text))
        self._sync_now()

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
        if now - prev_time > 2.0:
            return
        joined = prev_text + text
        joined_key = prev_key + key
        if len(joined) > 6 or len(joined_key) > 24:
            return
        model.learn(joined_key, joined, weight=2)
        LOG.info("合成词组 %r → %r", joined_key, joined)

    def _push_mode_hint(self) -> None:
        """在预编辑上方显示当前模式（简体/繁体/粤语），避免忘了在哪个模式。"""
        if self._ime is None:
            return
        label = {"simp": "简体", "trad": "繁体", "yue": "粤语"}.get(self.ime.mode, "")
        visible = bool(self.ime.buffer)
        self.update_auxiliary_text(IBus.Text.new_from_string(label if visible else ""), visible)

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


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argv = list(sys.argv if argv is None else argv)
    LOG.info("启动 ipinyin 引擎 pid=%d args=%s", os.getpid(), argv[1:])

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
    try:
        current = bus.get_global_engine()
        current_name = current.get_name() if current is not None else ""
    except GLib.Error:
        current_name = ""
    if not current_name:
        LOG.info("当前没有全局引擎，自动切回 ipinyin")
        try:
            bus.set_global_engine("ipinyin")
        except GLib.Error as exc:
            LOG.warning("自动设置全局引擎失败：%s", exc.message)

    LOG.info("引擎已注册，进入主循环")
    IBus.main()
    return 1 if quit_on_vanish.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
