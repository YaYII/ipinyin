#!/usr/bin/env python3
"""macOS 输入法外壳与 Python 引擎之间的 JSON 桥（纯标准库，/usr/bin/python3 可跑）。

InputMethodKit 外壳（Swift）不认识 ImeState，这里用一行 JSON 进 / 一行 JSON 出的
unix socket 协议把引擎暴露给它。语义与 IBus 引擎（engine.py）逐条对齐：
字母进缓冲、数字选词（选中算学习 weight=3）、空格上首选、回车上原串、
Tab 简繁粤、Esc 清空、标点先上屏再直通、空格合成词组、2 秒内连打合成词组。

请求（每行一个）：
  {"t":"event","keyval":<int>,"state":<int>}   IBus 风格键值（Swift 端做映射）
  {"t":"select","index":<int>}                 候选窗点选
  {"t":"state"}                                只要当前状态
  {"t":"reset"}                                客户端切换会话时清空
  {"t":"ping"}
应答：
  {"ok":true,"handled":bool,"commit":str|null,"passthrough":bool,
   "preedit":str,"visible":bool,"candidates":[...],"words":[...],
   "highlight":int,"page":int,"page_count":int,"mode":"simp|trad|yue","mode_label":"简体"}

用法：python3 tools/macos_server.py [--data 目录] [--socket 路径]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipinyin import keys  # noqa: E402
from ipinyin.ime import ImeState  # noqa: E402
from ipinyin.keys import Action, translate  # noqa: E402
from ipinyin.lexicon import Lexicon  # noqa: E402
from ipinyin.user_model import UserModel  # noqa: E402
from ipinyin.yue import YueConverter  # noqa: E402

PHRASE_WINDOW = 2.0        # 连打合成词组的时间窗（与 engine.py 一致）


class Session:
    """一个输入会话的全部状态：词库 + ImeState + 个人模型 + 学习。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.lexicon = Lexicon.load(os.path.join(data_dir, "lexicon.tsv"))
        self._load_data_home()
        self.ime = ImeState(self.lexicon, user_model=self.model)
        self._load_lm()
        self.ime.attach_lm(self.lm, 0.4)
        self._load_yue()
        self._last_commit = ""       # 上一次上屏的词（词对学习用）
        self._last_commit_info: tuple[str, str, float] | None = None
        self._commit: str | None = None   # 本次按键产出的上屏文本

    # ---- 装载（路径策略：用户数据在 ~/Library/Application Support，词库用包内数据）----
    def _load_data_home(self) -> None:
        home = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "ipinyin")
        os.makedirs(os.path.join(home, "data"), exist_ok=True)
        model_path = os.path.join(home, "data", "user_model.tsv")
        seed = os.path.join(self.data_dir, "user_seed.tsv")
        bigram = os.path.join(self.data_dir, "user_bigram_seed.tsv")

        def valid(key: str, word: str) -> bool:
            if any(not ("\u4e00" <= ch <= "\u9fff") for ch in word):
                return True
            return len(word) in self.lexicon.syllable_counts(key)

        self.model = UserModel(model_path, seed_path=seed if os.path.exists(seed) else None,
                               bigram_path=os.path.join(home, "data", "user_bigram.tsv"),
                               bigram_seed_path=bigram if os.path.exists(bigram) else None,
                               validator=valid)

    def _load_lm(self) -> None:
        self.lm = None
        path = os.path.join(self.data_dir, "lm_char.marshal")
        if os.path.exists(path):
            try:
                from ipinyin.lm import CharLM

                self.lm = CharLM.load(path)
            except Exception:
                self.lm = None

    def _load_yue(self) -> None:
        self.ime.yue = None
        try:
            yue = YueConverter.load_dict(
                os.path.join(self.data_dir, "yue.marshal"),
                YueConverter.load(os.path.join(self.data_dir, "yue.tsv"))._map)
            guard_path = os.path.join(self.data_dir, "yue_guard.marshal")
            if os.path.exists(guard_path):
                from ipinyin.yue_guard import YueGuard

                yue.guard = YueGuard.load(guard_path)
            nn_path = os.path.join(self.data_dir, "yue_nn.onnx")
            if os.path.exists(nn_path):
                try:
                    from ipinyin.yue_nn import NeuralYue

                    yue.nn = NeuralYue.load(self.data_dir)
                except Exception:
                    pass
            self.ime.yue = yue
        except Exception:
            pass                      # 粤语层坏了不连累简体打字

    # ---- 事件处理（与 engine.py 的分发逐条对齐）----
    def feed(self, keyval: int, state: int) -> dict:
        if state & keys.RELEASE_MASK:
            return self._state(handled=False)
        if state & (keys.CONTROL_MASK | keys.MOD1_MASK | keys.SUPER_MASK):
            return self._state(handled=False)

        action, value = translate(keyval, state)
        if action is Action.IGNORE:
            return self._state(handled=False)

        composing = bool(self.ime.buffer)
        if action is Action.LETTER:
            if value == "'" and not composing:
                return self._state(handled=False)
            self.ime.insert(value)
            return self._state(handled=True)

        if not composing:
            return self._state(handled=False)

        key = self.ime.buffer
        if action is Action.DIGIT:
            self._finish(self.ime.select(int(value)), key, weight=3)
        elif action is Action.BACKSPACE:
            self.ime.backspace()
        elif action is Action.COMMIT:
            self._finish(self.ime.commit_current(), key)
        elif action is Action.COMMIT_RAW:
            self._finish(self.ime.commit_raw(), key)
        elif action is Action.CLEAR:
            self.ime.clear()
        elif action is Action.PAGE_PREV:
            self.ime.page_by(-1)
        elif action is Action.PAGE_NEXT:
            self.ime.page_by(1)
        elif action is Action.TOGGLE_TRAD:
            self.ime.cycle_mode()
        elif action is Action.SEG_PREV:
            self.ime.move_segment(-1)
        elif action is Action.SEG_NEXT:
            self.ime.move_segment(1)
        elif action is Action.PREV:
            self.ime.move(-1)
        elif action is Action.NEXT:
            self.ime.move(1)
        elif action is Action.PASSTHROUGH:
            # 输入中遇到标点：先把首选词上屏，标点交给应用自己处理
            self._finish(self.ime.commit_current(), key)
            return self._state(handled=False)
        return self._state(handled=True)
    def select_index(self, index: int) -> dict:
        """候选窗里点选（全局下标）。"""
        if not self.ime.buffer:
            return self._state(handled=False)
        key = self.ime.buffer
        self._finish(self.ime.select_page_index(index), key, weight=3)
        return self._state(handled=True)

    def _finish(self, text: str | None, key: str | None = None,
                learn: bool = True, weight: int = 1) -> None:
        if not text:
            return
        self._commit = text               # 本次按键要上屏的内容
        if self.ime.last_commit_key:
            key = self.ime.last_commit_key
        if key and learn:
            self.model.learn(key, text, weight)
            self.model.learn_bigram(self._last_commit, text)
            self._learn_phrase(key, text)
            self._last_commit = text
            self.ime.context = text
            self.model.maybe_compact()

    def _learn_phrase(self, key: str, text: str) -> None:
        """连着打的两次上屏合成一个词组（先「西」后「大」→ xidai 西大）。"""
        now = time.monotonic()
        previous = self._last_commit_info
        self._last_commit_info = (key, text, now)
        if not previous:
            return
        prev_key, prev_text, prev_time = previous
        if now - prev_time > PHRASE_WINDOW:
            return
        joined = prev_text + text
        joined_key = prev_key + key
        if len(joined) > 6 or len(joined_key) > 24:
            return
        self.model.learn(joined_key, joined, weight=2)

    def reset(self) -> None:
        self.ime.clear()
        self._last_commit = ""
        self._last_commit_info = None

    # ---- 状态快照 ----
    def _state(self, handled: bool) -> dict:
        view = self.ime.view()
        commit = self._commit
        self._commit = None               # 一次性：随应答带走
        return {
            "ok": True,
            "handled": handled,
            "commit": commit,
            "passthrough": not handled and commit is not None,
            "preedit": view.preedit,
            "visible": view.visible,
            "candidates": view.candidates,
            "words": view.words,
            "highlight": view.highlighted,
            "page": view.page,
            "page_count": view.page_count,
            "page_size": self.ime.page_size,
            "mode": self.ime.mode,
            "mode_label": {"simp": "简体", "trad": "繁体", "yue": "粤语"}.get(self.ime.mode, ""),
        }

    def state(self) -> dict:
        return self._state(handled=True)


class Bridge:
    """把 Session 包成行协议服务；commit 通过 _pending_commit 传递。"""

    def __init__(self, session: Session):
        self.session = session
        self._lock = threading.Lock()
        self._commit: str | None = None

    def handle(self, line: str) -> str:
        try:
            req = json.loads(line)
        except ValueError:
            return json.dumps({"ok": False, "error": "bad json"}, ensure_ascii=False)
        with self._lock:
            kind = req.get("t", "event")
            if kind == "ping":
                return json.dumps({"ok": True, "pong": True})
            if kind == "reset":
                self.session.reset()
                return json.dumps(self.session.state(), ensure_ascii=False)
            if kind == "state":
                return json.dumps(self.session.state(), ensure_ascii=False)
            if kind == "select":
                resp = self.session.select_index(int(req.get("index", 0)))
            else:
                resp = self.session.feed(int(req.get("keyval", 0)), int(req.get("state", 0)))
            # feed 内部不直接知道 commit —— 从学习路径捞出来
            return json.dumps(resp, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--socket", default=os.path.join(
        os.path.expanduser("~"), "Library", "Application Support", "ipinyin", "engine.sock"))
    ap.add_argument("--stdio", action="store_true", help="调试：stdin/stdout 行协议，不起 socket")
    args = ap.parse_args()

    session = Session(args.data)
    bridge = Bridge(session)

    if args.stdio:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            sys.stdout.write(bridge.handle(line) + "\n")
            sys.stdout.flush()
        return 0

    sock_dir = os.path.dirname(args.socket)
    os.makedirs(sock_dir, exist_ok=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(4)
    os.chmod(args.socket, 0o600)

    def serve(conn: socket.socket) -> None:
        buf = b""
        try:
            with conn:
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            conn.sendall(bridge.handle(line.decode("utf-8")).encode("utf-8") + b"\n")
        except OSError:
            pass

    print(f"ipinyin engine bridge → {args.socket}", flush=True)
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
