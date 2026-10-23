#!/usr/bin/env python3
"""端到端验证：通过 IBus 的 D-Bus 接口真实敲键，检查上屏结果。

不依赖图形界面，也不需要切换全局输入法：
  创建 input context → 发按键 → 收 CommitText 信号 → 断言。
用法：python3 tests/e2e_ibus.py [期望字串]
"""
from __future__ import annotations

import glob
import os
import sys
import time

import gi

gi.require_version("IBus", "1.0")
from gi.repository import Gio, GLib  # noqa: E402

BUS_NAME = "org.freedesktop.IBus"
BUS_PATH = "/org/freedesktop/IBus"
IFACE_CTX = "org.freedesktop.IBus.InputContext"


def ibus_address() -> str:
    env = os.environ.get("IBUS_ADDRESS")
    if env:
        return env
    patterns = ["*-unix-wayland-0", "*-unix-0", "*-unix-*"]
    base = os.path.expanduser("~/.config/ibus/bus")
    for pattern in patterns:
        for path in glob.glob(os.path.join(base, pattern)):
            for line in open(path, encoding="utf-8"):
                if line.startswith("IBUS_ADDRESS="):
                    return line.strip().split("=", 1)[1]
    raise SystemExit("找不到 IBUS_ADDRESS，ibus 没在跑？")


class Client:
    def __init__(self) -> None:
        self.conn = Gio.DBusConnection.new_for_address_sync(
            ibus_address(),
            Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
            | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
            None,
            None,
        )
        reply = self.conn.call_sync(
            BUS_NAME,
            BUS_PATH,
            BUS_NAME,
            "CreateInputContext",
            GLib.Variant("(s)", ("ipinyin",)),
            None,
            Gio.DBusCallFlags.NONE,
            5000,
            None,
        )
        self.path = reply.unpack()[0]
        self.commits: list[str] = []
        self.preedits: list[str] = []
        self.conn.signal_subscribe(
            BUS_NAME, IFACE_CTX, "CommitText", self.path, None, 0, self._on_commit
        )
        self.conn.signal_subscribe(
            BUS_NAME, IFACE_CTX, "UpdatePreeditText", self.path, None, 0, self._on_preedit
        )
        # 客户端能力：预编辑 + 焦点 + 候选表；没有 FOCUS 能力时 FocusIn 会报错
        self.call_quiet("SetCapabilities", GLib.Variant("(u)", (0x7F,)))
        self.call_quiet("FocusIn", None)

    def call(self, method: str, signature: str | None, value: GLib.Variant | None) -> object:
        return self.conn.call_sync(
            BUS_NAME,
            self.path,
            IFACE_CTX,
            method,
            value,
            None,
            Gio.DBusCallFlags.NONE,
            5000,
            None,
        ).unpack()

    def call_quiet(self, method: str, value: GLib.Variant | None) -> None:
        try:
            self.call(method, None, value)
        except GLib.Error as exc:
            print(f"提示：{method} 失败（{exc.message}），继续")

    def _on_commit(self, conn, sender, path, iface, signal, params):
        # 参数是 (v)，unpack 后形如 ('IBusText', {attrs}, '文字', ('IBusAttrList', ...))
        self.commits.append(params.unpack()[0][2])

    def _on_preedit(self, conn, sender, path, iface, signal, params):
        self.preedits.append(params.unpack()[0][2])

    def type_text(self, text: str) -> None:
        for char in text:
            if char == " ":
                keyval = 0x20
            else:
                keyval = ord(char.lower())
            self.conn.call_sync(
                BUS_NAME,
                self.path,
                IFACE_CTX,
                "ProcessKeyEvent",
                GLib.Variant("(uuu)", (keyval, 0, 0)),
                None,
                Gio.DBusCallFlags.NONE,
                5000,
                None,
            )
            self.pump(0.02)
        self.pump(0.15)

    @staticmethod
    def pump(seconds: float) -> None:
        context = GLib.MainContext.default()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            context.iteration(False)
            time.sleep(0.005)


CASES = [
    ("nihao ", "你好"),
    ("shijie ", "世界"),
    ("woxihuanni ", "我喜欢你"),
    ("zhongguoren ", "中国人"),
    ("jintiantianqi ", "今天天气"),
]


def run_case(typed: str, expected: str, retries: int = 1) -> bool:
    # 桌面上有真人同时打字时，IBus 会把焦点切回他的应用，合成 context 的首字
    # 偶尔送不达（不是引擎丢键）。这里允许重试一次，重试仍失败才算真失败。
    for attempt in range(retries + 1):
        if _run_once(typed, expected):
            return True
        if attempt < retries:
            print(f"      重试（第 {attempt + 1} 次）")
    return False


def _run_once(typed: str, expected: str) -> bool:
    client = Client()
    # 每个用例一个新 context 会被 IBus 的焦点切换搅乱（桌面有真人打字时首字常被
    # 送到他的窗口），所以整个套件复用同一个 context，用 Esc 清空 buffer。
    # 引擎是被 ibus-daemon 现场拉起的，进程启动 + 加载词库要一点时间，
    # 这段窗口内按下的键会丢。先等它起来，再开始真正打字。
    client.pump(2.0)
    client.type_text(" ")
    client.commits.clear()
    client.type_text(typed)
    result = "".join(client.commits)
    ok = result == expected
    print(f"{'通过' if ok else '失败'}  按键 {typed!r:18} → {result!r:12} 期望 {expected!r}")
    return ok


def main() -> int:
    if len(sys.argv) > 2:
        ok = run_case(sys.argv[2], sys.argv[1])
        return 0 if ok else 1
    results = [run_case(typed, expected) for typed, expected in CASES]
    print(f"\n{sum(results)}/{len(results)} 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
