#!/usr/bin/env python3
"""IPinyin 测试垫：不依赖系统输入源，直接在窗口里体验引擎效果。

给「系统输入源还没注册好」的过渡期用 —— 打开就是一个小窗口，
按键走 unix socket 打进真正的引擎（和 IMKit 外壳同一条链路），
上屏文本累积在文本框里，候选展示在状态条。

用法：python3 tools/macos_testpad.py
"""
from __future__ import annotations

import json
import os
import socket
import tkinter as tk

SOCK = os.path.expanduser("~/Library/Application Support/ipinyin/engine.sock")

# 特殊键：Tk keysym → IBus keyval
SPECIAL = {
    "Return": 0xFF0D, "KP_Enter": 0xFF0D, "BackSpace": 0xFF08, "Escape": 0xFF1B,
    "Tab": 0xFF09, "space": 0x20, "Prior": 0xFF55, "Next": 0xFF56,
    "Up": 0xFF52, "Down": 0xFF54, "Left": 0xFF51, "Right": 0xFF53,
}


class Testpad:
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(SOCK)
        self.root = tk.Tk()
        self.root.title("IPinyin 测试垫（Shift 轻按切中/英，Tab 切简繁粤）")
        self.text = tk.Text(self.root, height=12, width=60, font=("PingFang SC", 16))
        self.text.pack(padx=8, pady=(8, 2))
        self.status = tk.Label(self.root, text="英文模式", anchor="w",
                               font=("PingFang SC", 12), fg="#666")
        self.status.pack(fill="x", padx=8, pady=(0, 8))
        self.ascii_mode = False
        self.shift_t0 = 0
        self.typed_during_shift = False
        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease-Shift_L>", self.on_shift_up)
        self.root.bind("<KeyRelease-Shift_R>", self.on_shift_up)
        self.root.focus_set()

    def send(self, req: dict) -> dict:
        self.sock.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            buf += self.sock.recv(65536)
        return json.loads(buf)

    def on_shift_up(self, event) -> None:
        if self.shift_t0 == 0:
            return
        elapsed = event.time - self.shift_t0
        self.shift_t0 = 0
        if elapsed < 350 and not self.typed_during_shift:
            self.ascii_mode = not self.ascii_mode
            self.typed_during_shift = False

    def on_press(self, event) -> str:
        if event.keysym in ("Shift_L", "Shift_R"):
            self.shift_t0 = event.time
            self.typed_during_shift = False
            return "break"
        if self.ascii_mode:
            return ""
        state = 1 if event.state & 0x1 else 0
        if state:
            self.typed_during_shift = True
        keysym = event.keysym
        if keysym in SPECIAL:
            keyval = SPECIAL[keysym]
        elif len(event.char) == 1 and 0x20 <= ord(event.char) <= 0x7E:
            keyval = ord(event.char)
        else:
            return ""                       # 其他键交还窗口
        resp = self.send({"t": "event", "keyval": keyval, "state": state})
        if resp.get("commit"):
            self.text.insert("insert", resp["commit"])
            self.text.see("insert")
        label = "英文" if self.ascii_mode else resp.get("mode_label", "")
        cand = "  ".join(resp.get("candidates", [])[:5])
        self.status.config(text=f"[{label}]  {resp.get('preedit', '')}    {cand}")
        return "break"

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Testpad().run()
