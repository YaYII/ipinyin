#!/usr/bin/env python3
"""GNOME 输入源（gsettings）增删：把 ipinyin 注册进 GNOME 的输入法列表。

GNOME 在 Wayland 下的输入源来自 org.gnome.desktop.input-sources，光有 IBus
引擎还不够，得在这里登记才会出现在 Super+Space 的切换里。
"""
from __future__ import annotations

import ast
import subprocess
import sys

SCHEMA = "org.gnome.desktop.input-sources"
KEY = "sources"
SOURCE = ("ibus", "ipinyin")


def read_sources() -> list[tuple[str, str]]:
    raw = subprocess.run(
        ["gsettings", "get", SCHEMA, KEY], check=True, capture_output=True, text=True
    ).stdout.strip()
    return [tuple(item) for item in ast.literal_eval(raw)]


def write_sources(sources: list[tuple[str, str]]) -> None:
    value = "[" + ", ".join(f"('{kind}', '{name}')" for kind, name in sources) + "]"
    subprocess.run(["gsettings", "set", SCHEMA, KEY, value], check=True)


def main(action: str) -> int:
    sources = read_sources()
    if action == "add":
        if SOURCE in sources:
            print("GNOME 输入源里已经有 ipinyin")
            return 0
        # 放在第一位：GNOME 的新窗口默认用第一项，"每次都要重新选"就是这么来的
        sources.insert(0, SOURCE)
        write_sources(sources)
        subprocess.run(["gsettings", "set", SCHEMA, "current", "0"], check=False)
        print("已加入 GNOME 输入源：Super+Space 可切换")
    elif action == "remove":
        if SOURCE not in sources:
            print("GNOME 输入源里没有 ipinyin")
            return 0
        sources = [item for item in sources if item != SOURCE]
        if not sources:
            sources = [("xkb", "us")]
        write_sources(sources)
        print("已从 GNOME 输入源移除 ipinyin")
    else:
        print("用法: gnome_source.py add|remove", file=sys.stderr)
        return 2
    print("当前输入源：", sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "add"))
