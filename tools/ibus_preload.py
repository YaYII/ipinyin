#!/usr/bin/env python3
"""把 ipinyin 加进 IBus 预加载列表。

IBus 只在「切换到一个还没启动的引擎」时才去 spawn 引擎进程，而 spawn 要等
Python 起来注册 factory，这个空档里第一次切换会报 “Object does not exist at
path /org/freedesktop/IBus/Factory”。预加载后引擎进程在登录时就已存在，
切换就没有这个竞态。
"""
from __future__ import annotations

import subprocess
import sys

SCHEMA = "org.freedesktop.ibus.general"
KEY = "preload-engines"
ENGINE = "ipinyin"


def main(action: str) -> int:
    try:
        raw = subprocess.run(
            ["gsettings", "get", SCHEMA, KEY], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"读取 {SCHEMA} {KEY} 失败：{exc}", file=sys.stderr)
        return 1

    if raw.startswith("@as"):
        raw = raw[3:].strip()  # gsettings 的空数组会打印成 "@as []"
    engines = [
        item.strip().strip("'")
        for item in raw.strip("[]").split(",")
        if item.strip() and not item.strip().startswith("@")
    ]
    if action == "add":
        if ENGINE not in engines:
            engines.append(ENGINE)
    elif action == "remove":
        engines = [item for item in engines if item != ENGINE]
    else:
        print("用法: ibus_preload.py add|remove", file=sys.stderr)
        return 2

    value = "[" + ", ".join(f"'{item}'" for item in engines) + "]"
    subprocess.run(["gsettings", "set", SCHEMA, KEY, value], check=True)
    print(f"preload-engines = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "add"))
