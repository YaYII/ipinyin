#!/usr/bin/env python3
"""切换输入方案：全拼（默认）/ 小鹤双拼。

用法：
  python3 tools/set_input_mode.py            # 查看当前
  python3 tools/set_input_mode.py full       # 全拼
  python3 tools/set_input_mode.py flypy      # 小鹤双拼
之后重启引擎：systemctl --user restart ipinyin-engine
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ipinyin.config as config  # noqa: E402

MODES = {"full": "全拼", "flypy": "小鹤双拼"}


def main() -> int:
    path = config.input_mode_path()
    if len(sys.argv) < 2:
        current = open(path, encoding="utf-8").read().strip() if os.path.exists(path) else "full"
        print(f"当前输入方案：{MODES.get(current, current)}（{path}）")
        return 0
    mode = sys.argv[1]
    if mode not in MODES:
        print(f"未知方案 {mode}，可选：{' / '.join(MODES)}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(mode)
    subprocess.run(["systemctl", "--user", "restart", "ipinyin-engine"], capture_output=True)
    print(f"已切换到 {MODES[mode]}，引擎已重启")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
