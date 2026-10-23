#!/usr/bin/env python3
"""手动给输入法加词：单个字有、合起来没有的词，一条命令加进去。

用法：
  python3 tools/add_word.py 只会 zhi hui
  python3 tools/add_word.py 无息贷款 wu xi dai kuan
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ipinyin.user_model import UserModel  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    word = sys.argv[1]
    key = "".join(sys.argv[2:]).lower().replace("'", "")
    path = os.path.join(os.path.expanduser("~"), ".local/share/ipinyin/data/user_model.tsv")
    model = UserModel(path)
    model.learn(key, word, weight=50)
    model.compact()
    print(f"已加入：{word} ← {key}（权重 50，下一个窗口即生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
