"""守护：engine 里引用的每个 Action.X 必须在 keys.Action 里存在。

踩过的坑：engine 用了 Action.VOICE，但部署的 keys.py 是 deb 装的旧版本，
没有这个枚举 —— 每次按键都抛 AttributeError，整个输入法打不出字。
"""
import re
from pathlib import Path

from ipinyin import engine, keys


def test_engine_action_names_exist():
    source = Path(engine.__file__).read_text(encoding="utf-8")
    used = set(re.findall(r"\bAction\.([A-Z_]+)\b", source))
    assert used, "没扫到任何 Action 引用，正则可能失效了"
    missing = sorted(name for name in used if not hasattr(keys.Action, name))
    assert not missing, f"engine 引用了 keys.Action 里不存在的名字：{missing}"
