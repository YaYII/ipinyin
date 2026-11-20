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


def test_no_accidental_self_recursion():
    """守护：这些方法不该调用自己。

    踩过的坑：替换函数体时把 _start_voice 的正文删掉，只剩 self._start_voice()，
    F8 一按就 RecursionError。
    """
    import inspect

    for name in ("_start_voice", "_toggle_voice", "_speak_sentence", "_commit_stage"):
        fn = getattr(engine.PinyinEngine, name, None)
        if fn is None:
            continue
        body = inspect.getsource(fn).split("\n", 1)[1]
        assert f"self.{name}(" not in body, f"{name} 调用了自己，会无限递归"
