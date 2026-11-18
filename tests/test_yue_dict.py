"""粤语字音表构建：多音字必须取主读音，不能盲取源文件第一行。"""
import importlib.util
import marshal
import pathlib
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_yue_dict", ROOT / "tools" / "build_yue_dict.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_pick_main_prefers_reading_without_percentage():
    """rime-cantonese 的主读音不带百分比，变体才带。"""
    # 去：第一行是 heoi2 3%（异读），主读音 heoi3 在第二行
    assert mod.pick_main([("heoi2", 3.0), ("heoi3", None)]) == "heoi3"


def test_pick_main_falls_back_to_highest_percentage():
    """没有主读音标记时取百分比最高的（阿：aa2 0%、aa3 5%）。"""
    assert mod.pick_main([("aa2", 0.0), ("aa3", 5.0)]) == "aa3"


def test_pick_main_single_reading():
    assert mod.pick_main([("m4", None)]) == "m4"


def test_shipped_dictionary_uses_main_reading():
    """随包分发的 yue.marshal 不能是旧构建（取错 1194 个多音字）的产物。"""
    raw = (ROOT / "data" / "yue.marshal").read_bytes()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    chars = marshal.loads(raw)["chars"]
    assert chars["去"] == "heoi3"
    assert chars["阿"] == "aa3"
    assert chars["雅"] == "ngaa5"
