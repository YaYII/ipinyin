"""路径与运行参数。"""
from __future__ import annotations

import os

APP_NAME = "ipinyin"
ENGINE_NAME = "ipinyin"


def data_home() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


def cache_home() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, APP_NAME)


def lexicon_path() -> str:
    """按优先级查找词库：环境变量 → 安装目录 → 代码仓库（开发时）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(os.path.dirname(os.path.dirname(here)), "data", "lexicon.tsv")
    candidates = [
        os.environ.get("IPINYIN_LEXICON"),
        os.path.join(data_home(), "data", "lexicon.tsv"),
        "/usr/share/ipinyin/data/lexicon.tsv",   # deb 安装位置
        repo,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("找不到词库 lexicon.tsv，请先运行 install.sh")


def user_model_path() -> str:
    return os.path.join(data_home(), "data", "user_model.tsv")


SYSTEM_DATA = "/usr/share/ipinyin/data"


def _find_data(name: str) -> str:
    """优先用户目录（可覆盖），其次系统目录（deb 安装），最后返回用户路径。"""
    user = os.path.join(data_home(), "data", name)
    if os.path.exists(user):
        return user
    system = os.path.join(SYSTEM_DATA, name)
    if os.path.exists(system):
        return system
    return user


def input_mode_path() -> str:
    return os.path.join(data_home(), "input_mode")


def emoji_path() -> str:
    return _find_data("emoji.tsv")


def initials_path() -> str:
    return _find_data("initials.marshal")


def yue_dict_path() -> str:
    return _find_data("yue.marshal")


def yue_path() -> str:
    return _find_data("yue.tsv")


def yue_guard_path() -> str:
    """神经粤语解码的守卫数据（tools/build_yue_guard.py 生成）。"""
    return _find_data("yue_guard.marshal")


def s2t_path() -> str:
    return _find_data("s2t.marshal")


def user_bigram_path() -> str:
    return os.path.join(data_home(), "data", "user_bigram.tsv")


def user_bigram_seed_path() -> str:
    return _find_data("user_bigram_seed.tsv")


def user_seed_path() -> str:
    return _find_data("user_seed.tsv")


def mode_path() -> str:
    return os.path.join(data_home(), "mode")


def log_path() -> str:
    return os.path.join(cache_home(), "engine.log")
