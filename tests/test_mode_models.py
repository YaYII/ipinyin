"""个人词频按模式分家：粤语/繁体的命中不能污染简体候选。"""
import os

import pytest

from ipinyin.ime import ImeState
from ipinyin.lexicon import Lexicon
from ipinyin.t2s import Converter
from ipinyin.user_model import ModeModels, UserModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "data", "lexicon.tsv")
S2T = os.path.join(ROOT, "data", "s2t.marshal")


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load(LEXICON)


@pytest.fixture()
def models(tmp_path):
    return ModeModels({
        mode: UserModel(str(tmp_path / f"{mode}.tsv"))
        for mode in ModeModels.MODES
    })


def test_learning_stays_in_its_mode(models):
    """粤语里选「唔啱」，简体那份不能被记上。"""
    models.for_mode("yue").learn("budui", "唔啱", 3)
    assert models.for_mode("yue").picks("budui", "唔啱") == 3
    assert models.for_mode("simp").picks("budui", "唔啱") == 0
    assert models.for_mode("trad").picks("budui", "唔啱") == 0


def test_ime_uses_model_of_current_mode(lexicon, models):
    ime = ImeState(lexicon, user_model=models)
    assert ime.user_model is models.for_mode("simp")
    ime.mode = "yue"
    assert ime.user_model is models.for_mode("yue")
    ime.mode = "trad"
    assert ime.user_model is models.for_mode("trad")


def test_traditional_commit_learns_simplified_form(lexicon, models):
    """繁体只是显示层：选「聲音」要记成「声音」，否则简体模式会冒出繁体字。"""
    converter = Converter.load(S2T)
    ime = ImeState(lexicon, user_model=models, converter=converter)
    assert ime.base_form("聲音") == "声音"
    assert ime.base_form("繁體中文") == "繁体中文"
    # 粤语词没有简体形，原样保留
    assert ime.base_form("嘅唔啱") == "嘅唔啱"
    # 本来就是简体：不动
    assert ime.base_form("声音") == "声音"


def test_without_converter_base_form_is_identity(lexicon, models):
    ime = ImeState(lexicon, user_model=models)
    assert ime.base_form("聲音") == "聲音"


def test_single_model_is_wrapped_as_simp(lexicon, tmp_path):
    """老调用方只传一个 UserModel：当成简体那份，不炸。"""
    model = UserModel(str(tmp_path / "one.tsv"))
    ime = ImeState(lexicon, user_model=model)
    assert ime.user_model is model
    ime.mode = "yue"
    assert ime.user_model is model


def test_migration_targets_do_not_collide(tmp_path, monkeypatch):
    """词表/词对的分家文件名不能撞车——踩过：user_model.trad.tsv 里
    没有 `user_model.tsv` 这个子串，replace 静默失效，词对被写进了词表。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "split_user_model", os.path.join(ROOT, "tools", "split_user_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = [os.path.basename(module.target_for("user_model.tsv", mode))
             for mode in ModeModels.MODES]
    assert names == ["user_model.tsv", "user_model.trad.tsv", "user_model.yue.tsv"]
    names = [os.path.basename(module.target_for("user_bigram.tsv", mode))
             for mode in ModeModels.MODES]
    assert names == ["user_bigram.tsv", "user_bigram.trad.tsv", "user_bigram.yue.tsv"]
