"""小鹤双拼编解码测试。"""
from ipinyin import shuangpin as sp


def test_encode_common_syllables():
    assert sp.encode("zhong") == "vs"
    assert sp.encode("shi") == "ui"
    assert sp.encode("hao") == "hc"
    assert sp.encode("wo") == "wo"
    assert sp.encode("men") == "mf"


def test_decode_round_trip():
    for syllable in ("zhong", "shi", "hao", "an", "ai", "ni"):
        code = sp.encode(syllable)
        assert sp.decode(code), f"{syllable} → {code} 解不回来"


def test_to_full_phrase():
    assert sp.to_full("womf") == "women"
    assert sp.to_full("vsui") == "zhongshi"
    assert sp.to_full("hc") == "hao"
