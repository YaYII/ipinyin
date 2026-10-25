"""句尾语气词表：片段边缘的语气词兜底转换。

背景（实测）：7 字单位下，语气词常独占片段尾（「吗？」「走吧。」），
神经给的粤语写法（咩/啦）会因缺邻字语境实证被守卫拦回（→嗎？/吧。）。
而句尾语气词的粤语写法几乎是固定的——查表比模型可靠。

这是**后处理兜底**，不推翻守卫：只有「源片段以表中语气词结尾、且输出里
它原样保留（没转过去）」时才替换；守卫/模型已经转好的（啦/咩）不会再动。
只收基本无歧义的句尾用法，宁可少收；更多映射走 --glossary 强制条目。
"""
from __future__ import annotations

# 源语气字（简/繁）→ 粤语书面写法。字形本身简繁同形，直接替换即可。
PARTICLES: dict[str, str] = {
    "吗": "咩", "嗎": "咩",      # 疑问：好吗？→好咩？
    "吧": "啦",                  # 祈使/商量：走吧→走啦
    "呗": "囉", "唄": "囉",      # 无所谓/随意：炒炒卖卖呗→…囉
}

# 守卫修复后的输出可能残留繁体字形（嗎/唄），匹配时也算「没转过去」。
_TRAD_OF = {"吗": "嗎", "呗": "唄"}


def _last_cjk_index(s: str) -> int:
    for i in range(len(s) - 1, -1, -1):
        if "\u4e00" <= s[i] <= "\u9fff":
            return i
    return -1


def fix_final_particle(src: str, dst: str) -> tuple[str, bool]:
    """src 尾部语气词在 dst 里原样保留时按表转换，返回 (新dst, 是否修过)。"""
    i = _last_cjk_index(src)
    if i < 0:
        return dst, False
    ch = src[i]
    want = PARTICLES.get(ch)
    if want is None:
        return dst, False
    j = _last_cjk_index(dst)
    if j < 0 or dst[j] not in (ch, _TRAD_OF.get(ch, ch)):   # 已经转过去了，不动
        return dst, False
    return dst[:j] + want + dst[j + 1:], True
