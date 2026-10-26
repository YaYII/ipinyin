#!/usr/bin/env python3
"""Suite ③：简中整句评测（zh-sentence）。

口径与 tools/eval_sentence.py 完全一致（直接复用其题面构造与判分）：
  - 题面：held-out 中文句 → 词库最长匹配换拼音；「自洽」过滤（切出的每个词
    必须是该拼音键的首选词），题面不合法的句子剔除并计数（invalid）
  - 桶：2-4字 / 5-7字 / 8-15字 / 16+字
  - 每桶报：有效题数、整句 Top1、前9、平均键数
  - 传 --lm 时加跑「不重排」对照：救回 / 弄坏 分开报（不合并成净分了事）

无 torch 依赖；lexicon / 语料缺失时返回 skipped。
题面构造集中在 select_questions()：内部评测、export_questions、score_submission
三方同源，改口径只动这一处。
"""
from __future__ import annotations

from collections import OrderedDict

from bench_common import DEFAULT_PAIRS, file_ok, fingerprint, repo, r4, question_hash, wilson

SUITE = "zh-sentence"
SUITE_VERSION = "1.1.0"

EVAL_BUCKETS = ("2-4字", "5-7字", "8-15字", "16+字")
QUICK_CAP = 50
FULL_CAP = 300
SCAN_LIMIT = 20000


def bucket_of(sentence: str) -> str:
    n = len(sentence)
    if n <= 4:
        return "2-4字"
    if n <= 7:
        return "5-7字"
    if n <= 15:
        return "8-15字"
    return "16+字"


def select_questions(params: dict):
    """题面构造（run / export / score 三方同源）。

    返回 (valid, invalid, scanned)：
      valid   = {bucket: [(key, sentence), ...]}  自洽过滤后、按 cap 截断的题面
      invalid = {bucket: n}                       被自洽过滤剔除的题数
      scanned = int                               题库扫描的句子总数
    口径：load_sentences 的 seed=0 洗牌 → 按桶 cap 截断原始句 → 逐句
    to_pinyin（带自洽检查），不合法的进 invalid。
    """
    from eval_sentence import build_word_map, load_sentences, to_pinyin
    from ipinyin.lexicon import Lexicon

    quick = params["quick"]
    cap = QUICK_CAP if quick else int(params.get("n") or FULL_CAP)
    pairs_path = params.get("pairs") or DEFAULT_PAIRS
    lexicon_path = params.get("lexicon") or repo("data", "lexicon.tsv")

    lexicon = Lexicon.load(lexicon_path)
    word_map = build_word_map(lexicon)
    sentences = load_sentences(pairs_path, int(params.get("scan") or SCAN_LIMIT))

    raw: "OrderedDict[str, list[str]]" = OrderedDict((k, []) for k in EVAL_BUCKETS)
    for s in sentences:
        key = bucket_of(s)
        if len(raw[key]) < cap:
            raw[key].append(s)
        if all(len(v) >= cap for v in raw.values()):
            break

    valid: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    invalid: "OrderedDict[str, int]" = OrderedDict()
    for name, items in raw.items():
        ok: list[tuple[str, str]] = []
        for s in items:
            r = to_pinyin(s, word_map, lexicon=lexicon)
            if r:
                ok.append((r[0], s))
        valid[name] = ok
        invalid[name] = len(items) - len(ok)
    return valid, invalid, len(sentences)


def run(params: dict) -> tuple[str, dict | None, str | None]:
    quick = params["quick"]
    pairs_path = params.get("pairs") or DEFAULT_PAIRS
    lexicon_path = params.get("lexicon") or repo("data", "lexicon.tsv")

    if not file_ok(pairs_path):
        return "skipped", None, f"语料不存在：{pairs_path}（先跑 `make corpus`）"
    if not file_ok(lexicon_path):
        return "skipped", None, f"词库不存在：{lexicon_path}（先跑 `make lexicon`）"

    from ipinyin.ime import ImeState
    from ipinyin.lexicon import Lexicon

    ime = ImeState(Lexicon.load(lexicon_path))

    lm, lm_path = None, params.get("lm")
    if lm_path:
        from ipinyin.lm import CharLM
        lm = CharLM.load(lm_path)
        if lm is None:
            return "error", None, f"语言模型加载失败：{lm_path}"
        ime.attach_lm(lm, float(params.get("lm_weight") or 0.6))

    valid, invalid, scanned = select_questions(params)

    cand_fn = params.get("candidate_fn")     # 测试/外部系统注入：key -> 候选词列表

    def leg(items: list[tuple[str, str]], use_lm: bool) -> dict[str, tuple[bool, bool, int, str]]:
        saved = ime.lm
        ime.attach_lm(lm if use_lm else None, float(params.get("lm_weight") or 0.6))
        out: dict[str, tuple[bool, bool, int, str]] = {}
        for key, s in items:
            if cand_fn is not None:
                words = list(cand_fn(key))
            else:
                ime.clear()
                for ch in key:
                    ime.insert(ch)
                words = ime.view().words
            out[s] = (words[:1] == [s], s in words[:9], len(key), key)
        ime.attach_lm(saved, float(params.get("lm_weight") or 0.6))
        return out

    out_buckets: "OrderedDict[str, dict]" = OrderedDict()
    empty: list[str] = []
    per_item: list[dict] = []
    questions: list[dict] = []
    inputs: list[dict] = []
    for p in (pairs_path, lexicon_path, lm_path):
        fp = fingerprint(p)
        if fp:
            inputs.append(fp)
    tot = {"n": 0, "top1": 0, "top9": 0}
    rr_tot = {"saved": 0, "broken": 0, "n": 0, "base_top1": 0, "cur_top1": 0}
    for name, items in valid.items():
        if not items:
            empty.append(name)
            continue
        base = leg(items, use_lm=False) if lm is not None else {}
        cur = leg(items, use_lm=True)
        n_valid = len(cur)
        n1 = n9 = keys = 0
        for i, (key, s) in enumerate(items):
            top1_ok, top9_ok, klen, _ = cur[s]
            per_item.append({
                "id": f"{name}:{i}", "bucket": name, "key": key,
                "expected": s, "top1_ok": top1_ok, "top9_ok": top9_ok,
                "keystrokes": klen,
                "base_top1_ok": (base[s][0] if lm is not None else None),
            })
            questions.append({"id": f"{name}:{i}", "bucket": name, "key": key})
            n1 += top1_ok
            n9 += top9_ok
            keys += klen
        entry = {
            "n": n_valid,
            "invalid": invalid[name],            # 题面不合法被剔除的题数（透明度）
            "top1": r4(n1 / n_valid),
            "top9": r4(n9 / n_valid),
            "avg_keystrokes": r4(keys / n_valid),
            "wilson": {"top1": wilson(n1, n_valid), "top9": wilson(n9, n_valid)},
        }
        if lm is not None:
            fixed = sum(1 for s in cur if not base[s][0] and cur[s][0])
            broken = sum(1 for s in cur if base[s][0] and not cur[s][0])
            entry["fixed"] = fixed               # 本来错、重排救回
            entry["broken"] = broken             # 本来对、重排弄坏
            rr_tot["saved"] += fixed
            rr_tot["broken"] += broken
            rr_tot["n"] += n_valid
            rr_tot["base_top1"] += sum(base[s][0] for s in cur)
            rr_tot["cur_top1"] += n1
        out_buckets[name] = entry
        tot["n"] += n_valid
        tot["top1"] += n1
        tot["top9"] += n9

    aggregate = {
        "n": tot["n"],
        "top1": r4(tot["top1"] / tot["n"]) if tot["n"] else None,
        "top9": r4(tot["top9"] / tot["n"]) if tot["n"] else None,
    }
    rerank = None
    if lm is not None and rr_tot["n"]:
        rerank = {
            "lm": lm_path,
            "lm_weight": float(params.get("lm_weight") or 0.6),
            "top1_no_lm": r4(rr_tot["base_top1"] / rr_tot["n"]),
            "top1_lm": r4(rr_tot["cur_top1"] / rr_tot["n"]),
            "saved": rr_tot["saved"],
            "broken": rr_tot["broken"],
            "net": rr_tot["saved"] - rr_tot["broken"],
        }

    results = {
        "buckets": out_buckets,
        "aggregate": aggregate,
        "rerank": rerank,
        "empty_buckets": [k for k in valid if not valid[k]],
        "questions_scanned": scanned,
        "inputs": inputs,
        "question_set_hash": question_hash(questions),
        "per_item": per_item,
    }
    return "ok", results, None
