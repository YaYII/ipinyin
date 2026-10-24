#!/usr/bin/env python3
"""下载「普通话-粤语」平行语料（HuggingFace indiejoseph/yue-zh-translation，169,949 句对）。

用法：python3 tools/fetch_yue_pairs.py [输出路径]
输出：默认 vendor/yue_pairs.json（约 20MB）
依赖：pyarrow（pip install pyarrow）
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = ("datasets/indiejoseph/yue-zh-translation/"
        "resolve/main/data/train-00000-of-00001-201d2bed27e30c1d.parquet")
# 大陆网络下 huggingface.co 直连会超时（实测 25s timeout），hf-mirror.com 可用。
# 顺序：HF_ENDPOINT 环境变量 → 官方 → 镜像
HOSTS = [os.environ.get("HF_ENDPOINT", "").rstrip("/")] if os.environ.get("HF_ENDPOINT") else []
HOSTS += ["https://huggingface.co", "https://hf-mirror.com"]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "vendor", "yue_pairs.json")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("需要 pyarrow：pip install pyarrow", file=sys.stderr)
        return 2

    cache = os.path.join(ROOT, "vendor", "yue_train.parquet")
    if not os.path.exists(cache):
        print("下载 parquet（约 12MB）…")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        last: Exception | None = None
        for host in HOSTS:
            url = f"{host}/{PATH}"
            try:
                print(f"  尝试 {host} …", flush=True)
                urllib.request.urlretrieve(url, cache)
                last = None
                break
            except Exception as exc:              # 网络/超时/404，换下一个源
                last = exc
                print(f"  失败（{exc}），换源", file=sys.stderr)
        if last is not None:
            print("所有源都下载失败，可手动下载后放到", cache, file=sys.stderr)
            return 3

    table = pq.read_table(cache)
    pairs = [{"zh": item["zh"], "yue": item["yue"]}
             for item in table.column("translation").to_pylist()
             if item and item.get("zh") and item.get("yue")]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, ensure_ascii=False)
    print(f"{len(pairs)} 句对 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
