#!/bin/sh
# 下载编译词库所需的开源数据（只在生成 data/lexicon.tsv 时需要）。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"
cd "$VENDOR"

fetch() {
    url=$1
    out=$2
    if [ -s "$out" ]; then
        echo "已存在 $out，跳过"
        return 0
    fi
    echo "下载 $url"
    curl -fsSL --max-time 180 -o "$out.part" "$url"
    mv "$out.part" "$out"
}

fetch https://raw.githubusercontent.com/mozillazg/pinyin-data/master/pinyin.txt pinyin.txt
fetch https://raw.githubusercontent.com/mozillazg/phrase-pinyin-data/master/large_pinyin.txt large_pinyin.txt
fetch https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt jieba.txt
echo "完成：$(ls -l "$VENDOR" | tail -n +2 | awk '{print $5, $9}' | tr '\n' ' ')"

# 粤语发音（rime-cantonese，CC BY 4.0）
fetch "https://raw.githubusercontent.com/rime/rime-cantonese/main/jyut6ping3.chars.dict.yaml" \
      "$ROOT/vendor/yue_chars.yaml"
fetch "https://raw.githubusercontent.com/rime/rime-cantonese/main/jyut6ping3.words.dict.yaml" \
      "$ROOT/vendor/yue_words.yaml"
python3 "$ROOT/tools/build_yue_dict.py"
