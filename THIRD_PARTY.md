# 第三方数据

代码本体是 MIT，但词库来自多个开源项目，各自许可见下。分发时请保留本文件。

| 数据 | 用途 | 许可 | 来源 |
| --- | --- | --- | --- |
| pinyin-data | 单字读音 | MIT | https://github.com/mozillazg/pinyin-data |
| phrase-pinyin-data | 词组读音（41 万条） | MIT | https://github.com/mozillazg/phrase-pinyin-data |
| jieba dict | 词频 | MIT | https://github.com/fxsjy/jieba |
| CC-CEDICT | 中英词典（12.5 万条） | CC BY-SA 4.0 | https://www.mdbg.net/chinese/dictionary?page=cc-cedict |
| OpenCC STCharacters | 简→繁映射 | Apache-2.0 | https://github.com/BYVoid/OpenCC |
| rime-cantonese | 粤语字音/词音（粤拼） | CC BY 4.0 | https://github.com/rime/rime-cantonese |
| 開放粵語字典 (kfcd/yyzd) | 粤语字音校验 | CC BY 3.0 | https://github.com/kfcd/yyzd |
| CanCLID/sentences | 粤语对话语料（仅统计高频用法，不转载原文） | 未标注许可，仅作词频统计 | https://github.com/CanCLID/sentences |
| CanCLID/awesome-cantonese-nlp | 资源索引 | — | https://github.com/CanCLID/awesome-cantonese-nlp |

`tools/fetch_data.sh` 会下载上述数据到 `vendor/`，`tools/build_lexicon.py` 编译成
`data/lexicon.tsv` 与 `data/lexicon.marshal`（不含在仓库里，按需生成）。
