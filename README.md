# IPinyin

一个自己写的 Linux 拼音输入法（IBus 引擎）。装在 IBus 上，任何应用的输入框都能用。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![IBus](https://img.shields.io/badge/ibus-1.5%2B-green.svg)](https://github.com/ibus/ibus)

```
woxihuanni  →  我喜欢你
zhihui      →  只会 / 指挥 / 智慧
abc         →  abc（不是"阿波罗"）
zhongguo    →  中国  中國      ← Tab 切 简体/繁体/粤语
```

## 特性

- **整句切分**：`woxihuanni` → 我喜欢你，一元语言模型 + 动态规划
- **前缀补全**：`zenm` → 怎么、`went` → 问题、`xin` → 新/心/信
- **组合补词**：词库没有的词按音节拼（只 + 会 → 只会）
- **手滑容忍**：`zhogn` → 中（相邻字母打反）、`zongguo` → 中国（zh/z 模糊音）
- **永不空白**：任何输入都有兜底候选，英文串原样上屏
- **四列对照 + 粤拼注音**：候选固定显示 `简体  繁体  粤语  粤拼`（`我们  我們  我哋  ngo5 dei6`），
  `Tab` 循环切换上屏形式，模式记忆；港澳用户可直接输出书面粤语并顺带学发音
- **左右切段**：`nihaoma` 按 `←` 缩到 `nihao` / `ni`，只转这一段，剩下的拼音保留
- **长拼音选单字**：翻页就能选单字，不用删字母
- **个人化**（全在本机）：命中率排序、词对预测、连打合成词组
- **快**：单键 p50 3ms / p99 10ms，UI 刷新合并到 15ms
- **稳**：崩溃 2 秒自愈，引擎位自动恢复

## 安装

### 分发给别人（.deb）

```bash
./packaging/build_deb.sh            # 生成 dist/ipinyin_x.y.z_all.deb
sudo dpkg -i dist/ipinyin_*_all.deb && sudo apt -f install   # 或从 Releases 下载
ipinyin-setup                       # 用自己账号跑，注册并置顶输入源
```

### 自己机器（用户目录，无需 root）

```bash
./tools/fetch_data.sh               # 下载词库数据（约 15MB）
python3 tools/build_lexicon.py
python3 tools/build_s2t.py
./install.sh
```

装完 `Super+Space` 切到「IPinyin 拼音」。

## 用法

| 键 | 作用 |
| --- | --- |
| 字母 | 输入拼音 |
| `1`–`9` | 选候选（长拼音里选单字时，剩余拼音保留） |
| 空格 | 上屏当前候选；不像拼音的串原样上屏 |
| 回车 | 原样上屏拼音 |
| `Tab` | 简体 → 繁体 → 粤语 → 简体（粤语模式带粤拼注音） |
| `Shift`（轻按） | 中/英切换 |
| `←` / `→` | 切换要转换的拼音段 |
| `↑` / `↓` | 移动候选高亮 |
| `[` `]`、`{` `}`、`-` `=`、`PgUp` `PgDn` | 翻页 |
| `Esc` | 清空 |

## 质量基准

用真实语料当考题（`python3 tools/eval_quality.py`）：

| 指标 | 结果 |
| --- | --- |
| 单词 Top1（一次就对） | 77% |
| 单词前三 | 95% |
| 单词第一页（≤9） | 99.5% |
| 整句一次命中 | 37.5%（同音字是主要瓶颈） |

## 结构

```
bin/ipinyin-engine        IBus 调起的入口
src/ipinyin/engine.py     IBus 适配：按键/预编辑/候选表/节流
src/ipinyin/ime.py        状态机（纯逻辑，可单测）
src/ipinyin/lexicon.py    词库（紧凑快照，124k 拼音键）
src/ipinyin/segmenter.py  整句切分（Viterbi）
src/ipinyin/user_model.py 命中率/词对/学习
src/ipinyin/t2s.py        简→繁映射
src/ipinyin/keys.py       键值 → 动作
tools/                    词库编译、质量基准、加词工具
packaging/                IBus 组件描述、deb 打包
```

## 词库

| 数据 | 内容 | 许可 |
| --- | --- | --- |
| pinyin-data | 单字读音 | MIT |
| phrase-pinyin-data | 41 万词组 | MIT |
| jieba dict | 词频 | MIT |
| CC-CEDICT | 12.5 万中英词条 | CC BY-SA 4.0 |
| OpenCC STCharacters | 简→繁 | Apache-2.0 |
| data/yue.tsv | 普→粤书面用字对照（本项目整理） | MIT |
| rime-cantonese | 粤语字音/词音（粤拼） | CC BY 4.0 |

合并后 12.4 万拼音键 / 19.4 万候选。详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 个人化

每次上屏记 `拼音 → 词 → 选中/展示`，排序分 = `先验 + 选中 + 2×命中率`；
连续两次上屏会合成词组（先「西」后「大」→ `xidai 西大`）；词对用于猜下一个词。
数据都在 `~/.local/share/ipinyin/`，不上传，删掉即恢复出厂。见 [docs/personalization.md](docs/personalization.md)。

## 参与

见 [CONTRIBUTING.md](CONTRIBUTING.md)。有词打不出来、排序不对，欢迎提 issue 并附上
「打了什么拼音、期望哪个词」。

## English

IPinyin is a lightweight Chinese pinyin IME for Linux, implemented as an IBus engine
in Python. It supports full-sentence segmentation, prefix/abbreviation completion,
fuzzy pinyin, per-user adaptive ranking (local only), and simplified/traditional
dual-column candidates toggled with `Tab`. MIT licensed; dictionary data from
open sources (see THIRD_PARTY.md).

## License

[MIT](LICENSE)
