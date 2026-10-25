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
- **粤语转换三层**：语序规则（我走先/畀本书你）→ 人工词表 → 用语料统计出的短语表；
  基准分数见 `tools/eval_cantonese.py`（12,600 句对：整句 33.1%，字符重合 73.1%）
- **粤语神经模型 + 约束解码**：装上 ONNX 模型后口语词大幅提升（不懂→唔識），
  输出过一道「实证守卫」—— 语料里没出现过的改法（亡羊补牢→亡羊補監）回退规则版，
  成语/古诗不再被静默改坏；加权分 66.2% → 83.3%，见 [docs/nn-upgrade.md](docs/nn-upgrade.md)
- **四列对照 + 粤拼注音**：候选固定四列 `简体  繁体  粤语  粤拼`，没有对应写法就用 `—` 占位
  （`你  —  —  nei5`、`是  —  係  hai6`、`我们  我們  我哋  ngo5 dei6`），
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

```bash
python3 tools/eval_quality.py            # 词级：给拼音，词能不能一次出来
python3 tools/eval_quality.py --log      # 同上，但用真机输入日志当题库（最真实）
python3 tools/eval_sentence.py           # 整句：给一串拼音，第一候选是不是你想打的那句
make eval-docs                           # 拿你自己的简体文档端到端测（4~8 字）
make lm                                  # 训练字级语言模型（整句重排用，约 3MB，不入库）
```

| 指标 | 结果 | 说明 |
| --- | --- | --- |
| 单词 Top1（一次就对） | 100%（60 题） | 自带语料只有 10 句、词全高频，**参考价值有限**；换成 `--log` 用真机日志才准 |
| 整句 Top1（通用语料） | 81.0%（315 题） | 挂了字级语言模型重排；不挂是 75.6% |
| **整句 Top1（你的文档，≤8 字）** | **93.3%（1997 题）** | 不重排 91.9%；2-4字 95.0%、5-8字 92.4% |
| 整句前 9 | 87.9% | 整句给 3 条备选之后从 75.6% 提到 87.9% |

重排权重 λ 越小越安全：λ=0.2 时救回 54 条 / 弄坏 20 条，λ=0.8 变成 92 / 88。
默认 0.4 是「通用 81.0%、你的文档 93.3%」的折中，想更稳用 `--weight 0.3`。

语料只吃简体：繁体是显示层的字符映射（`Converter`），粤语才需要模型，
两者都不该混进简中语言模型的训练语料（详见 [docs/nn-chinese.md](docs/nn-chinese.md)）。

> 旧版 README 挂的「77% / 95% / 99.5% / 37.5%」在四列显示上线后就无法复现了：
> 当时的脚本拿四列展示串 `"是否  —  —"` 跟目标词比较，永远不相等（实测 0.0%）。现已修正。

粤语神经模型的实测分数、已知限制和升级路线见 [docs/nn-upgrade.md](docs/nn-upgrade.md)，
训练步骤见 [docs/train-yue.md](docs/train-yue.md)（Mac 上 `./tools/dev_macos.sh --with-nn` 一键铺环境）。
简中这条链路的现状、整句基准和「神经网络该放在哪」见 [docs/nn-chinese.md](docs/nn-chinese.md)，
整句基准跑 `make eval-sentence`。

## 推到 GitHub

```bash
make push          # 等价于 python3 tools/gh_push.py HEAD master
```

大陆网络下 `github.com:443` 直连不通（`curl 28` 超时），而 `api.github.com` 可直连，
所以 `make push` 走 GitHub 的 **git-data API**：本地算出 blob/tree/commit 的 sha，
经 API 创建远端对象，再快进 `refs/heads/master` —— 内容与 `git push` 完全等价。
凭据取自 macOS 钥匙串里的 `github.com`（个人访问令牌），不落在仓库里。

> 别用 `git push https://gh-proxy.com/https://github.com/...` 这类镜像 URL：
> 镜像会剥离 `Authorization` 头，结果是 `No anonymous write access`。
> 镜像只适合 `git fetch`（拉取不需认证）。

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
