# 朗读（Stage 0：机器人音）

> 2026-11-18。按 `Ctrl+回车` 把当前候选念出来；没在打拼音时念刚上屏的那句。
> 粤语模式念粤语，简繁模式念普通话。

## 为什么先做机器人音

音色和感情属于「声学模型」的事，而**前端已经通了**：

```
候选文字（粤语写法）→ yue.marshal 词音优先 → 粤拼 → 音素（含 T1–T6 声调）
```

`projects/03-tts-g2p` 里这条链已经能产出 Piper 风格的音素序列，缺的只是
「音素 → 波形」。所以先用 espeak-ng 顶住，验证交互值不值得做；
真人音色（有感情、粤语）是 Stage 2，见文末。

## 用法

| 键 | 行为 |
| --- | --- |
| `Ctrl+回车` | 有候选：念当前高亮的候选（就是按空格会上屏的那个）；没候选：念刚上屏的那句 |
| `IPINYIN_TTS=0` | 环境变量整体关掉朗读 |

念的是**当前模式下会上屏的写法**：粤语模式念粤语（`我哋一齊去飲茶`），
简繁模式念普通话。所以切到粤语模式按一下，就是「我写的这句粤语怎么念」。

## 两个声线：粤语给汉字，普通话给拼音

这是实测踩出来的，不是设计出来的：

| 声线 | 输入 | 例 | 音素 |
| --- | --- | --- | --- |
| `yue` | **汉字**（简繁都认） | 嘅唔啱 | `g'e3  N4  N'aa1m` |
| `cmn-Latn-pinyin` | **带调号拼音** | shēng yīn | `s.'@55N  j'i55n` |

**坑**：espeak 的普通话声线 `cmn` 读汉字是**逐字拼英文字母**——
`们` → `m-e-n-f-a-i-v`、`音` → `y-i-n`、`嘅` → `k-a-i-t-r-i`，
听感完全是乱码。第一版就是拿 `cmn` 直接读汉字，粤语模式下念「係 / 嘅 / 唔啱」
全部变成字母串——这就是「声音播放得唔啱」的根因。

所以普通话走 `汉字 → 拼音 → cmn-Latn-pinyin`：
`data/chars.marshal`（20,925 字，`tools/build_char_pinyin.py` 生成，349KB）
提供「汉字 → 带调号拼音」，取第一个读音（最常用）。多音字会读错（长/行/重），
预览用途可接受；要更准得接分词 + 词级读音，那是另一件事。

粤语不用转：`yue` 声线自带完整字典（`yue_dict`），简体繁体都能读对。

## 两级后端：edge-tts 优先，espeak 兜底

espeak 的粤语六声调实测与调值不符（同一音节的 `si1` 该高平却下滑 50Hz、`si3` 该中平却上扬 42Hz），
听感就是「音素对、声调怪」。所以装上 edge-tts 就用神经声线，失败（断网/超时）自动回落 espeak，绝不没声音。

| 后端 | 声线 | 首次合成 | 缓存命中 | 联网 | 许可 |
| --- | --- | --- | --- | --- | --- |
| edge-tts | 粤语 `zh-HK-WanLungNeural`、普通話 `zh-CN-YunxiNeural` | 1.2–1.8s | 0.1ms | 要 | GPLv3（子进程） |
| espeak-ng | `yue` / `cmn-Latn-pinyin` | 37–42ms | 0.1ms | 不要 | GPLv3（子进程） |

装法（免 root；依赖落在项目私有目录，不碰系统 Python——Ubuntu 24.04 有 PEP 668，
`pip install --user` 会被拒）：

```bash
pip3 install --target ~/.local/share/ipinyin/lib edge-tts
```

装到项目私有目录（不碰系统 Python——Ubuntu 24.04 的 PEP 668 会拒绝 `pip install --user`），
`_find_edge()` 直接给这个目录注入 `sys.path`，不需要额外包装脚本，也不用改 `PATH`。

| 环境变量 | 作用 |
| --- | --- |
| `IPINYIN_TTS=0` | 整体关掉朗读 |
| `IPINYIN_TTS_EDGE=0` | 只用本地 espeak，不要联网声线 |

缓存键带 backend（`sha1(backend|voice|speed|text)`），所以断网后不会播到 edge 留下的缓存，
装上 edge 也不会一直播旧的机械音。

## 词级音库：离线、可逐条校对

edge-tts 是端到端模型，读哪个音由模型决定；要「每个音都标准、且不依赖别人的服务」，
就把常用词用 edge **预合成**成本地单元，朗读时按词拼接（20 句 A/B 对比，用户确认可接受）：

```
文本 → 最长匹配切词（词表 101,811）→ 词单元（Opus 24kbps）
     → 该词不在库：退单字
     → 单字也不在：整句回落 edge → espeak
```

| 项 | 数值 |
| --- | --- |
| 建库 | `python3 tools/build_tts_units.py`（默认 3000 词 + 1500 字） |
| 选词 | **语料词频**（`vendor/yue_pairs.json` 16.9 万句对的粤语侧），不是通用词频——通用词频只能覆盖日常句 11%，语料词频 78%，加单字兜底到 100% |
| 体积 | 约 4500 单元，Opus 24kbps，约 15–20MB |
| 校对 | 单元文件名 = 词哈希；发现哪个词读错，删掉那一个 `.opus` 重跑即可，不必重建 |
| 播放 | 解码后 PCM 缓存 400 条（约 17MB），词间 40ms 间隙 |
| 关掉 | `IPINYIN_TTS_UNITS=0`；目录可用 `IPINYIN_TTS_UNITS_DIR` 改 |

优先级：**edge（整句）→ units（离线兜底）→ espeak**。

为什么 edge 排前面：词级单元是单独录的，每个词自带句尾语调，拼起来会有词边界
（实测词间必须留 15ms 以上才不黏，但仍听得出停顿）。整句合成韵律自然，且结果
会进缓存——同一句第二次起离线也能播。要强制离线（不用云端），设 `IPINYIN_TTS_OFFLINE=1`。

音库是粤语男声，只对粤语声线生效；普通话仍走 edge（`zh-CN-YunxiNeural`）或 espeak 拼音。

上屏即读：`IPINYIN_TTS_AUTO=1`（默认关，打字时不吵）。

## 实现要点

| 关注点 | 做法 | 原因 |
| --- | --- | --- |
| 不阻塞 IBus | 合成 + 播放全在后台线程，`speak()` 只发指令（实测 0.6ms 返回） | 主循环卡一下，整机输入都卡 |
| 打断 | 新朗读先 `terminate()` 上一条 | 连按不能叠音 |
| 缓存 | `~/.cache/ipinyin/tts/<sha1>.wav`，LRU 200 条；命中 0.05ms | 同一句重播不该再合成 |
| 许可证 | espeak-ng 是 GPLv3，**只做子进程调用**，不链接 | 本项目保持 MIT |
| 降级 | 没有 espeak-ng / 播放器 → 记一条 warning，其余功能照常 | 朗读是加法，不能变成故障点 |
| 隐私 | 纯本地，不出网；密码框由 IBus 层决定是否聚焦，不额外读 | 朗读内容比翻译更敏感 |

实测（本机，i5，espeak-ng 1.51）：粤语 7 字 37ms、普通话 7 字 34ms / 14 字 42ms。
音质是电子词典级别——**用来确认「念出来是不是我想说的」够用，用来听感不够**。

## 装 espeak-ng

Ubuntu 只装 `libespeak-ng1`（speech-dispatcher 的依赖）时，**语音数据是齐的**
（`/usr/lib/x86_64-linux-gnu/espeak-ng-data/yue_dict` 已存在），缺的只是命令行二进制：

```bash
sudo apt install espeak-ng          # 正规做法
```

没有 root 时：把 `espeak-ng` 二进制放进 `~/.local/bin/` 即可，
`speak.py` 会去那里找（`PATH` 里没有也能用）。

验证粤语声在不在：

```bash
espeak-ng --voices | grep yue       # 应看到 yue 与 yue-Latn-jyutping
espeak-ng -v yue -x -q "我哋"       # 应输出 ŋɔː5 dei2 一类的音素
```

## Stage 2：真人粤语声

缺的是**音色 + 感情**，不是发音规则。路线：

1. 声学模型：VITS / Piper 训练，前端直接用 `projects/03-tts-g2p` 的音素（已含声调 token）。
2. 数据：WenetSpeech-Yue（21,800 小时，开源）+ Common Voice zh-HK。
3. 感情：先做「语速/停顿/句末语调」三个可控维度，够日常；情绪标签（开心/严肃）要标注语料，另算一个项目。
4. 分发：模型 50–150MB，deb 体积会跳一个量级——先做成可选下载，不随包。

顺带一提：Piper 至今没有粤语 voice（[issue #511](https://github.com/rhasspy/piper/issues/511) 开着），
训出来是可以单独发 release 的东西。

## 开与关：F9 切换，状态显示在预编辑上方

| 操作 | 效果 |
| --- | --- |
| `F9` | 切换「上屏即读」开/关；状态持久化到 `~/.local/share/ipinyin/tts_auto` |
| `Ctrl+回车` | 随时念一次（不论开关） |

预编辑上方的提示会带上状态：`简体 · 音:开` / `繁体 · 音:关` / `粤语 · 音:开`。
「音:开」= 每次上屏自动念；「音:关」= 只在按 `Ctrl+回车` 时念。

引擎日志也会记一行：`上屏即读 开启/关闭`，方便排查「到底开没开」。

## 自监控：异常自动记录，下次直接看数据

每次朗读都记一条 JSONL（`~/.cache/ipinyin/tts-metrics.jsonl`）：后端
（edge / units / espeak / cache）、合成耗时、音频时长、语速（字/秒）、样本（前两个字）。

异常按规则自动标记——规则都是踩过的坑，不靠人记：

| 规则 | 判定 | 对应的坑 |
| --- | --- | --- |
| `padding` | ≤3 字却 >1.5s | edge 把短文本补到 1.87s |
| `too-slow` / `too-fast` | 语速 <2 或 >9 字/秒 | 补静音没裁 / 合成吞字 |
| `slow-synth` | 单次合成 >8s | 网络抖动，用户觉得「按了没反应」 |
| `fallback-espeak` / `fallback-units` | 回落 | edge 不可用 |
| `empty-audio` | 音频为空 | 引擎或转码失败 |

体检命令：

```bash
python3 tools/tts_doctor.py          # 语速/耗时分布 + 异常清单 + 环境（输出设备/音库/后端）
python3 tools/tts_doctor.py --all    # 连上一轮（.1）一起看
python3 tools/tts_doctor.py --no-env # 只看记录
```

记录只留样本前两个字，**不存原文**（朗读内容比翻译更敏感）；文件超过 1MB 自动轮转成 `.1`。
缓存命中也会记（后端记为 `cache`），所以命中率一眼可见。

## 已知限制

- **数字/英文混读**：espeak 会把 `2026` 念成英文数字。`g2p.py` 里已有中文数字规范化（4 位年逐位读），
  接进朗读前先用它洗一遍。
- **多音字**：词级优先的 `yue.marshal` 解决了大半，人名/生僻词仍会读错——接受。
- **朗读与输入抢焦点**：朗读时继续打字会立刻停声（新朗读打断旧朗读），不排队。


## 三档朗读（2026-11-19）

`F9` 循环三档，状态条显示 `音:关` / `音:粤语` / `音:汕头`：

| 档 | 念什么 | 声线 |
|---|---|---|
| 关 | 不念（Ctrl+回车 仍可手念） | — |
| 粤语 | 粤语写法（汉字） | espeak `yue` / edge `zh-HK-WanLungNeural` |
| 汕头 | **潮拼**（如 `le2 ho2`） | espeak `yue-Latn-jyutping` |

**注意**：espeak-ng 没有潮州话声线（132 个声线里只有 cmn / yue），
汕头档是拿粤拼声线读潮拼，属于**近似**——声调数和韵母都不完全对得上。
要真声需要自己训 TTS：现成数据有 [Teochew-Wild](https://huggingface.co/datasets/panlr/teochew_wild)
（18.9 小时、正字+拼音标注）和 [BigVGAN 潮州话声码器](https://huggingface.co/panlr/BigVGAN_22khz_teochew)。
