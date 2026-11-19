# 输入法记忆体：把上屏记录变成个人语料画像

> 个人自用功能。采集默认开，`IPINYIN_MEMORY=0` 可关。

## 它看得到什么、看不到什么

| 看得到 | 看不到 |
| --- | --- |
| 每次上屏的词、拼音键、输出模式 | 你在应用里上屏之后改了什么 |
| 选的是第几个候选（越过首选 = 首选不准） | 窗口标题、网页内容、剪贴板 |
| 两次上屏之间的间隔（输入节奏） | 密码框（IBus purpose 判定，直接跳过） |

所以「错别字」只能从**输入行为**推：首选是 A 你选了 C、删了重打、同一个拼音反复选不同词。
它看不到你最终发给别人的那句话，也不该假装看得到。

## 数据放在哪

```
~/.local/share/ipinyin/memory/
├── events.jsonl      每次上屏一行（时间/拼音/词/模式/候选排名/间隔）
├── cursor.json       已整理到哪个字节（增量，不重复统计）
├── profile.json      画像：上屏总数、高频词、拼音键、模式分布、时段、节奏
└── corrections.tsv   首选不准：拼音 → 你选的词 → 次数
```

超过 5MB 自动轮转成 `events.jsonl.1`；画像只留 Top 500 词，不会无限长。

## 为什么空闲才整理

整理要读全量事件做聚合。放在按键路径上会卡输入（实测聚合一次 20–60ms，打字能感觉到）。
所以用 GLib 定时器每 10 秒检查一次：**超过 60 秒没按键**且有待整理数据，才起后台线程聚合。
这就是「离开键盘时它在后台消化」的机制。

## 用法

```bash
ipinyin-memory status         # 采集了多少、待整理多少
ipinyin-memory show           # 高频词 Top 20
ipinyin-memory consolidate    # 不等空闲，立刻整理
ipinyin-memory export 我的画像.md   # 导出 Markdown，给别的 AI / 笔记软件读
ipinyin-memory wipe           # 清空全部
```

导出的 Markdown 分三块：高频词、领域词（你的输入里常见、通用语料里罕见）、首选不准。
这三块就是「你平时在写什么、输入法哪里不懂你」的浓缩。

## 已知限制

- **拿不到应用名**：IBus 引擎只收到按键和焦点事件，不知道焦点在哪个窗口（Wayland 下更拿不到）。
  想要「在写文档 / 在回邮件」这种粒度，得另接窗口标题采集，那是另一个决定。
- **错别字只能间接推**：见上表。
- **领域词判定靠通用词频对比**：`data/lexicon.tsv` 里 Top 3000 视为通用词，之外的才算领域词。

## 触角清单（`ipinyin-memory sensors`）

| 触角 | 来源 | 状态 |
| --- | --- | --- |
| 上屏词 / 拼音键 / 输出模式 | `do_process_key_event` | ✅ |
| 候选排名（越过首选 = 首选不准） | 数字选词 / 点击 | ✅ |
| 退格次数（打错 / 改主意） | Backspace | ✅ |
| 翻页次数（候选不够好） | PageUp/PageDown | ✅ |
| 决策耗时（首键 → 上屏） | 时间戳 | ✅ |
| 放弃（打到一半按 Esc） | Escape | ✅ |
| 输入节奏（两次上屏间隔） | 时间戳 | ✅ |
| 时段分布 | 时间戳 | ✅ |
| 焦点切换频率（多任务跳动） | `do_focus_in` | ✅ |
| 窗口标题 / 应用名 | — | ❌ GNOME Wayland 不提供 |
| 上屏后在应用里的修改 | — | ❌ IBus 只看自己上屏的 |
| 剪贴板 / 鼠标位置 | — | ❌ IBus 不提供 |

「拿不到」的三项不是没做，是平台限制：Wayland 下引擎拿不到焦点窗口信息，
XWayland 应用理论上可以用 `xdotool` 取到，但原生 Wayland 应用一律不可见。

## 屏幕感知（vision）

输入法自己截不了图：GNOME Wayland 下 `org.gnome.Shell.Screenshot` 直接返回
`AccessDenied: Screenshot is not allowed`，Screenshot portal 每次都要弹窗确认。

所以当前用「监看截图目录」：你按 PrintScreen，`ipinyin-vision` 发现新图后调用
可插拔的视觉模型命令，把「这一屏在做什么」写进 `vision.jsonl`。

```bash
ipinyin-vision status      # 监看哪些目录、模型命令配没配
ipinyin-vision scan        # 只列新截图
ipinyin-vision once        # 扫一次并调用模型
ipinyin-vision watch       # 每 20 秒扫一次
ipinyin-vision recent 10   # 最近 10 条上下文

# 模型命令（{image}/{prompt} 会被替换）——不配也行，默认走 opencode-go
export IPINYIN_VISION_CMD='opencode run --model deepseek-v4.1-flash -f {image} "{prompt}"'
```

### 默认模型：复用 opencode-go 订阅

不配置任何东西时，`describe()` 会走 opencode-go（OpenAI 兼容），
key 从 DSH 凭证 `~/.dsh/.credentials.yaml` 的 `OPENCODE_GO_API_KEY` 读取——
和 DSH 用的是同一个订阅，不需要另外申请。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `IPINYIN_VISION_BASE` | `https://opencode.ai/zen/go/v1` | 网关地址 |
| `IPINYIN_VISION_MODEL` | `deepseek-v4.1-flash` | 视觉模型 |
| `IPINYIN_VISION_SESSION` | `ses_ipinyin_vision` | 网关要求的路由头 `x-opencode-session` |
| `IPINYIN_VISION_CMD` | 空 | 自定义命令，优先于内置通道 |

实测（三张真实截图）：

```
财务/还款管理后台核对 2026年9月还款结果，MOP 666,116.00
中葡双语公文表单，日期栏已填入 05/10/2026
文档查看器中的标识符 20261007364/DAPE/FR P
```

默认监看 `~/Pictures/截图`、`~/Pictures/Screenshots` 和 `vision/inbox`，
可用 `IPINYIN_VISION_DIRS` 覆盖（`:` 分隔）。

### 帧的生命周期（自动清理）

处理截图时**先复制一份到 `vision/frames/`**，模型看的是副本：

- 你的原图（`~/Pictures/...`）**永远不删**
- 副本保留 **1 天**，`ipinyin-vision cleanup` 或 `watch` 循环每小时自动清
- 另装一个 systemd 定时器每天 00:05 兜底：`ipinyin-vision-cleanup.timer`
- 上下文日志 `vision.jsonl` 超过 2000 行只留最近的

```bash
ipinyin-vision cleanup        # 手动清（默认保留 1 天）
ipinyin-vision cleanup 7      # 保留 7 天
```

**真正的全自动**需要 ScreenCast portal 的一次性授权 + `restore_token`
（GNOME 42+ 支持恢复令牌，授权一次之后可静默重连）；或者切到 Xorg 会话，
X11 下 `import`/`xwd` 截屏无需任何授权。

## 全自动屏幕感知（ScreenCast，方案 A）

授权一次之后不再弹窗，常驻服务自己抓帧、自己理解。

| 服务 | 作用 | 周期 |
| --- | --- | --- |
| `ipinyin-screencast.service` | ScreenCast 授权 + 抓帧到 `vision/incoming/` | 20 秒 |
| `ipinyin-vision-watch.service` | 消费新帧 → 视觉模型 → `vision.jsonl` | 30 秒扫一次 |
| `ipinyin-vision-cleanup.timer` | 删 1 天前的帧副本 | 每天 00:05 |

### 双屏：你在哪块屏幕打字就抓哪块

1. 授权时勾选**两块屏幕**，`Start` 返回两条串流，各带 `position/size`
2. IBus 引擎的 `do_set_cursor_location` 把输入光标坐标写进 `vision/cursor.json`
3. 抓帧前按坐标判断落在哪块屏的矩形里，就抓那条串流

实测：左屏 `2560x1440 @ (0,474)`、右屏 `3840x2160 @ (2560,0)`，
光标移动后下一帧立即切换（日志里节点 78 ⇄ 77）。

### 界面上的状态

预编辑上方会显示三态：`简体 · 音:开 · AI:开`

- **音:开/关** = 上屏即读（F9 或 `ipinyin-tts on/off`）
- **AI:开/关** = 屏幕感知是否在跑（读 `vision/heartbeat.json`，3 秒缓存）
  - 抓帧服务每抓一帧写一次心跳；心跳超过 90 秒没更新就显示 `AI:关`

### 免重复授权

`SelectSources` 带 `persist_mode=2` 才会返回 `restore_token`（默认不返回）。
实测：第二次启动 10 秒内直接抓到帧，没有弹窗。

### 踩过的坑（都写进代码注释了）

| 坑 | 表现 | 正确做法 |
| --- | --- | --- |
| Response 回调少一个参数 | 永远收不到结果，看起来像「不弹窗」 | 回调要接 7 个参数（多一个 user_data） |
| `Start` 签名 | `(oa{sv})` 报 InvalidArgs | 是 `(osa{sv})`，中间要 `parent_window` |
| D-Bus `h` 类型 | 拿到 `fd: 0`，gst 报 not-negotiated | 用 `call_with_unix_fd_list_sync` + `UnixFDList` 取真 fd |
| 少了 `x-opencode-session` | 视觉接口报 MissingSessionID | 请求头必须带 |
