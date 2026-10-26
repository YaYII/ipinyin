# IPinyin 安装说明（Ubuntu / Debian）

## 1. 安装

```bash
sudo dpkg -i ipinyin_x.y.z_all.deb
# 如果提示缺依赖：
sudo apt -f install
```

粤语神经模型**开箱即用**：onnxruntime 运行时与模型都已打进包里，
唯一依赖 `python3-numpy` 由 apt 自动安装。不需要联网、不需要 pip。

依赖：`python3`、`python3-gi`、`gir1.2-ibus-1.0`、`ibus`（Ubuntu 桌面默认都有）。

## 2. 注册到你的账号（不要用 sudo）

```bash
ipinyin-setup
```

它会：把 `IPinyin 拼音` 置顶为默认输入源、加入 IBus 预加载、重启 IBus 并切过去。

## 3. 使用

- `Super+Space` 切换到「IPinyin 拼音」
- 打字：字母 → `1`~`9` 选词，空格上屏首选，回车原样上屏，`[` / `]`、`{` / `}`、`-` / `=`、`PgUp` / `PgDn` 翻页
- **`←` `→` 切拼音段**：长拼音不用删字母，按 ← 可以只转前几个音节（nihaoma → nihao → ni）
- **简/繁/粤切换：Tab**（候选窗显示 `我们  我們  我哋  ngo5 dei6`，粤语模式带粤拼注音）
- **中/英切换：轻按一下 Shift**（英文模式下所有按键原样交给应用，打 abc 就是 abc）
- 不像拼音的串（abc / hello / zzz）不会再被硬凑成中文，空格原样上屏
- 首次在某个应用里可能需要在输入源里选一次；若切换列表里没有，**注销重登一次**

## 4. 卸载

```bash
ipinyin-setup --remove
sudo dpkg -r ipinyin
sudo dpkg -P ipinyin     # 连配置一起删
```

## 5. 常见问题

| 现象 | 处理 |
| --- | --- |
| 切换列表里没有「IPinyin 拼音」 | 注销重登；或 `ibus-daemon -drx` 后再看 |
| 某些应用打不出来 | Electron/Chromium 走 Wayland 需加 `--enable-wayland-ime`；或让它跑 XWayland（默认） |
| 想看日志 | `tail -f ~/.cache/ipinyin/engine.log` |
| 想加自己的词 | `python3 /usr/lib/ipinyin/tools/...` 不适用系统安装，直接编辑 `~/.local/share/ipinyin/data/user_model.tsv` 或联系维护者 |

数据都在本机：`~/.local/share/ipinyin/data/user_model.tsv`（你的选词与命中率），不上传。
