# 开发模式：直接用仓库跑输入法

不要每次改代码都打 deb。三条命令搞定：

```bash
bin/ipinyin-dev            # 切到仓库引擎
bin/ipinyin-dev --status   # 看现在跑的是仓库版还是 deb 版
bin/ipinyin-dev --restore  # 切回 deb 版
```

或 `make dev` / `make dev-status` / `make dev-restore`。

## 为什么不能直接改 /usr

`/usr/bin/ipinyin-engine` 里硬编码了 `sys.path.insert(0, "/usr/lib/ipinyin")`，
系统引擎永远读 `/usr/lib/ipinyin`。想让 IBus 跑仓库代码，只能让它启动另一个组件。

## 三步原理（踩过的坑）

1. **组件目录**：ibus-daemon 按 `$IBUS_COMPONENT_PATH` 找组件 XML。本机 systemd 单元
   `org.freedesktop.IBus.session.GNOME.service` 的 drop-in 已经把它设成
   `/usr/share/ibus/component:~/.local/share/ipinyin/component`，
   所以仓库组件必须放 **`~/.local/share/ipinyin/component/ipinyin.xml`**。
   （放 `~/.config/ibus/component/` 不生效——那个目录不在搜索路径里。）

2. **引擎名不能撞**：`/usr/share` 里的 deb 组件先被读到，同名 `ipinyin` 抢不过它，
   仓库这份必须叫 **`ipinyin-dev`**。

3. **输入源跟着改**：`gsettings org.gnome.desktop.input-sources sources`
   要指向 `('ibus','ipinyin-dev')`，否则 gnome-shell 会把你切回系统引擎。

改完组件还要 **删 `~/.cache/ibus/bus/registry`** 再重启 daemon，否则读的是旧缓存。

## 改完代码怎么生效

```bash
bin/ipinyin-restart     # 杀干净引擎，IBus 重新拉起（跑的还是仓库代码）
```

不用重装、不用重启桌面。日志在 `~/.cache/ipinyin/engine.log`。

## 验证真的在用仓库代码

```bash
bin/ipinyin-dev --status
# 引擎：仓库 → python3 /…/in_pinyin/bin/ipinyin-engine --ibus
```

只要路径里有 `in_pinyin/bin`，就是仓库版。
