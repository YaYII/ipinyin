# IPinyin on macOS

Linux 的 IBus 引擎**原样复用**（Python 纯标准库，/usr/bin/python3 直接跑），
外面套两层 Mac 的壳：

```
应用输入框
  └─ InputMethodKit 外壳（Swift，tools/macos/*.swift）
       └─ JSON 桥（unix socket，tools/macos_server.py）
            └─ ImeState / Lexicon / user_model / 粤语规则+守卫   ← 与 Linux 完全同一份代码
```

按键语义与 Linux 版逐条对齐（复用 `keys.py` 的 translate）：整句切分、前缀补全、
模糊音、个人化学习（选中 weight=3 / 空格 weight=1 / 词对 / 连打合成词组）、
Tab 简繁粤、四列候选（简/繁/粤/粤拼）、守卫防误改。

## 构建

```bash
make macos        # 打包引擎 → 编译 IMKit 外壳 → 签名 → 装入 ~/Library/Input Methods → 注册
```

## 激活（一次性）

macOS 在**会话初始化时**才扫描 `~/Library/Input Methods`（TISRegisterInputSource
在较新系统上不可靠，Squirrel/fcitx5 也是这个流程）：

1. **注销并重新登录**（构建脚本已把 IPinyin 写进启用列表，重登后多半直接出现在
   菜单栏输入法菜单里；构建脚本也会尝试自动启用）；
2. 若没出现：系统设置 → 键盘 → 输入法 → 编辑 → 添加「中文，简体 → IPinyin 拼音」；
3. `Ctrl+Space`（或输入法菜单）切到 IPinyin，任意输入框直接打拼音。

## 不注销也能先试：测试垫

```bash
make macos-testpad    # 打开一个小窗口，按键直接打进真正的引擎（同一条桥接链路）
```

## 按键

与 [README](../README.md) 的用法表一致，两点差异：

- 中/英切换：**Shift 轻按**（350ms 内无其他按键）；
- 候选窗是 IMK 的原生单行 9 列面板，鼠标点选可用。

## 数据与调试

- 用户数据：`~/Library/Application Support/ipinyin/`（user_model.tsv / user_bigram.tsv，
  与 Linux 的 `~/.local/share/ipinyin/` 平级概念，删掉即恢复出厂）；
- 引擎日志：`~/Library/Application Support/ipinyin/engine.log`；
- 词库/粤语表/守卫随包内置于 `IPinyin.app/Contents/Resources/engine/data/`。

## 当前限制

- 粤语**规则 + 守卫**可用；神经模型（onnxruntime）暂不随包 —— 把
  `yue_nn.onnx` + `yue_nn_vocab.json` 拷进 Resources/engine/data 即可启用（需 pip 装 onnxruntime）；
- 键盘映射按 ASCII 字符直取（`event.charactersIgnoringModifiers`），非 ASCII 布局的
  字符键未适配；双拼/笔画同理未适配；
- 简中字级 LM 重排：包内 data 含 `lm_char.marshal` 时自动挂（`make lm` 生成的不入库，
  本机构建会带上本机的那份）。
