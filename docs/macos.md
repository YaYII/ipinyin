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

## 激活（重要结论，2026-10-25 实测 macOS 26.6）

**程序化注册只对「Developer ID + 公证」的 app 生效**，其余被静默拒绝：

- `TISRegisterInputSource` 对未公证 app 返回 noErr 但什么都不做（用户/roots、
  ad-hoc/开发者证书签名、系统级目录、重启 inputswitchd、注销重登、真重启全试过）；
- `spctl --add` 手动放行已被 Apple 移除（"no longer supported"）；
- **对照实验**：装官方公证的 Squirrel 1.1.2（同样的 /Library/Input Methods、
  同样的 root 注册流程），3 个输入源**注销前立即注册成功**；同流程的 IPinyin
  （plist 逐键同构）始终无效。唯一变量是公证。

所以在本机的激活路径只有两条：

1. **公证**：加入 Apple Developer Program（$99/年）后 `xcrun notarytool submit`
   公证 IPinyin.app，注册即通（这是长期正解）；
2. **打字条**（无需公证）：见下一节。

（系统设置「+」面板枚举的也是 TIS 注册表，未公证 app 同样不出现，已实测。）

## 打字条：不公证也能在任意 App 打字

Apple 只拦「系统输入源」注册，不拦本地 app 往前台应用注入文字
（Espanso / CapslockWriter 同款思路）：

```bash
make macos-bar        # 启动打字条（常驻后台）
```

- 首次启动系统会弹「辅助功能」授权：系统设置 → 隐私与安全性 → 辅助功能 →
  打开 IPinyin，然后重新 `make macos-bar` 一次；
- 任意 App 光标处按 **⌃⌥空格** 唤出悬浮输入条 → 打拼音 → 空格/数字上屏，
  文字直接打进那个 App 的光标处（CGEvent 注入，剪贴板不动）；
- Esc 关闭；空缓冲时按空格/回车会原样转给前台应用；
- 日志：`~/Library/Application Support/ipinyin/bar.log`。

与系统输入法相比少了「光标跟随」（输入条固定在屏幕上方）和直接聚焦输入，
但打字体验（整句切分、简繁粤、个人化学习）完全一致 —— 同一条引擎链路。

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
