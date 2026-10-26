# 参与开发

> 完整交接手册（仓库地图、引擎管线、评测口径纪律、模型迭代工作流、踩坑速查、
> 当前优先级）见 [docs/handover.md](docs/handover.md)；六个衍生项目见
> [projects/README.md](projects/README.md)。本文只讲最小上手。

## 跑起来

```bash
git clone https://github.com/YaYII/ipinyin.git && cd ipinyin
./tools/fetch_data.sh          # 下载词库数据
python3 tools/build_lexicon.py # 编译词库
python3 tools/build_s2t.py     # 编译简繁映射
python3 -m pytest tests -q     # 单测
./install.sh                   # 装到 ~/.local/share/ipinyin
```

## 改代码的约定

- 逻辑都在 `src/ipinyin/`：`ime.py` 是纯状态机（不依赖 IBus，可单测），
  `engine.py` 只做 IBus 适配，改行为优先改 `ime.py`。
- 所有新行为都要有单测；`tests/e2e_ibus.py` 走真实 IBus 总线，桌面有人打字时会互相干扰。
- 词库问题（拼音错、缺词）优先在 `tools/user_words.txt` 或构建脚本里修，不要硬编码在逻辑里。

## 提 PR

1. fork + 分支
2. 跑 `python3 -m pytest tests -q`
3. 说明改了什么、为什么、验证方式
