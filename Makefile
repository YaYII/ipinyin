.PHONY: install uninstall test e2e lexicon clean dev-macos corpus lm train-yue eval-yue eval-word eval-sentence eval-docs eval-coverage sogou-dict yue-guard eval-yue-guard macos macos-testpad push

install:
	./install.sh

uninstall:
	./uninstall.sh

test:
	python3 -m pytest tests -q

e2e:
	python3 tests/e2e_ibus.py

lexicon:
	./tools/fetch_data.sh
	python3 tools/build_lexicon.py

# macOS 开发环境（不装 IBus，只跑逻辑层 + 训练）
dev-macos:
	./tools/dev_macos.sh --with-nn

# 粤语平行语料（大陆网络自动走 hf-mirror）
corpus:
	python3 tools/fetch_yue_pairs.py

# 字级语言模型（给整句候选重排；约 3MB，生成物不入库）
# 加上自己的文档：make lm CORPUS="vendor/yue_pairs.json vendor/personal"
CORPUS ?= vendor/yue_pairs.json vendor/personal/_clean
lm:
	python3 tools/merge_personal_corpus.py vendor/personal
	python3 tools/build_lm.py --corpus $(CORPUS)

# 在 M 系列上训练（约 10~15 分钟，d=256 / 4 层 / 15 万句 4 轮）
train-yue:
	python3 experiments/train_yue_seq2seq.py --limit 150000 --epochs 4 \
		--d 256 --layers 4 --device auto --out experiments/yue_model.pt

# 与规则版对比评测（--beam 4 是升级后的解码）
eval-yue:
	python3 experiments/eval_yue_seq2seq.py experiments/yue_model.pt --beam 4

# 简中：词级（给拼音，词能不能一次出来）
eval-word:
	python3 tools/eval_quality.py

# 简中：整句（给一串拼音，第一候选是不是你想打的那句）— 核心指标
eval-sentence:
	python3 tools/eval_sentence.py

# 简中：拿你自己的简体文档端到端测（无重排 vs 重排，救回/弄坏分开报）
eval-docs:
	python3 tools/eval_personal_docs.py

# 简中：逐词覆盖率（评补词专用；整句自洽口径量不到含缺词的句子）
eval-coverage:
	python3 tools/eval_word_coverage.py

# 搜狗细胞词库（本地评测用；许可：不得用于商业用途）
sogou-dict:
	python3 tools/fetch_sogou_dict.py --out vendor/sogou

# 粤语解码守卫数据（P1 约束解码；全量语料，入库）
yue-guard:
	python3 tools/build_yue_guard.py --all --out data/yue_guard.marshal

# 粤语守卫评测（分桶 + 修复样例；自动用训练侧切分的守卫，先跑一次 build_yue_guard.py）
eval-yue-guard:
	python3 experiments/exp_yue_guard.py --model experiments/yue_model_d256.pt

# 推 GitHub（大陆网络：github.com 直连不通，走 api.github.com 的 git-data API）
push:
	python3 tools/gh_push.py HEAD master

# macOS 输入法（构建 + 装机 + 注册；激活需注销重登，见 docs/macos.md）
macos:
	./tools/macos/build_macos_app.sh

# macOS 测试垫（不依赖系统输入源，直接体验引擎）
macos-testpad:
	/usr/bin/python3 tools/macos_testpad.py

# macOS 打字条（⌃⌥空格唤出，文字注入任意 App；首次要授权「辅助功能」）
macos-bar:
	pkill -f "IPinyin --bar" 2>/dev/null || true
	nohup "$$HOME/Library/Input Methods/IPinyin.app/Contents/MacOS/IPinyin" --bar \
		>> "$$HOME/Library/Application Support/ipinyin/bar.log" 2>&1 &

clean:
	rm -rf .pytest_cache src/ipinyin/__pycache__ tests/__pycache__ tools/__pycache__
