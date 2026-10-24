.PHONY: install uninstall test e2e lexicon clean dev-macos corpus lm train-yue eval-yue eval-word eval-sentence eval-docs

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

clean:
	rm -rf .pytest_cache src/ipinyin/__pycache__ tests/__pycache__ tools/__pycache__
