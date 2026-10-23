.PHONY: install uninstall test e2e lexicon clean

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

clean:
	rm -rf .pytest_cache src/ipinyin/__pycache__ tests/__pycache__ tools/__pycache__
