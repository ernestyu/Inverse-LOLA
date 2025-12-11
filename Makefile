CONF ?= configs/t1_gridworld.yaml
SEED ?= 0
PYTHON ?= python

ifeq ($(OS),Windows_NT)
ifneq ("$(wildcard .venv/Scripts/python.exe)","")
	PYTHON := .venv/Scripts/python.exe
endif
else
ifneq ("$(wildcard .venv/bin/python)","")
	PYTHON := .venv/bin/python
endif
endif

.PHONY: check gen

check:
	$(PYTHON) check_env.py

gen:
	$(PYTHON) -m runners.gen_data --config $(CONF) --seed $(SEED)

.PHONY: lfl

lfl:
	$(PYTHON) -m runners.run_ma_lfl --config $(CONF) --seed $(SEED)
