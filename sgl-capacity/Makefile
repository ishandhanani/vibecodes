.PHONY: install-sglang

ROOT ?= $(HOME)/test
REPO_DIR ?= $(ROOT)/sglang
PYTHON_BIN ?= python3.12

install-sglang:
	@bash -c '\
	set -euo pipefail; \
	mkdir -p "$(ROOT)"; \
	cd "$(ROOT)"; \
	if [[ ! -d "$(REPO_DIR)/.git" ]]; then \
		git clone https://github.com/sgl-project/sglang.git "$(REPO_DIR)"; \
	else \
		git -C "$(REPO_DIR)" fetch --all -p; \
	fi; \
	if [[ ! -d "$(ROOT)/.venv" ]]; then \
		uv venv --python "$(PYTHON_BIN)" "$(ROOT)/.venv"; \
	fi; \
	source "$(ROOT)/.venv/bin/activate"; \
	sudo apt-get update; \
	sudo apt-get install -y --no-install-recommends \
		git curl build-essential \
		libnuma-dev libnuma1 numactl \
		libtbb-dev google-perftools; \
	uv pip install -U pip setuptools wheel "scikit-build-core>=0.10" cmake ninja; \
	cd "$(REPO_DIR)/python"; \
	cp pyproject_cpu.toml pyproject.toml; \
	uv pip install .; \
	uv pip install --index-url https://download.pytorch.org/whl/cpu \
		--force-reinstall \
		torch==2.9.0 torchvision==0.24.0 triton==3.5.0; \
	cd "$(REPO_DIR)/sgl-kernel"; \
	cp pyproject_cpu.toml pyproject.toml; \
	CUDA_HOME= CUDA_PATH= uv pip install --no-build-isolation .; \
	echo ""; \
	echo "OK: installed sglang + sgl-kernel (CPU)."; \
	echo "Activate: source $(ROOT)/.venv/bin/activate"'
