# "Reproduce the paper" environment: pinned espeak-ng + locked Python deps.
#
#   make                    # CUDA 12.4 torch
#   make CUDA_VERSION=      # CPU-only torch
#
# Needs git, cmake and a C/C++ compiler, but no sudo: espeak-ng is built into
# tools/espeak-ng/ and the venv is pointed at it. Without those build tools,
# run the same command inside the container: apptainer exec pathbench.sif make
PYTHON:= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
CUDA_VERSION:= 12.4
PYTORCH_VERSION:= 2.6.0
ESPEAK_COMMIT:= 2ea41210
DOT:= .
MIN_PYTHON:= 3.10
MAX_PYTHON:= 3.12
ESPEAK_PREFIX:= $(CURDIR)/espeak-ng
ifeq ($(CUDA_VERSION),)
TORCH_INDEX:= cpu
else
TORCH_INDEX:= cu$(subst $(DOT),,$(CUDA_VERSION))
endif
.PHONY: all clean show_variables check_python check_build_tools
all: show_variables check_python espeak.done virtualenv.done pytorch.done bench.done

show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION) TORCH_INDEX=$(TORCH_INDEX)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)
	@echo ESPEAK_COMMIT=$(ESPEAK_COMMIT) ESPEAK_PREFIX=$(ESPEAK_PREFIX)

check_python:
	@test -n "$(PYTHON)" || { echo "Error: no python or python3 found in PATH"; exit 1; }
	@$(PYTHON) -c "import sys; v=sys.version_info; mi=($(subst .,$(comma),$(MIN_PYTHON))); ma=($(subst .,$(comma),$(MAX_PYTHON))); \
		ok=mi<=(v.major,v.minor)<=ma; \
		print(f'Python {v.major}.{v.minor}.{v.micro}' + (' OK' if ok else ' UNSUPPORTED')); \
		sys.exit(0 if ok else 1)" || { echo "Error: Python $(MIN_PYTHON)-$(MAX_PYTHON) required"; exit 1; }

check_build_tools:
	@for t in git cmake cc c++; do command -v $$t >/dev/null || { \
		echo "Error: '$$t' not found. Install it, or run make inside the container (see README)."; exit 1; }; done

comma:= ,

# espeak-ng at the pinned commit, installed under tools/espeak-ng (no sudo).
# Different espeak-ng versions phonemize some languages differently (e.g.
# Italian r vs ɾ), which changes the phoneme-based metrics.
espeak.done: check_build_tools
	rm -rf espeak-ng-src
	git clone https://github.com/espeak-ng/espeak-ng.git espeak-ng-src
	cd espeak-ng-src && git checkout $(ESPEAK_COMMIT)
	cd espeak-ng-src && cmake -B build -DCMAKE_INSTALL_PREFIX=$(ESPEAK_PREFIX) -DUSE_ASYNC=OFF -DBUILD_SHARED_LIBS=ON
	cd espeak-ng-src && cmake --build build -j$$(nproc) && cmake --install build
	rm -rf espeak-ng-src
	touch espeak.done

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; cd ../; pip install -U pip
	touch virtualenv.done

pytorch.done: virtualenv.done
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) torchaudio==$(PYTORCH_VERSION) \
		--index-url https://download.pytorch.org/whl/$(TORCH_INDEX)
	touch pytorch.done

# A .pth file runs at every interpreter start, so the venv uses the pinned
# espeak-ng even when called without `activate` (e.g. tools/venv/bin/python).
# setdefault keeps an explicit PHONEMIZER_ESPEAK_LIBRARY from the user.
bench.done: virtualenv.done pytorch.done espeak.done
	. venv/bin/activate; cd ../; pip install -c constraints.txt -e ".[lm,scripts]" pytest
	. venv/bin/activate; echo "import os; os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', '$(ESPEAK_PREFIX)/lib/libespeak-ng.so')" \
		> $$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/pathbench_espeak.pth
	touch bench.done

clean:
	rm -fr venv espeak-ng espeak-ng-src *.done
	find -iname "*.pyc" -delete
