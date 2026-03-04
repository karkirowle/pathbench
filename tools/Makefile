PYTHON:= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
CUDA_VERSION:= 11.2
PYTORCH_VERSION:= 2.6.0
DOT:= .
MIN_PYTHON:= 3.9
MAX_PYTHON:= 3.12
.PHONY: all clean show_variables check_python
all: show_variables check_python virtualenv.done pytorch.done bench.done

# load singularity, and ~/.bashrc before make to have python
# module load gcc/11.3.0 module load cuda is needed in advance on the cluster
show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)

check_python:
	@test -n "$(PYTHON)" || { echo "Error: no python or python3 found in PATH"; exit 1; }
	@$(PYTHON) -c "import sys; v=sys.version_info; mi=($(subst .,$(comma),$(MIN_PYTHON))); ma=($(subst .,$(comma),$(MAX_PYTHON))); \
		ok=mi<=(v.major,v.minor)<=ma; \
		print(f'Python {v.major}.{v.minor}.{v.micro}' + (' OK' if ok else ' UNSUPPORTED')); \
		sys.exit(0 if ok else 1)" || { echo "Error: Python $(MIN_PYTHON)-$(MAX_PYTHON) required"; exit 1; }

comma:= ,

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; cd ../; pip install -U pip
	. venv/bin/activate; cd ../; pip install numpy
	touch virtualenv.done

pytorch.done: virtualenv.done
ifeq ($(CUDA_VERSION),)
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cpu/stable.html
else
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
endif
	touch pytorch.done

bench.done: virtualenv.done pytorch.done
	. venv/bin/activate; cd ../; pip install -e ".[all]"
	. venv/bin/activate; pip install pytest kenlm
	touch bench.done

clean:
	rm -fr venv *.done
	find -iname "*.pyc" -delete
