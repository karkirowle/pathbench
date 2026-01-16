PYTHON:= python
CUDA_VERSION:= 11.2
PYTORCH_VERSION:= 2.6.0
DOT:= .
.PHONY: all clean show_variables bench.done
all: show_variables virtualenv.done pytorch.done  bench.done

# load singularity, and ~/.bashrc before make to have python
# module load gcc/11.3.0 module load cuda is needed in advance on the cluster
show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; cd ../; pip install -U pip
	# install numpy here since python3.6 is not supported in > 1.20
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
	# Phonemizer temporarily replaced with: https://github.com/thewh1teagle/phonemizer-fork.git
	# This fork fixes caching issues with the original phonemizer
	. venv/bin/activate; pip install transformers matplotlib librosa tqdm jiwer phonemizer
	touch bench.done

clean:
	rm -fr venv *.done
	find -iname "*.pyc" -delete
