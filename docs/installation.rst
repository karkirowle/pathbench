Installation
============

There are two routes, depending on what you need. Both pin the Hugging Face
model weights to fixed revisions.

Use PathBench in your project
-----------------------------

::

    sudo apt-get install espeak-ng            # or your OS's package manager
    pip install "pathbench @ git+https://github.com/karkirowle/pathbench.git"

Any library versions within the ranges in ``pyproject.toml`` work; CI tests
both the lowest and the latest allowed versions. Extras: ``pathbench[lm]``
(``kenlm``, compiled from source; needed by DArtP and Double ASR) and
``pathbench[scripts]``.

Phoneme-based metrics (PER, dPER, ArtP, DArtP, Double ASR) depend on the
espeak-ng version, which pip cannot control. Different versions phonemize some
languages differently (e.g. Italian ``r`` vs ``ɾ``). To match the paper, point
phonemizer at an espeak-ng built from commit ``2ea41210``::

    export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so

Reproduce the paper
-------------------

::

    git clone https://github.com/karkirowle/pathbench.git
    cd pathbench
    make -C tools                  # CUDA 12.4 PyTorch
    make -C tools CUDA_VERSION=    # or CPU-only PyTorch
    source tools/venv/bin/activate

This builds espeak-ng ``2ea41210`` into ``tools/espeak-ng/`` without sudo,
creates ``tools/venv`` with the exact versions in ``constraints.txt`` and points
the venv at the local espeak-ng. Scores then match the paper up to GPU
floating-point rounding. It needs Python 3.10-3.12, ``git``, ``cmake`` and a
C/C++ compiler.

Without those build tools (e.g. on an HPC cluster), run the same ``make``
inside the released container, defined in ``container/pathbench.def``::

    apptainer exec --nv pathbench.sif make -C tools
