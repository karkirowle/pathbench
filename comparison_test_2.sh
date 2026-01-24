#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -L jobenv=singularity
#PJM -j

module load singularity
singularity exec \
	--bind $HOME,/data/group1/${USER} \
	--nv /data/group1/${USER}/latest.sif \
	bash -c "
    source ~/.bashrc
    source tools/venv/bin/activate
    export TMPDIR=/data/group1/z40484r/projects/pathbench/tmp
    PHONEMIZER_ESPEAK_LIBRARY=/data/group1/z40484r/projects/vowel_space_area/tools/espeak-ng/.local/lib/libespeak-ng.so python -u /data/group1/z40484r/projects/pathbench/scripts/evaluate_spk2score_test_2.py
    "

