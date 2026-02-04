#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -L jobenv=singularity
#PJM -o "utt_evaluation_%j.out"
#PJM -j

# --- User Configuration ---
# List of all dataset paths to be evaluated in this job.
DATASET_PATHS=(
    "datasets/neurovoz/pathological/utterances/balanced"
    "datasets/neurovoz/pathological/utterances/unbalanced"
    "datasets/neurovoz/pathological/utterances/all"
    "datasets/youtube/"
)

# --- Execution ---
module load singularity
singularity exec \
	--bind $HOME,/data/group1/${USER} \
	--nv /data/group1/${USER}/latest.sif \
	bash -c '
    source ~/.bashrc
    source tools/venv/bin/activate
    export TMPDIR=/data/group1/z40484r/projects/pathbench/tmp
    
    echo "Running utt2score evaluation for the following datasets:"
    echo "${DATASET_PATHS[@]}"
    
    # The dataset paths are passed as arguments to the python script
    PHONEMIZER_ESPEAK_LIBRARY=/data/group1/z40484r/projects/vowel_space_area/tools/espeak-ng/.local/lib/libespeak-ng.so python -u /data/group1/z40484r/projects/pathbench/scripts/evaluate_utt2score.py "$@"
    ' -- "${DATASET_PATHS[@]}"

echo "Job finished."
