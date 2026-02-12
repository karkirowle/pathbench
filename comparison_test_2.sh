#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -L jobenv=singularity

# --- User Configuration ---
# List of dataset paths to evaluate in this job.
# To run a single dataset, use: DATASET_PATHS=("datasets/copas/pathological/word/balanced")
# To run multiple datasets in the same job, list them separated by spaces:
# DATASET_PATHS=("datasets/copas/pathological/word/balanced" "datasets/copas/pathological/word/all")
# DATASET_PATHS=("datasets/copas/pathological/word/balanced")
# DATASET_PATHS=("datasets/copas/pathological/word/unbalanced") # run done
# DATASET_PATHS=("datasets/copas/pathological/word/all") # ???
# DATASET_PATHS=("datasets/copas/pathological/utterances/balanced") # run done 
# DATASET_PATHS=("datasets/copas/pathological/utterances/unbalanced") # run done
# DATASET_PATHS=("datasets/copas/pathological/utterances/all") # run done
# DATASET_PATHS=("datasets/easycall/pathological/word/balanced") # run done
DATASET_PATHS=("datasets/youtube/")
# DATASET_PATHS=("datasets/easycall/pathological/word/all") # run done
# DATASET_PATHS=("datasets/easycall/pathological/utterances/balanced") # run done
DATASET_PATHS=("datasets/youtube/")
# DATASET_PATHS=("datasets/easycall/pathological/utterances/all") # run done 
# DATASET_PATHS=("datasets/neurovoz/pathological/utterances/balanced") # run done
# DATASET_PATHS=("datasets/neurovoz/pathological/utterances/unbalanced") # run done
# DATASET_PATHS=("datasets/neurovoz/pathological/utterances/all") # run done
# DATASET_PATHS=("datasets/uaspeech/pathological/word/balanced") # run done
# DATASET_PATHS=("datasets/uaspeech/pathological/word/unbalanced") # run done
# DATASET_PATHS=("datasets/uaspeech/pathological/word/all") # run done

# DATASET_PATHS=("datasets/youtube/") # run done

# DATASET_PATHS=("datasets/torgo/pathological/utterances/balanced") 
# DATASET_PATHS=("datasets/torgo/pathological/utterances/unbalanced") 
# DATASET_PATHS=("datasets/torgo/pathological/utterances/all") 

# --- Job Configuration ---
# Generate a job name from the first dataset path. This will be used for the output file name.
# It replaces '/' with '_' to create a valid filename.
FIRST_DATASET_NAME=$(echo "${DATASET_PATHS[0]}" | tr '/' '_')
#PJM -o "${FIRST_DATASET_NAME}_%j.out"
#PJM -j

# --- Execution ---
module load singularity
singularity exec \
	--bind $HOME,/data/group1/${USER} \
	--nv /data/group1/${USER}/latest.sif \
	bash -c '
    source ~/.bashrc
    source tools/venv/bin/activate
    export TMPDIR=/data/group1/z40484r/projects/pathbench/tmp
    echo ${DATASET_PATHS[@]}
    # The dataset paths are passed as arguments to the python script
    PHONEMIZER_ESPEAK_LIBRARY=/data/group1/z40484r/projects/vowel_space_area/tools/espeak-ng/.local/lib/libespeak-ng.so python -u /data/group1/z40484r/projects/pathbench/scripts/evaluate_spk2score_test_2.py "$@"
    ' -- "${DATASET_PATHS[@]}"
