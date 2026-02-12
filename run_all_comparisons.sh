#!/bin/bash

# This script runs the comparison_test_2.sh for all datasets one-by-one.

# List of all dataset paths to be evaluated
ALL_DATASETS=(
    "datasets/copas/pathological/word/balanced"
    "datasets/copas/pathological/word/unbalanced"
    "datasets/copas/pathological/word/all"
    "datasets/copas/pathological/utterances/balanced"
    "datasets/copas/pathological/utterances/unbalanced"
    "datasets/copas/pathological/utterances/all"
    "datasets/easycall/pathological/word/balanced"
    "datasets/easycall/pathological/word/unbalanced"
    "datasets/easycall/pathological/word/all"
    "datasets/easycall/pathological/utterances/balanced"
    "datasets/easycall/pathological/utterances/unbalanced"
   "datasets/easycall/pathological/utterances/all"
    "datasets/neurovoz/pathological/utterances/balanced"
    "datasets/neurovoz/pathological/utterances/unbalanced"
    "datasets/neurovoz/pathological/utterances/all"
    "datasets/uaspeech/pathological/word/balanced"
    "datasets/uaspeech/pathological/word/unbalanced"
    "datasets/uaspeech/pathological/word/all"
     "datasets/torgo/pathological/utterances/balanced"
     "datasets/torgo/pathological/utterances/unbalanced"
     "datasets/torgo/pathological/word/balanced"
     "datasets/torgo/pathological/word/unbalanced"
     "datasets/youtube/"
)

# Path to the script to be modified
TARGET_SCRIPT="comparison_test_2.sh"

# Loop through all datasets and submit a job for each
for DATASET in "${ALL_DATASETS[@]}"; do
    echo "Submitting job for dataset: ${DATASET}"

    # The following sed command modifies the TARGET_SCRIPT in-place.
    # It finds the line starting with DATASET_PATHS= and replaces the entire line
    # with the current dataset path from the loop.
    sed -i "s|^DATASET_PATHS=.*|DATASET_PATHS=(\"${DATASET}\")|" "${TARGET_SCRIPT}"

    # Submit the job
    pjsub "${TARGET_SCRIPT}"

    # Optional: wait for a second before submitting the next job
    sleep 1
done

echo "All jobs submitted."
