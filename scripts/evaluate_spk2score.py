import argparse
import sys
from collections import defaultdict
import numpy as np

from pathbench.dataset import Dataset
from pathbench.evaluator import Spk2ScoreEvaluator, DurationEvaluator


def main():
    dataset_dirs = [
        "datasets/copas/pathological/utterances",
        "datasets/copas/pathological/word",
        "datasets/easycall/pathological/commands",
        "datasets/torgo/pathological/utterances",
        "datasets/torgo/pathological/word",
    ]

    results = {}
    for dataset_dir in dataset_dirs:
        print(f"--- Evaluating: {dataset_dir} ---")
        try:
            pcc = evaluate_dataset(dataset_dir)
            if pcc is not None:
                results[dataset_dir] = pcc
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
        print("\n")

    print("\n--- Evaluation Summary ---")
    
    datasets = list(results.keys())
    header = "| Metric |" + "".join([f" {dataset} |" for dataset in datasets])
    print(header)
    print("|" + "---|" * (len(datasets) + 1))
    
    row = "| PCC (spk2score vs duration) |"
    for dataset in datasets:
        pcc = results[dataset]
        row += f" {pcc:.4f} |"
    print(row)


def evaluate_dataset(dataset_dir):
    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {dataset_dir}")
    dataset = Dataset(dataset_dir)

    # --- 2. Initialize Evaluators ---
    if not dataset.spk2score or not dataset.utt2spk:
        print(
            f"Error: Cannot run spk2score evaluation, required files not found in {dataset_dir}",
            file=sys.stderr,
        )
        return None

    evaluators = {
        "spk2score": Spk2ScoreEvaluator(dataset.spk2score, dataset.utt2spk),
        "duration": DurationEvaluator(),
    }

    # --- 3. Run Evaluation ---
    print("\nRunning evaluation...")
    scores = defaultdict(list)
    scored_utterances = 0

    for utt_id, audio_path, transcription in dataset:
        evaluator_scores = {}
        for name, evaluator in evaluators.items():
            score = evaluator.score(
                utterance_id=utt_id,
                audio_path=audio_path,
                transcription=transcription,
                language=dataset.language,
            )
            evaluator_scores[name] = score

        # Only include this utterance if all evaluators returned a valid score
        if all(s is not None for s in evaluator_scores.values()):
            for name, score in evaluator_scores.items():
                scores[name].append(score)
            scored_utterances += 1
        else:
            print(f"  - Skipping utterance {utt_id} (at least one evaluator failed)")

    # --- 4. Report Results ---
    print("\n--- Evaluation Summary ---")
    print(f"Successfully scored {scored_utterances} utterances across all evaluators.")

    if scored_utterances > 0:
        print("\nAverage Scores:")
        for name, score_list in scores.items():
            avg_score = np.mean(score_list)
            print(f"  - {name}: {avg_score:.4f}")

    if "duration" in scores and "spk2score" in scores:
        if len(scores["duration"]) > 1:
            pcc = np.corrcoef(scores["spk2score"], scores["duration"])[0, 1]
            print("\n--- Correlation Analysis ---")
            print(f"Pearson Correlation between 'spk2score' and 'duration': {pcc:.4f}")
            print(f"(Based on {len(scores['duration'])} commonly scored utterances)")
            return pcc
        else:
            print("\nCould not calculate correlation: not enough common data points.")
    
    return None


if __name__ == "__main__":
    main()