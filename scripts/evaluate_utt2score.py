import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from pathbench.evaluator import Utt2ScoreEvaluator, PEREvaluator, ASREvaluator
from pathbench.dataset import Dataset


def main():
    dataset_dirs = [
        "datasets/torgo/pathological/utterances",
    ]

    results = {}
    for dataset_dir in dataset_dirs:
        print(f"--- Evaluating: {dataset_dir} ---")
        try:
            res = evaluate_dataset(dataset_dir)
            if res is not None:
                results[dataset_dir] = res
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
        print("\n")

    print("\n--- Evaluation Summary ---")

    datasets = list(results.keys())
    header = "| Metric |" + "".join([f" {dataset} |" for dataset in datasets])
    print(header)
    print("|" + "---|" * (len(datasets) + 1))

    row = "| ASR (1-PER) |"
    for dataset in datasets:
        per = results[dataset]["per"]
        row += f" {per:.4f} |"
    print(row)

    row = "| ASR (1-WER) |"
    for dataset in datasets:
        wer = results[dataset]["wer"]
        row += f" {wer:.4f} |"
    print(row)


def evaluate_dataset(dataset_dir):
    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {dataset_dir}")
    dataset = Dataset(dataset_dir)
    # --- 2. Initialize Evaluators ---
    if not dataset.utt2score:
        print(
            f"Error: Cannot run utt2score evaluation, utt2score file not found in {dataset_dir}",
            file=sys.stderr,
        )
        return None

    evaluators = {
        "utt2score": Utt2ScoreEvaluator(dataset.utt2score),
        "per": PEREvaluator("facebook/wav2vec2-base-960h"),
        "wer": ASREvaluator("facebook/wav2vec2-base-960h"),
    }

    # --- 3. Run Evaluation ---
    print("\nRunning evaluation...")
    scores = defaultdict(list)
    scored_utterances = 0

    for utt_id, audio_path, transcription in tqdm(dataset, desc=f"Evaluating {dataset_dir}"):
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

    results = {}
    if "per" in scores:
        results["per"] = np.mean(scores["per"])
    if "wer" in scores:
        results["wer"] = np.mean(scores["wer"])

    return results if results else None


if __name__ == "__main__":
    main()