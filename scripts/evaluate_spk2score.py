import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from pathbench.evaluator import Spk2ScoreEvaluator, PEREvaluator, ASREvaluator, DirectPEREvaluator
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator
from pathbench.dataset import Dataset


def main():
    dataset_dirs = [
        #"datasets/uaspeech/pathological/word",
        #"datasets/copas/pathological/utterances",
        #"datasets/neurovoz_clean",
        "datasets/neurovoz_balanced/pathological",
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

    metrics = [ "dper", "p-estoi" ]
    for metric in metrics:
        row = f"| PCC (spk2score vs {metric}) |"
        for dataset in datasets:
            pcc = results[dataset].get(f"pcc_{metric}")
            if pcc is not None:
                row += f" {pcc:.4f} |"
            else:
                row += " N/A |"
        print(row)


def _calculate_pcc(scores, metric, results):
    """Calculates and stores the Pearson correlation for a given metric against spk2score."""
    if metric in scores and "spk2score" in scores:
        if len(scores[metric]) > 1:
            pcc = np.corrcoef(scores["spk2score"], scores[metric])[0, 1]
            print(f"Pearson Correlation between 'spk2score' and '{metric}': {pcc:.4f}")
            print(f"(Based on {len(scores[metric])} commonly scored utterances)")
            results[f"pcc_{metric}"] = pcc
        else:
            print(f"\nCould not calculate correlation for {metric}: not enough common data points.")

def evaluate_dataset(dataset_dir):
    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {dataset_dir}")
    use_reference = "pathological" in dataset_dir
    reference_path = dataset_dir.replace("pathological", "control") if use_reference else None
    
    dataset = Dataset(
        dataset_dir,
        use_reference=use_reference,
        reference_path=reference_path,
        reference_type='control'
    )

    # --- 2. Initialize Evaluators ---
    if not dataset.spk2score or not dataset.utt2spk:
        print(
            f"Error: Cannot run spk2score evaluation, required files not found in {dataset_dir}",
            file=sys.stderr,
        )
        return None

    evaluators = {
        "spk2score": Spk2ScoreEvaluator(dataset.spk2score, dataset.utt2spk),
        #"artp": ArticulatoryPrecisionEvaluator(),
        "dper" : DirectPEREvaluator(),
        # "per": PEREvaluator("jonatasgrosman/wav2vec2-large-xlsr-53-spanish"),
        #"wer": ASREvaluator("jonatasgrosman/wav2vec2-large-xlsr-53-spanish"),
    }
    # Example of how other evaluators would be added
    if use_reference:
         evaluators["p-estoi"] = ESTOIEvaluator(
             normalization_method='RMS',
             centroid_ind=0,
             frame_deletion=True
         )

    # --- 3. Run Evaluation & Collect Utterance Scores ---
    print("\nRunning evaluation...")
    # Structure: {speaker_id: {metric: [scores]}}
    spk_utt_scores = defaultdict(lambda: defaultdict(list))
    
    for utt_id, audio_path, transcription, _ in tqdm(dataset, desc=f"Evaluating {dataset_dir}"):
        speaker_id = dataset.utt2spk.get(utt_id)
        if not speaker_id:
            continue

        for name, evaluator in evaluators.items():
            score = evaluator.score(
                utterance_id=utt_id,
                audio_path=audio_path,
                transcription=transcription,
                language=dataset.language,
            )
            if score is not None:
                spk_utt_scores[speaker_id][name].append(score)

    # --- 4. Aggregate to Speaker Level ---
    print("\nAggregating scores to speaker level...")
    # Structure: {metric: [speaker_level_scores]}
    agg_spk_metrics = defaultdict(list)
    
    # Find speakers who have a ground truth score and at least one evaluated score
    valid_speakers = [
        spk for spk, metrics in spk_utt_scores.items()
        if "spk2score" in metrics and len(metrics) > 1
    ]
    
    if not valid_speakers:
        print("Error: No speakers with both ground truth and evaluated scores found.", file=sys.stderr)
        return None

    print(f"Found {len(valid_speakers)} speakers with complete scores for aggregation.")

    for spk_id in sorted(valid_speakers):
        # For each metric, calculate the speaker-level score
        for metric_name, utt_scores in spk_utt_scores[spk_id].items():
            if not utt_scores:
                continue
            
            # The Spk2ScoreEvaluator already gives a speaker-level score, so just take the first one.
            # For all other evaluators, calculate the mean of the utterance scores.
            if isinstance(evaluators[metric_name], Spk2ScoreEvaluator):
                spk_level_score = utt_scores[0]
            else:
                spk_level_score = np.mean(utt_scores)
            
            agg_spk_metrics[metric_name].append(spk_level_score)

    # --- 5. Report Results & Correlation ---
    results = {}
    print("\n--- Correlation Analysis (Speaker Level) ---")
    
    # Get a list of metrics that were successfully aggregated, excluding the ground truth itself
    evaluated_metrics = [m for m in agg_spk_metrics.keys() if m != "spk2score"]

    for metric_name in evaluated_metrics:
        _calculate_pcc(agg_spk_metrics, metric_name, results)

    return results if results else None


if __name__ == "__main__":
    main()
