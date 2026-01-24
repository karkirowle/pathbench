import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse

from pathbench.evaluator import Spk2ScoreEvaluator, PEREvaluator, ASREvaluator, DirectPEREvaluator, SpeakerEvaluator
from pathbench.f0_range_evaluator import F0RangeEvaluator
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.nad_evaluator import NADEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator
from pathbench.speech_rate import WpmEvaluator
from pathbench.dataset import Dataset
from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from pathbench.cpp_evaluator import CPPEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker scores against various metrics.")
    parser.add_argument('dataset_dirs', nargs='+', help='List of dataset directories to evaluate.')
    args = parser.parse_args()

    results = {}
    for dataset_dir in args.dataset_dirs:
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

    metrics = [ "speech_rate", "cpp", "per", "dper", "artp", "p_estoi", "p_estoi_fa", "nad", "f0_range" ]
    for metric in metrics:
        row = f"| PCC (spk2score vs {metric}) |"
        for dataset in datasets:
            pcc = results[dataset].get(f"pcc_{metric}")
            if pcc is not None:
                row += f" {pcc:.4f} |"
            else:
                row += " N/A |"
        print(row)


def evaluate_dataset(dataset_dir):
    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {dataset_dir}")
    use_reference = True  # Always try to use reference audio
    reference_path = dataset_dir.replace("pathological", "control")
    
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

    utt_evaluators = {
        "spk2score": Spk2ScoreEvaluator(dataset.spk2score, dataset.utt2spk),
        "speech_rate": WpmEvaluator(),
        "cpp": CPPEvaluator(),
        "per": PEREvaluator(),
        "dper" : DirectPEREvaluator(),
        "artp": ArticulatoryPrecisionEvaluator(),
        "p_estoi": ESTOIEvaluator(
            normalization_method="RMS",
            centroid_ind=0,
            frame_deletion=True
        ),
        "p_estoi_fa": ForcedAlignmentPESTOIEvaluator(),
        "nad": NADEvaluator(),
    }
    spk_evaluators = {
        "f0_range": F0RangeEvaluator(),
    }

    # --- 3. Run Evaluation & Collect Utterance Scores ---
    print("\nRunning utterance-level evaluation...")
    # Structure: {speaker_id: {metric: [scores]}}
    spk_utt_scores = defaultdict(lambda: defaultdict(list))
    
    for utt_id, audio_path, transcription, reference_audios, start_time, end_time in tqdm(dataset, desc=f"Evaluating {dataset_dir}"):
        speaker_id = dataset.utt2spk.get(utt_id)
        if not speaker_id:
            continue
        
        for name, evaluator in utt_evaluators.items():
            # For reference-based evaluators, only score if reference_audios are available
            if isinstance(evaluator, (ESTOIEvaluator, ForcedAlignmentPESTOIEvaluator, NADEvaluator)) and not reference_audios:
                continue

            score = evaluator.score(
                utterance_id=utt_id,
                audio_path=audio_path,
                reference_audios=reference_audios,
                transcription=transcription,
                language=dataset.language,
                start_time=start_time,
                end_time=end_time,
            )
            if score is not None:
                spk_utt_scores[speaker_id][name].append(score)

    # --- 3.5 Run Speaker-level Evaluation ---
    print("\nRunning speaker-level evaluation...")
    spk_audio_files = defaultdict(list)
    spk_transcriptions = defaultdict(list)
    for utt_id, audio_path, transcription, _, start_time, end_time in dataset:
        speaker_id = dataset.utt2spk.get(utt_id)
        if speaker_id:
            spk_audio_files[speaker_id].append((audio_path, start_time, end_time))
            spk_transcriptions[speaker_id].append(transcription)

    for spk_id in tqdm(spk_audio_files.keys(), desc="Evaluating speakers"):
        for name, evaluator in spk_evaluators.items():
            score = evaluator.score(
                audio_files=spk_audio_files[spk_id],
                transcriptions=spk_transcriptions[spk_id],
                language=dataset.language,
            )
            if score is not None:
                spk_utt_scores[spk_id][name] = [score]

    # --- 4. Aggregate to Speaker Level ---
    print("\nAggregating scores to speaker level...")
    # Structure: {metric: {speaker_id: speaker_level_score}}
    agg_spk_metrics = defaultdict(dict)
    
    all_evaluators = {**utt_evaluators, **spk_evaluators}

    for spk_id, metrics in spk_utt_scores.items():
        # For each metric, calculate the speaker-level score
        for metric_name, utt_scores in metrics.items():
            if not utt_scores:
                continue
            
            if isinstance(all_evaluators.get(metric_name), (Spk2ScoreEvaluator, SpeakerEvaluator)):
                spk_level_score = utt_scores[0]
            else:
                spk_level_score = np.mean(utt_scores)
            
            agg_spk_metrics[metric_name][spk_id] = spk_level_score

    # --- 5. Report Results & Correlation ---
    results = {}
    print("\n--- Correlation Analysis (Speaker Level) ---")
    
    evaluated_metrics = [m for m in agg_spk_metrics.keys() if m != "spk2score"]
    ground_truth_scores = agg_spk_metrics.get("spk2score", {})

    for metric_name in evaluated_metrics:
        metric_scores = agg_spk_metrics.get(metric_name, {})
        
        # Find common speakers
        common_speakers = sorted(list(ground_truth_scores.keys() & metric_scores.keys()))
        
        if len(common_speakers) < 2:
            print(f"\nCould not calculate correlation for {metric_name}: not enough common data points.")
            continue
            
        gt_vals = [ground_truth_scores[spk] for spk in common_speakers]
        metric_vals = [metric_scores[spk] for spk in common_speakers]
        
        pcc = np.corrcoef(gt_vals, metric_vals)[0, 1]
        print(f"Pearson Correlation between 'spk2score' and '{metric_name}': {pcc:.4f}")
        print(f"(Based on {len(common_speakers)} commonly scored speakers)")
        results[f"pcc_{metric_name}"] = pcc

    return results if results else None


if __name__ == "__main__":
    main()
