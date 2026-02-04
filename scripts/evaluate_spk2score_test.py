import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import datetime
import inspect
import hashlib

from pathbench.evaluator import Spk2ScoreEvaluator, PEREvaluator, ASREvaluator, DirectPEREvaluator
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.nad_evaluator import NADEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator
from pathbench.speech_rate import WpmEvaluator
from pathbench.dataset import Dataset
from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from pathbench.cpp_evaluator import CPPEvaluator
from pathbench.wada_snr import WadaSnrEvaluator
from pathbench.age_evaluator import Spk2AgeEvaluator

def get_class_hash(instance):
    """Gets the SHA256 hash of the source code of an object's class."""
    try:
        source_code = inspect.getsource(instance.__class__)
        return hashlib.sha256(source_code.encode()).hexdigest()
    except (TypeError, OSError):
        # Handle cases where source code can't be found (e.g., built-in types, C extensions)
        return "N/A"

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"results/evaluation_results_test_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        output_file.write(f"Evaluation run on: {timestamp}\n\n")

        dataset_dirs = [
            "datasets/easycall/pathological/word/balanced",
            "datasets/easycall/pathological/word/unbalanced",
            "datasets/easycall/pathological/word/all",
            "datasets/easycall/pathological/utterances/balanced",
            "datasets/easycall/pathological/utterances/unbalanced",
            "datasets/easycall/pathological/utterances/all",
        ]

        results = {}
        for dataset_dir in dataset_dirs:
            output_file.write(f"--- Evaluating: {dataset_dir} ---\n")
            try:
                res = evaluate_dataset(dataset_dir, output_file)
                if res is not None:
                    results[dataset_dir] = res
            except FileNotFoundError as e:
                print(f"Error loading dataset: {e}", file=sys.stderr)
            output_file.write("\n")

        output_file.write("\n--- Evaluation Summary ---\n")

        datasets = list(results.keys())
        header = "| Metric |" + "".join([f" {dataset} |" for dataset in datasets])
        output_file.write(header + "\n")
        output_file.write("|" + "---|" * (len(datasets) + 1) + "\n")

        metrics = [ "speech_rate", "cpp", "per", "dper", "artp", "p_estoi", "p_estoi_fa", "nad", "wada_snr", "spk2age" ]
        for metric in metrics:
            row = f"| PCC (spk2score vs {metric}) |"
            for dataset in datasets:
                pcc = results[dataset].get(f"pcc_{metric}")
                if pcc is not None:
                    row += f" {pcc:.4f} |"
                else:
                    row += " N/A |"
            output_file.write(row + "\n")


def _calculate_pcc(scores, metric, results, output_file):
    """Calculates and stores the Pearson correlation for a given metric against spk2score."""
    if metric in scores and "spk2score" in scores:
        if len(scores[metric]) > 1:
            pcc = np.corrcoef(scores["spk2score"], scores[metric])[0, 1]
            output_file.write(f"Pearson Correlation between 'spk2score' and '{metric}': {pcc:.4f}\n")
            output_file.write(f"(Based on {len(scores[metric])} commonly scored utterances)\n")
            results[f"pcc_{metric}"] = pcc
        else:
            output_file.write(f"\nCould not calculate correlation for {metric}: not enough common data points.\n")

def evaluate_dataset(dataset_dir, output_file):
    # --- 1. Load Dataset ---
    output_file.write(f"Loading dataset from: {dataset_dir}\n")
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
        "speech_rate": WpmEvaluator(),
        "cpp": CPPEvaluator(),
        "per": PEREvaluator(),
        "dper" : DirectPEREvaluator(),
        "artp": ArticulatoryPrecisionEvaluator(),
        "wada_snr": WadaSnrEvaluator(),
        # "per": PEREvaluator("jonatasgrosman/wav2vec2-large-xlsr-53-spanish"),
        #"wer": ASREvaluator("jonatasgrosman/wav2vec2-large-xlsr-53-spanish"),
    }
    if dataset.spk2age:
        evaluators["spk2age"] = Spk2AgeEvaluator(dataset.spk2age, dataset.utt2spk)
    # Example of how other evaluators would be added
    if use_reference:
         
         evaluators["p_estoi"] = ESTOIEvaluator(
                normalization_method = "RMS",
                centroid_ind = 0,
                frame_deletion = True)
             
         evaluators["p_estoi_fa"] = ForcedAlignmentPESTOIEvaluator(
         )

         evaluators["nad"] = NADEvaluator()

    output_file.write("\n--- Evaluator Hashes ---\n")
    for name, evaluator in evaluators.items():
        output_file.write(f"{name}: {get_class_hash(evaluator)}\n")
    output_file.write("\n")

    # --- 3. Run Evaluation & Collect Utterance Scores ---
    output_file.write("\nRunning evaluation...\n")
    # Structure: {speaker_id: {metric: [scores]}}
    spk_utt_scores = defaultdict(lambda: defaultdict(list))
    
    for utt_id, audio_path, transcription, reference_audios in tqdm(dataset, desc=f"Evaluating {dataset_dir}"):
        speaker_id = dataset.utt2spk.get(utt_id)
        if not speaker_id:
            continue
        # print("reference_audios:", reference_audios)
        for name, evaluator in evaluators.items():
            score = evaluator.score(
                utterance_id=utt_id,
                audio_path=audio_path,
                reference_audios=reference_audios,
                transcription=transcription,
                language=dataset.language,
            )
            if score is not None:
                spk_utt_scores[speaker_id][name].append(score)

    # --- 4. Aggregate to Speaker Level ---
    output_file.write("\nAggregating scores to speaker level...\n")
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

    output_file.write(f"Found {len(valid_speakers)} speakers with complete scores for aggregation.\n")

    for spk_id in sorted(valid_speakers):
        # For each metric, calculate the speaker-level score
        for metric_name, utt_scores in spk_utt_scores[spk_id].items():
            if not utt_scores:
                continue
            
            # The Spk2ScoreEvaluator already gives a speaker-level score, so just take the first one.
            # For all other evaluators, calculate the mean of the utterance scores.
            if isinstance(evaluators[metric_name], (Spk2ScoreEvaluator, Spk2AgeEvaluator)):
                spk_level_score = utt_scores[0]
            else:
                spk_level_score = np.mean(utt_scores)
            
            agg_spk_metrics[metric_name].append(spk_level_score)

    # --- 5. Report Results & Correlation ---
    results = {}
    output_file.write("\n--- Correlation Analysis (Speaker Level) ---\n")
    
    # Get a list of metrics that were successfully aggregated, excluding the ground truth itself
    evaluated_metrics = [m for m in agg_spk_metrics.keys() if m != "spk2score"]

    for metric_name in evaluated_metrics:
        _calculate_pcc(agg_spk_metrics, metric_name, results, output_file)

    return results if results else None


if __name__ == "__main__":
    main()
