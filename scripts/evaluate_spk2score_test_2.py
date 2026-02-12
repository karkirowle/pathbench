import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import inspect
import hashlib

from pathbench.evaluator import Spk2ScoreEvaluator, PEREvaluator, ASREvaluator, DirectPEREvaluator, SpeakerEvaluator, DoubleASREvaluator
from pathbench.f0_range_evaluator import F0RangeEvaluator, StdPitchEvaluator
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.nad_evaluator import NADEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator, ArticulatoryPrecisionEvaluatorOld
from pathbench.speech_rate import WpmEvaluator, PraatSpeechRateEvaluator
from pathbench.dataset import Dataset
from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from pathbench.cpp_evaluator import CPPEvaluator
from pathbench.wada_snr import WadaSnrEvaluator
from pathbench.age_evaluator import Spk2AgeEvaluator
from pathbench.artp_double_asr_evaluator import ArtPDoubleASREvaluator
from pathbench.vsa_evaluator import VSAEvaluator
from pathbench.vad import FATrimmer

def get_class_hash(instance):
    """Gets the SHA256 hash of the source code of an object's class."""
    try:
        source_code = inspect.getsource(instance.__class__)
        return hashlib.sha256(source_code.encode()).hexdigest()
    except (TypeError, OSError):
        return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker scores against various metrics.")
    parser.add_argument('dataset_dirs', nargs='+', help='List of dataset directories to evaluate.')
    args = parser.parse_args()

    dataset_name = args.dataset_dirs[0].replace('/', '_')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"results_7/{dataset_name}_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        output_file.write(f"Evaluation run on: {timestamp}\n\n")

        results = {}
        for dataset_dir in args.dataset_dirs:
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

        metrics = [
            "speech_rate", "cpp", "per", "dper", "artp", "artp_old", "artp_double_asr", "double_asr",
            "p_estoi_control", "p_estoi_all",
            "p_estoi_fa_control", "p_estoi_fa_all",
            "nad_control", "nad_all",
            "f0_range", "wada_snr", "spk2age", "vsa", "std_pitch",
            "speech_rate_fa", "cpp_fa", "nad_fa_control", "nad_fa_all",
            "f0_range_fa", "vsa_fa", "std_pitch_fa",
            "praat_speech_rate", "praat_speech_rate_fa"
        ]
        for metric in metrics:
            row = f"| PCC (spk2score vs {metric}) |"
            for dataset in datasets:
                pcc = results[dataset].get(f"pcc_{metric}")
                if pcc is not None:
                    row += f" {pcc:.4f} |"
                else:
                    row += " N/A |"
            output_file.write(row + "\n")


def evaluate_dataset(dataset_dir, output_file):
    # --- 1. Load Dataset (Initial load for setup) ---
    output_file.write(f"Loading dataset from: {dataset_dir}\n")
    try:
        base_dataset = Dataset(dataset_dir, use_reference=False)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return None

    # --- 2. Initialize Evaluators ---
    if not base_dataset.spk2score or not base_dataset.utt2spk:
        print(
            f"Error: Cannot run spk2score evaluation, required files not found in {dataset_dir}",
            file=sys.stderr,
        )
        return None

    trimmer = FATrimmer()

    utt_evaluators = {
        "spk2score": Spk2ScoreEvaluator(base_dataset.spk2score, base_dataset.utt2spk),
        "speech_rate": WpmEvaluator(),
        "cpp": CPPEvaluator(),
        "per": PEREvaluator(language=base_dataset.language),
        "dper" : DirectPEREvaluator(),
        "artp": ArticulatoryPrecisionEvaluator(),
        "artp_old": ArticulatoryPrecisionEvaluatorOld(),
        "artp_double_asr": ArtPDoubleASREvaluator(language=base_dataset.language),
        "double_asr": DoubleASREvaluator(language=base_dataset.language),
        "p_estoi": ESTOIEvaluator(
            normalization_method="RMS",
            centroid_ind=0,
            frame_deletion=True
        ),
        "p_estoi_fa": ForcedAlignmentPESTOIEvaluator(),
        "nad": NADEvaluator(),
        "wada_snr": WadaSnrEvaluator(),
        "std_pitch": StdPitchEvaluator(),
        "speech_rate_fa": WpmEvaluator(trimmer=trimmer),
        "cpp_fa": CPPEvaluator(trimmer=trimmer),
        "nad_fa": NADEvaluator(trimmer=trimmer),
        "std_pitch_fa": StdPitchEvaluator(trimmer=trimmer),
        "praat_speech_rate": PraatSpeechRateEvaluator(),
        "praat_speech_rate_fa": PraatSpeechRateEvaluator(trimmer=trimmer),
    }
    if base_dataset.spk2age:
        utt_evaluators["spk2age"] = Spk2AgeEvaluator(base_dataset.spk2age, base_dataset.utt2spk)

    spk_evaluators = {
        "f0_range": F0RangeEvaluator(),
        "vsa": VSAEvaluator(),
        "f0_range_fa": F0RangeEvaluator(trimmer=trimmer),
        "vsa_fa": VSAEvaluator(trimmer=trimmer),
    }

    all_evaluators = {**utt_evaluators, **spk_evaluators}
    output_file.write("\n--- Evaluator Hashes ---\n")
    for name, evaluator in all_evaluators.items():
        output_file.write(f"{name}: {get_class_hash(evaluator)}\n")
    output_file.write("\n")

    # --- 3. Run Evaluation & Collect Utterance Scores ---
    spk_utt_scores = defaultdict(lambda: defaultdict(list))
    
    ref_evaluator_names = ["p_estoi", "p_estoi_fa", "nad"]
    non_ref_evaluators = {k: v for k, v in utt_evaluators.items() if k not in ref_evaluator_names}

    # Run non-reference based utterance evaluators
    output_file.write("\nRunning non-reference utterance-level evaluation...\n")
    for utt_id, audio_path, transcription, _, start_time, end_time in tqdm(base_dataset, desc=f"Evaluating {dataset_dir} (non-ref)"):
        speaker_id = base_dataset.utt2spk.get(utt_id)
        if not speaker_id:
            continue
        
        for name, evaluator in non_ref_evaluators.items():
            score = None
            try:
                score = evaluator.score(
                    utterance_id=utt_id,
                    audio_path=audio_path,
                    reference_audios=None,
                    transcription=transcription,
                    language=base_dataset.language,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception as e:
                print(f"Evaluator {name} failed for {utt_id} with error: {e}")

            if score is not None:
                spk_utt_scores[speaker_id][name].append(score)

    # Run reference-based evaluators for each reference type
    reference_path = dataset_dir.replace("pathological", "control")
    for ref_type in ['control', 'all']:
        output_file.write(f"\nRunning reference-based evaluation with reference_type='{ref_type}'...\n")
        try:
            ref_dataset = Dataset(
                dataset_dir,
                use_reference=True,
                reference_path=reference_path,
                reference_type=ref_type
            )
        except FileNotFoundError as e:
            print(f"Error loading reference dataset for {ref_type}: {e}", file=sys.stderr)
            continue

        is_all_dataset = "/all" in dataset_dir
        skip_ref_metrics = False

        for utt_id, audio_path, transcription, reference_audios, start_time, end_time in tqdm(ref_dataset, desc=f"Evaluating {dataset_dir} ({ref_type})"):
            if skip_ref_metrics: break

            speaker_id = ref_dataset.utt2spk.get(utt_id)
            if not speaker_id:
                continue
            
            for name in ref_evaluator_names:
                evaluator = utt_evaluators[name]

                if not reference_audios:
                    if is_all_dataset:
                        print(f"Missing reference audio for {utt_id} in 'all' dataset with ref_type {ref_type}. Skipping all reference-based metrics for this dataset configuration.")
                        skip_ref_metrics = True
                        break 
                    else:
                        continue

                score = None
                try:
                    score = evaluator.score(
                        utterance_id=utt_id,
                        audio_path=audio_path,
                        reference_audios=reference_audios,
                        transcription=transcription,
                        language=ref_dataset.language,
                        start_time=start_time,
                        end_time=end_time,
                    )
                except Exception as e:
                    print(f"Evaluator {name} ({ref_type}) failed for {utt_id} with error: {e}")

                if score is None:
                    if is_all_dataset:
                        print(f"Reference-based evaluator {name} ({ref_type}) returned None for {utt_id} in 'all' dataset. Skipping all reference-based metrics for this dataset configuration.")
                        skip_ref_metrics = True
                        break
                else:
                    metric_name_with_ref_type = f"{name}_{ref_type}"
                    spk_utt_scores[speaker_id][metric_name_with_ref_type].append(score)

    # --- 3.5 Run Speaker-level Evaluation ---
    output_file.write("\nRunning speaker-level evaluation...\n")
    spk_audio_files = defaultdict(list)
    spk_transcriptions = defaultdict(list)
    for utt_id, audio_path, transcription, _, start_time, end_time in base_dataset:
        speaker_id = base_dataset.utt2spk.get(utt_id)
        if speaker_id:
            spk_audio_files[speaker_id].append((audio_path, start_time, end_time))
            spk_transcriptions[speaker_id].append(transcription)

    for spk_id in tqdm(spk_audio_files.keys(), desc="Evaluating speakers"):
        for name, evaluator in spk_evaluators.items():
            score = evaluator.score(
                audio_files=spk_audio_files[spk_id],
                transcriptions=spk_transcriptions[spk_id],
                language=base_dataset.language,
            )
            if score is not None:
                spk_utt_scores[spk_id][name] = [score]

    # --- 4. Aggregate to Speaker Level ---
    output_file.write("\nAggregating scores to speaker level...\n")
    agg_spk_metrics = defaultdict(dict)
    
    all_evaluators_for_agg = {**utt_evaluators, **spk_evaluators}

    for spk_id, metrics in spk_utt_scores.items():
        for metric_name, utt_scores in metrics.items():
            if not utt_scores:
                continue
            
            base_metric_name = metric_name.rsplit('_', 1)[0] if any(metric_name.endswith(f"_{s}") for s in ['control', 'all']) else metric_name
            evaluator_instance = all_evaluators_for_agg.get(base_metric_name)

            if isinstance(evaluator_instance, (Spk2ScoreEvaluator, Spk2AgeEvaluator, SpeakerEvaluator)):
                spk_level_score = utt_scores[0]
            else:
                spk_level_score = np.mean(utt_scores)
            
            agg_spk_metrics[metric_name][spk_id] = spk_level_score

    # --- 5. Report Results & Correlation ---
    results = {}
    output_file.write("\n--- Correlation Analysis (Speaker Level) ---\n")
    
    evaluated_metrics = [m for m in agg_spk_metrics.keys() if m != "spk2score"]
    ground_truth_scores = agg_spk_metrics.get("spk2score", {})

    for metric_name in sorted(evaluated_metrics):
        metric_scores = agg_spk_metrics.get(metric_name, {})
        
        common_speakers = sorted(list(ground_truth_scores.keys() & metric_scores.keys()))
        
        if len(common_speakers) < 2:
            output_file.write(f"\nCould not calculate correlation for {metric_name}: not enough common data points.\n")
            continue
            
        gt_vals = [ground_truth_scores[spk] for spk in common_speakers]
        metric_vals = [metric_scores[spk] for spk in common_speakers]
        
        valid_indices = ~np.isnan(gt_vals) & ~np.isnan(metric_vals)
        gt_vals = np.array(gt_vals)[valid_indices]
        metric_vals = np.array(metric_vals)[valid_indices]

        if len(gt_vals) < 2:
            output_file.write(f"\nCould not calculate correlation for {metric_name}: not enough common data points after removing NaNs.\n")
            continue

        pcc = np.corrcoef(gt_vals, metric_vals)[0, 1]
        output_file.write(f"Pearson Correlation between 'spk2score' and '{metric_name}': {pcc:.4f}\n")
        output_file.write(f"(Based on {len(gt_vals)} commonly scored speakers)\n")
        results[f"pcc_{metric_name}"] = pcc

    return results if results else None


if __name__ == "__main__":
    main()
