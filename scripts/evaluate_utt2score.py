import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from pathbench.evaluator import (
    Utt2ScoreEvaluator,
    PEREvaluator,
    ASREvaluator,
    DirectPEREvaluator,
    DoubleASREvaluator,
)
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.nad_evaluator import NADEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator, ArticulatoryPrecisionEvaluatorOld
from pathbench.speech_rate import PraatSpeechRateEvaluator
from pathbench.artp_double_asr_evaluator import ArtPDoubleASREvaluator
from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from pathbench.cpp_evaluator import CPPEvaluator, CPPDoubleLogEvaluator, PraatCPPEvaluator
from pathbench.wada_snr import WadaSnrEvaluator
from pathbench.dataset import Dataset
from pathbench.f0_range_evaluator import StdPitchEvaluator
from pathbench.vad import FATrimmer


def main():
    if len(sys.argv) < 2:
        print(
            f"Usage: python {sys.argv[0]} <dataset_dir1> <dataset_dir2> ...",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_dirs = sys.argv[1:]

    results = {}
    for dataset_dir in dataset_dirs:
        print(f"--- Evaluating: {dataset_dir} ---")
        try:
            print("--- Control ---")
            res_control = evaluate_dataset(dataset_dir, reference_type="control")
            if res_control is not None:
                results[f"{dataset_dir}-control"] = res_control

            if "pathological" in dataset_dir:
                print("--- All ---")
                res_all = evaluate_dataset(dataset_dir, reference_type="all")
                if res_all is not None:
                    results[f"{dataset_dir}-all"] = res_all
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
        print("\n")

    print("\n--- Evaluation Summary ---")

    datasets = list(results.keys())
    if not datasets:
        print("No results to display.")
        return

    # Get all unique metric names from all datasets
    all_metrics = sorted(
        list(set(metric for res in results.values() for metric in res.keys()))
    )

    header = "| Metric |" + "".join([f" {dataset} |" for dataset in datasets])
    print(header)
    print("|" + "---|" * (len(datasets) + 1))

    for metric in all_metrics:
        row = f"| {metric} |"
        for dataset in datasets:
            value = results[dataset].get(metric)
            if value is not None:
                row += f" {value:.4f} |"
            else:
                row += " N/A |"
        print(row)


def evaluate_dataset(dataset_dir, reference_type="control"):
    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {dataset_dir} with reference type: {reference_type}")
    use_reference = "pathological" in dataset_dir
    reference_path = (
        dataset_dir.replace("pathological", "control") if use_reference else None
    )

    if not use_reference and reference_type == "all":
        return None

    dataset = Dataset(
        dataset_dir,
        use_reference=use_reference,
        reference_path=reference_path,
        reference_type=reference_type,
    )

    # --- 2. Initialize Evaluators ---
    if not dataset.utt2score:
        print(
            f"Error: Cannot run utt2score evaluation, utt2score file not found in {dataset_dir}",
            file=sys.stderr,
        )
        return None

    fa_trimmer = FATrimmer()
    evaluators = {
        "utt2score": Utt2ScoreEvaluator(dataset.utt2score),
        "per": PEREvaluator(language=dataset.language),
        "wer": ASREvaluator("facebook/wav2vec2-base-960h"),
        "cpp": CPPEvaluator(),
        "cpp_double_log": CPPDoubleLogEvaluator(),
        "cpp_praat": PraatCPPEvaluator(),
        # Dper and double asr are not swapped
        "dper": DirectPEREvaluator(),
        "double_asr": DoubleASREvaluator(language=dataset.language),
        "artp": ArticulatoryPrecisionEvaluator(),
        "artp_old": ArticulatoryPrecisionEvaluatorOld(),
        "artp_dasr": ArtPDoubleASREvaluator(language=dataset.language),
        "wada_snr": WadaSnrEvaluator(),
        "std_pitch": StdPitchEvaluator(),
        "std_pitch_fa": StdPitchEvaluator(trimmer=fa_trimmer),
        "praat_speech_rate": PraatSpeechRateEvaluator(),
        "praat_speech_rate_fa": PraatSpeechRateEvaluator(trimmer=fa_trimmer),
    }

    if use_reference:
        evaluators["p_estoi"] = ESTOIEvaluator(
            normalization_method="RMS", centroid_ind=0, frame_deletion=True
        )
        evaluators["p_estoi_fa"] = ForcedAlignmentPESTOIEvaluator()
        evaluators["nad"] = NADEvaluator()
        evaluators["nad_fa"] = NADEvaluator(trimmer=fa_trimmer)
        # all settings for p-estoi
    # --- 3. Run Evaluation ---
    print("\nRunning evaluation...")
    scores = defaultdict(list)
    scored_utterances = 0

    for (
        utt_id,
        audio_path,
        transcription,
        reference_audios,
        start_time,
        end_time,
    ) in tqdm(dataset, desc=f"Evaluating {dataset_dir}"):
        evaluator_scores = {}
        for name, evaluator in evaluators.items():
            score = evaluator.score(
                utterance_id=utt_id,
                audio_path=audio_path,
                transcription=transcription,
                language=dataset.language,
                start_time=start_time,
                end_time=end_time,
                reference_audios=reference_audios,
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

    results = {}
    if scored_utterances > 1 and "utt2score" in scores:
        print("\nCorrelations with utt2score:")
        utt_scores = scores.pop("utt2score")
        for name, score_list in scores.items():
            if len(utt_scores) == len(score_list):
                correlation = np.corrcoef(utt_scores, score_list)[0, 1]
                results[name] = correlation
                print(f"  - {name}: {correlation:.4f}")
            else:
                print(f"  - {name}: N/A (score list length mismatch)")
    elif "utt2score" not in scores:
        print("  - 'utt2score' not found in scores, cannot calculate correlation.")
    elif scored_utterances <= 1:
        print("  - Not enough scored utterances to calculate correlation.")

    return results if results else None


if __name__ == "__main__":
    main()
