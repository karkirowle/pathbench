import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import inspect
import hashlib

from pathbench.evaluator import (
    Spk2ScoreEvaluator,
    LookupEvaluator, ReferenceFreeEvaluator, ReferenceTxtEvaluator,
    ReferenceAudioEvaluator, ReferenceTxtAndAudioEvaluator,
    ReferenceFreeSpeakerEvaluator, LanguageAwareSpeakerEvaluator,
    TrimmedReferenceFreeEvaluator, TrimmedReferenceFreeSpeakerEvaluator,
    TrimmedLanguageAwareSpeakerEvaluator, load_audios,
)
from pathbench.asr_evaluators import PEREvaluator, DirectPEREvaluator, DoubleASREvaluator
from pathbench.f0_range_evaluator import StdPitchEvaluator
from pathbench.reference_evaluator import ESTOIEvaluator
from pathbench.nad_evaluator import NADEvaluator, TrimmedNADEvaluator
from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator, PhoneticConfidenceEvaluator
from pathbench.speech_rate import PraatSpeechRateEvaluator
from pathbench.dataset import Dataset
from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from pathbench.cpp_evaluator import CPPEvaluator
from pathbench.wada_snr import WadaSnrEvaluator
from pathbench.age_evaluator import Spk2AgeEvaluator
from pathbench.artp_double_asr_evaluator import ArtPDoubleASREvaluator
from pathbench.vsa_evaluator import VSAEvaluator
from pathbench.vad import FATrimmer
from pathbench.utils import write_correlation_table

# Metrics where partial speaker coverage is expected and not an error.
# spk2age: not all speakers have age data.
# wada_snr: can fail for some speakers.
PARTIAL_COVERAGE_ALLOWLIST = {"spk2age", "wada_snr"}

# Evaluator keys that require reference audio and are run separately.
REF_EVALUATOR_NAMES = ["p_estoi", "p_estoi_fa", "nad", "nad_fa"]

# Evaluator types that produce a single score per speaker (not averaged over utterances).
_SINGLE_SCORE_TYPES = (
    LookupEvaluator,
    ReferenceFreeSpeakerEvaluator,
    LanguageAwareSpeakerEvaluator,
    TrimmedReferenceFreeSpeakerEvaluator,
    TrimmedLanguageAwareSpeakerEvaluator,
)

SUMMARY_METRICS = [
    "cpp", "per", "dper", "artp", "artp_old", "artp_double_asr", "double_asr",
    "p_estoi_control", "p_estoi_all", "p_estoi_fa_control", "p_estoi_fa_all",
    "nad_control", "nad_all", "wada_snr", "spk2age", "vsa", "std_pitch",
    "cpp_fa", "nad_fa_control", "nad_fa_all", "vsa_fa", "std_pitch_fa",
    "praat_speech_rate", "praat_speech_rate_fa",
]


def _base_metric(metric_name: str) -> str:
    """Strips '_control' / '_all' suffix to recover the base evaluator key."""
    for suffix in ("_control", "_all"):
        if metric_name.endswith(suffix):
            return metric_name[: -len(suffix)]
    return metric_name


def get_class_hash(instance) -> str:
    """SHA256 hash of an evaluator's class source code, for reproducibility tracking."""
    try:
        return hashlib.sha256(inspect.getsource(instance.__class__).encode()).hexdigest()
    except (TypeError, OSError):
        return "N/A"


def build_evaluators(base_dataset, trimmer):
    """Instantiates all utterance- and speaker-level evaluators for a dataset.

    Returns:
        utt_evaluators: dict of utterance-level evaluators (including spk2score)
        spk_evaluators: dict of speaker-level evaluators
    """
    utt_evaluators = {
        "spk2score":          Spk2ScoreEvaluator(base_dataset.spk2score, base_dataset.utt2spk),
        "cpp":                CPPEvaluator(),
        "per":                PEREvaluator(language=base_dataset.language),
        "dper":               DirectPEREvaluator(),
        "artp":               ArticulatoryPrecisionEvaluator(),
        "artp_old":           PhoneticConfidenceEvaluator(),
        "artp_double_asr":    ArtPDoubleASREvaluator(language=base_dataset.language),
        "double_asr":         DoubleASREvaluator(language=base_dataset.language),
        "p_estoi":            ESTOIEvaluator(normalization_method="RMS", centroid_ind=0, frame_deletion=True),
        "p_estoi_fa":         ForcedAlignmentPESTOIEvaluator(),
        "nad":                NADEvaluator(),
        "wada_snr":           WadaSnrEvaluator(),
        "std_pitch":          StdPitchEvaluator(),
        "cpp_fa":             TrimmedReferenceFreeEvaluator(inner=CPPEvaluator(), trimmer=trimmer),
        "nad_fa":             TrimmedNADEvaluator(trimmer=trimmer),
        "std_pitch_fa":       TrimmedReferenceFreeEvaluator(inner=StdPitchEvaluator(), trimmer=trimmer),
        "praat_speech_rate":  PraatSpeechRateEvaluator(),
        "praat_speech_rate_fa": TrimmedReferenceFreeEvaluator(inner=PraatSpeechRateEvaluator(), trimmer=trimmer),
    }
    if base_dataset.spk2age:
        utt_evaluators["spk2age"] = Spk2AgeEvaluator(base_dataset.spk2age, base_dataset.utt2spk)

    spk_evaluators = {
        "vsa":    VSAEvaluator(),
        "vsa_fa": TrimmedLanguageAwareSpeakerEvaluator(inner=VSAEvaluator(), trimmer=trimmer),
    }
    return utt_evaluators, spk_evaluators


def _dispatch_utt_score(evaluator, utt_id, audio_path, transcription, language, start_time, end_time):
    """Routes an utterance score call to the correct evaluator method signature."""
    if isinstance(evaluator, LookupEvaluator):
        return evaluator.score(utt_id)
    elif isinstance(evaluator, ReferenceFreeEvaluator):
        return evaluator.score(utt_id, audio_path, start_time, end_time)
    elif isinstance(evaluator, ReferenceTxtEvaluator):
        return evaluator.score(utt_id, audio_path, transcription, language, start_time, end_time)
    else:
        return evaluator.score(
            utterance_id=utt_id, audio_path=audio_path,
            reference_audios=None, transcription=transcription,
            language=language, start_time=start_time, end_time=end_time,
        )


def _dispatch_ref_score(evaluator, utt_id, audio_path, transcription, language, reference_audios, start_time, end_time):
    """Routes a reference-based utterance score call to the correct evaluator method signature."""
    if isinstance(evaluator, ReferenceAudioEvaluator):
        return evaluator.score(utt_id, audio_path, reference_audios, start_time, end_time)
    elif isinstance(evaluator, ReferenceTxtAndAudioEvaluator):
        return evaluator.score(utt_id, audio_path, transcription, language, reference_audios, start_time, end_time)
    else:
        return evaluator.score(
            utterance_id=utt_id, audio_path=audio_path,
            reference_audios=reference_audios, transcription=transcription,
            language=language, start_time=start_time, end_time=end_time,
        )


def run_utterance_evaluators(base_dataset, non_ref_evaluators):
    """Scores each utterance with all non-reference evaluators.

    Returns:
        spk_utt_scores: speaker_id -> metric_name -> [scores]
    """
    spk_utt_scores = defaultdict(lambda: defaultdict(list))

    for utt_id, audio_path, transcription, _, start_time, end_time in tqdm(
        base_dataset, desc="Non-reference evaluation"
    ):
        speaker_id = base_dataset.utt2spk.get(utt_id)
        if not speaker_id:
            continue

        for name, evaluator in non_ref_evaluators.items():
            try:
                score = _dispatch_utt_score(
                    evaluator, utt_id, audio_path, transcription,
                    base_dataset.language, start_time, end_time,
                )
            except Exception as e:
                print(f"Evaluator '{name}' failed for {utt_id}: {e}")
                score = None

            if score is not None:
                spk_utt_scores[speaker_id][name].append(score)

    return spk_utt_scores


def run_reference_evaluators(dataset_dir, utt_evaluators, spk_utt_scores):
    """Scores utterances with reference-based evaluators for each reference type.

    Updates spk_utt_scores in place.

    Note: For the '/all' dataset variant, evaluation aborts on the first missing
    reference audio (structural limitation — many speakers have no control counterpart).
    For balanced/unbalanced datasets, utterances without references are simply skipped.
    """
    reference_path = dataset_dir.replace("pathological", "control")
    is_all_dataset = "/all" in dataset_dir

    for ref_type in ["control", "all"]:
        try:
            ref_dataset = Dataset(
                dataset_dir,
                use_reference=True,
                reference_path=reference_path,
                reference_type=ref_type,
            )
        except FileNotFoundError as e:
            print(f"Could not load reference dataset (type={ref_type}): {e}", file=sys.stderr)
            continue

        stop = False
        for utt_id, audio_path, transcription, reference_audios, start_time, end_time in tqdm(
            ref_dataset, desc=f"Reference evaluation (type={ref_type})"
        ):
            if stop:
                break

            speaker_id = ref_dataset.utt2spk.get(utt_id)
            if not speaker_id:
                continue

            if not reference_audios:
                if is_all_dataset:
                    print(
                        f"No reference audio for {utt_id} in 'all' dataset "
                        f"(ref_type={ref_type}). Stopping reference evaluation."
                    )
                    stop = True
                continue

            for name in REF_EVALUATOR_NAMES:
                evaluator = utt_evaluators[name]
                try:
                    score = _dispatch_ref_score(
                        evaluator, utt_id, audio_path, transcription,
                        ref_dataset.language, reference_audios, start_time, end_time,
                    )
                except Exception as e:
                    print(f"Evaluator '{name}' ({ref_type}) failed for {utt_id}: {e}")
                    score = None

                if score is None:
                    if is_all_dataset:
                        print(
                            f"Evaluator '{name}' ({ref_type}) returned None for {utt_id} "
                            "in 'all' dataset. Stopping reference evaluation."
                        )
                        stop = True
                        break
                else:
                    spk_utt_scores[speaker_id][f"{name}_{ref_type}"].append(score)


def run_speaker_evaluators(base_dataset, spk_evaluators, spk_utt_scores):
    """Scores each speaker with speaker-level evaluators.

    Updates spk_utt_scores in place (score stored as a single-element list
    for consistency with utterance-level scores).
    """
    spk_audio_files = defaultdict(list)
    spk_transcriptions = defaultdict(list)
    for utt_id, audio_path, transcription, _, start_time, end_time in base_dataset:
        speaker_id = base_dataset.utt2spk.get(utt_id)
        if speaker_id:
            spk_audio_files[speaker_id].append((audio_path, start_time, end_time))
            spk_transcriptions[speaker_id].append(transcription)

    for spk_id in tqdm(spk_audio_files, desc="Speaker-level evaluation"):
        for name, evaluator in spk_evaluators.items():
            if isinstance(evaluator, (TrimmedReferenceFreeSpeakerEvaluator, TrimmedLanguageAwareSpeakerEvaluator)):
                score = evaluator.score(spk_audio_files[spk_id], spk_transcriptions[spk_id], base_dataset.language)
            elif isinstance(evaluator, (ReferenceFreeSpeakerEvaluator, LanguageAwareSpeakerEvaluator)):
                audios = load_audios(spk_audio_files[spk_id])
                if not audios:
                    score = None
                elif isinstance(evaluator, LanguageAwareSpeakerEvaluator):
                    score = evaluator._score_audio_list(audios, base_dataset.language)
                else:
                    score = evaluator._score_audio_list(audios)
            else:
                score = evaluator.score(
                    audio_files=spk_audio_files[spk_id],
                    transcriptions=spk_transcriptions[spk_id],
                    language=base_dataset.language,
                )
            if score is not None:
                spk_utt_scores[spk_id][name] = [score]


def aggregate_to_speaker_level(spk_utt_scores, all_evaluators):
    """Averages utterance scores per speaker to produce a single speaker-level score.

    Args:
        spk_utt_scores: speaker_id -> metric_name -> [utterance scores]
        all_evaluators: metric_name -> evaluator instance (used to detect lookup/speaker types)

    Returns:
        agg_spk_metrics: metric_name -> speaker_id -> aggregated score

    Uses np.nanmean so that individual NaN utterance scores (returned as floats
    rather than None by some evaluators) do not invalidate the entire speaker.
    Speakers for whom ALL utterance scores are NaN are excluded with a warning,
    unless the metric is in PARTIAL_COVERAGE_ALLOWLIST.
    """
    agg_spk_metrics = defaultdict(dict)

    for spk_id, metrics in spk_utt_scores.items():
        for metric_name, utt_scores in metrics.items():
            if not utt_scores:
                continue

            evaluator = all_evaluators.get(_base_metric(metric_name))
            if isinstance(evaluator, _SINGLE_SCORE_TYPES):
                spk_score = utt_scores[0]
            else:
                spk_score = np.nanmean(utt_scores)

            if np.isnan(spk_score):
                if _base_metric(metric_name) not in PARTIAL_COVERAGE_ALLOWLIST:
                    print(
                        f"Warning: all utterance scores are NaN for speaker '{spk_id}', "
                        f"metric '{metric_name}'. Speaker excluded from correlation.",
                        file=sys.stderr,
                    )
                continue

            agg_spk_metrics[metric_name][spk_id] = spk_score

    return agg_spk_metrics


def compute_correlations(agg_spk_metrics, output_file):
    """Computes Pearson correlation between spk2score and each metric at speaker level.

    Args:
        agg_spk_metrics: metric_name -> speaker_id -> aggregated score
        output_file: file-like object for logging results

    Returns:
        results: {'pcc_<metric_name>': float}

    Warns to stderr if a non-allowlisted metric covers fewer speakers than spk2score,
    which would indicate unexpected speaker dropout.
    """
    results = {}
    ground_truth = agg_spk_metrics.get("spk2score", {})
    n_gt = len(ground_truth)

    output_file.write("\n--- Correlation Analysis (Speaker Level) ---\n")

    for metric_name in sorted(m for m in agg_spk_metrics if m != "spk2score"):
        metric_scores = agg_spk_metrics[metric_name]
        common = sorted(ground_truth.keys() & metric_scores.keys())

        if len(common) < n_gt and _base_metric(metric_name) not in PARTIAL_COVERAGE_ALLOWLIST:
            print(
                f"Warning: '{metric_name}' covers {len(common)}/{n_gt} speakers. "
                "Some speakers have no valid score.",
                file=sys.stderr,
            )

        if len(common) < 2:
            output_file.write(
                f"  {metric_name}: not enough common speakers ({len(common)}) — skipped.\n"
            )
            continue

        gt_vals = np.array([ground_truth[s] for s in common])
        metric_vals = np.array([metric_scores[s] for s in common])

        # Safety net: with nanmean aggregation upstream, NaNs should not reach here.
        # Uncomment if debugging unexpected NaN values in correlation inputs:
        # valid = ~np.isnan(gt_vals) & ~np.isnan(metric_vals)
        # gt_vals, metric_vals = gt_vals[valid], metric_vals[valid]

        if len(gt_vals) < 2:
            output_file.write(
                f"  {metric_name}: not enough valid speakers after NaN removal — skipped.\n"
            )
            continue

        pcc = np.corrcoef(gt_vals, metric_vals)[0, 1]
        output_file.write(
            f"  PCC (spk2score vs {metric_name}): {pcc:.4f}  [{len(gt_vals)} speakers]\n"
        )
        results[f"pcc_{metric_name}"] = pcc

    return results


def evaluate_dataset(dataset_dir, output_file):
    """Runs the full evaluation pipeline for a single dataset directory."""
    output_file.write(f"Loading dataset: {dataset_dir}\n")
    try:
        base_dataset = Dataset(dataset_dir, use_reference=False)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return None

    if not base_dataset.spk2score or not base_dataset.utt2spk:
        print(
            f"Cannot evaluate {dataset_dir}: spk2score or utt2spk not found.",
            file=sys.stderr,
        )
        return None

    trimmer = FATrimmer()
    utt_evaluators, spk_evaluators = build_evaluators(base_dataset, trimmer)
    all_evaluators = {**utt_evaluators, **spk_evaluators}

    output_file.write("\n--- Evaluator Hashes ---\n")
    for name, evaluator in all_evaluators.items():
        output_file.write(f"  {name}: {get_class_hash(evaluator)}\n")

    non_ref_evaluators = {k: v for k, v in utt_evaluators.items() if k not in REF_EVALUATOR_NAMES}

    output_file.write("\nRunning non-reference utterance evaluation...\n")
    spk_utt_scores = run_utterance_evaluators(base_dataset, non_ref_evaluators)

    output_file.write("\nRunning reference-based utterance evaluation...\n")
    run_reference_evaluators(dataset_dir, utt_evaluators, spk_utt_scores)

    output_file.write("\nRunning speaker-level evaluation...\n")
    run_speaker_evaluators(base_dataset, spk_evaluators, spk_utt_scores)

    output_file.write("\nAggregating utterance scores to speaker level...\n")
    agg_spk_metrics = aggregate_to_speaker_level(spk_utt_scores, all_evaluators)

    return compute_correlations(agg_spk_metrics, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speaker intelligibility metrics against spk2score ground truth."
    )
    parser.add_argument("dataset_dirs", nargs="+", help="Dataset directories to evaluate.")
    args = parser.parse_args()

    dataset_name = args.dataset_dirs[0].replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"results_10/{dataset_name}_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        output_file.write(f"Evaluation run: {timestamp}\n\n")

        all_results = {}
        for dataset_dir in args.dataset_dirs:
            output_file.write(f"\n--- Dataset: {dataset_dir} ---\n")
            try:
                result = evaluate_dataset(dataset_dir, output_file)
                if result:
                    all_results[dataset_dir] = result
            except FileNotFoundError as e:
                print(f"Error: {e}", file=sys.stderr)
            output_file.write("\n")

        write_correlation_table(output_file, all_results, SUMMARY_METRICS)


if __name__ == "__main__":
    main()
