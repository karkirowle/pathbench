"""
Evaluate predicted speaker scores against PathBench ground truth.

Computes Pearson Correlation Coefficient (PCC) between a CSV of predicted
scores and the Kaldi-style spk2score files in the dataset directories.

All speaker IDs in the prediction CSV must appear in the ground truth and
vice versa. A mismatch causes the script to exit with an error.

# Single dataset
    python scripts/evaluate_from_csv.py \\
        --predictions results/copas_pathological_word_balanced.csv \\
        --ground-truth datasets/copas/pathological/word/balanced/spk2score

# Full benchmark (results directory mirrors datasets directory structure)
    python scripts/evaluate_from_csv.py \\
        --results-dir results/ \\
        --datasets-root datasets/

    Expected layout:
        results/
          copas/pathological/word/balanced/scores.csv
          torgo/pathological/utterances/balanced/scores.csv
          youtube/scores.csv
        datasets/
          copas/pathological/word/balanced/spk2score
          torgo/pathological/utterances/balanced/spk2score
          youtube/spk2score

    The CSV filename does not matter — only its path relative to --results-dir.

# CSV format
    speaker_id,score
    F01,2.5
    M03,4.0

    Column names are configurable via --id-column and --score-column.
    If the CSV has no header, the first column is treated as ID and second as score.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np


def get_git_hash() -> str:
    """Returns the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_predictions(csv_path: str, id_col: str, score_col: str) -> dict:
    """Load a CSV of predictions into {id: score}. Errors on duplicate IDs."""
    scores = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"Error: {csv_path} appears to be empty.", file=sys.stderr)
            sys.exit(1)

        if id_col not in reader.fieldnames or score_col not in reader.fieldnames:
            # Fall back to positional columns if named columns not found
            f.seek(0)
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None or len(first_row) < 2:
                print(f"Error: {csv_path} must have at least two columns.", file=sys.stderr)
                sys.exit(1)
            # Treat first row as header if it looks like one (non-numeric second field)
            try:
                float(first_row[1])
                rows = [first_row] + list(reader)
            except ValueError:
                rows = list(reader)
            for row in rows:
                if len(row) < 2:
                    continue
                spk_id = row[0].strip()
                try:
                    val = float(row[1].strip())
                except ValueError:
                    continue
                if spk_id in scores:
                    print(f"Error: duplicate ID '{spk_id}' in {csv_path}", file=sys.stderr)
                    sys.exit(1)
                scores[spk_id] = val
        else:
            for row in reader:
                spk_id = row[id_col].strip()
                try:
                    val = float(row[score_col].strip())
                except ValueError:
                    print(f"Error: non-numeric score for ID '{spk_id}' in {csv_path}",
                          file=sys.stderr)
                    sys.exit(1)
                if spk_id in scores:
                    print(f"Error: duplicate ID '{spk_id}' in {csv_path}", file=sys.stderr)
                    sys.exit(1)
                scores[spk_id] = val
    return scores


def load_ground_truth(spk2score_path: str) -> dict:
    """Load a Kaldi-style spk2score file into {speaker_id: score}."""
    scores = {}
    with open(spk2score_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            spk_id = parts[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            scores[spk_id] = val
    return scores


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(predictions: dict, ground_truth: dict, label: str, verbose: bool) -> float:
    """
    Compute PCC between predictions and ground truth.
    Exits with error if any ID is missing from either side.
    """
    pred_ids = set(predictions)
    gt_ids = set(ground_truth)

    missing_from_gt   = pred_ids - gt_ids
    missing_from_pred = gt_ids - pred_ids

    errors = []
    if missing_from_gt:
        errors.append(
            f"  IDs in predictions but not in ground truth ({len(missing_from_gt)}): "
            + ", ".join(sorted(missing_from_gt))
        )
    if missing_from_pred:
        errors.append(
            f"  IDs in ground truth but not in predictions ({len(missing_from_pred)}): "
            + ", ".join(sorted(missing_from_pred))
        )
    if errors:
        print(f"\nError: ID mismatch for '{label}':", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    ids = sorted(pred_ids)
    pred_vals = np.array([predictions[i] for i in ids])
    gt_vals   = np.array([ground_truth[i]  for i in ids])

    pcc = np.corrcoef(pred_vals, gt_vals)[0, 1]

    if verbose:
        print(f"\n{'ID':<30} {'Predicted':>12} {'Ground Truth':>12}")
        print("-" * 56)
        for i in ids:
            print(f"{i:<30} {predictions[i]:>12.4f} {ground_truth[i]:>12.4f}")

    return float(pcc)


# ---------------------------------------------------------------------------
# Batch mode: walk results directory, match against datasets root
# ---------------------------------------------------------------------------

def run_batch(results_dir: str, datasets_root: str, id_col: str, score_col: str,
              verbose: bool):
    results = []  # (label, n, pcc)
    all_pred = []
    all_gt   = []

    csv_files = []
    for dirpath, _, filenames in os.walk(results_dir):
        for fname in filenames:
            if fname.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, fname))

    if not csv_files:
        print(f"No CSV files found under: {results_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files.sort()

    for csv_path in csv_files:
        rel = os.path.relpath(os.path.dirname(csv_path), results_dir)
        spk2score_path = os.path.join(datasets_root, rel, "spk2score")

        if not os.path.exists(spk2score_path):
            print(f"Warning: no spk2score found at {spk2score_path} (skipping {csv_path})",
                  file=sys.stderr)
            continue

        predictions  = load_predictions(csv_path, id_col, score_col)
        ground_truth = load_ground_truth(spk2score_path)

        label = rel if rel != "." else os.path.basename(results_dir)
        pcc   = evaluate(predictions, ground_truth, label, verbose)
        n     = len(predictions)
        results.append((label, n, pcc))

        ids = sorted(predictions)
        all_pred.extend(predictions[i] for i in ids)
        all_gt.extend(ground_truth[i]  for i in ids)

    if not results:
        print("No datasets could be evaluated.", file=sys.stderr)
        sys.exit(1)

    col = max(len(r[0]) for r in results)
    col = max(col, len("Dataset"))

    header = f"{'Dataset':<{col}}  {'N':>6}  {'PCC':>8}"
    print(header)
    print("-" * len(header))
    for label, n, pcc in results:
        print(f"{label:<{col}}  {n:>6}  {pcc:>8.4f}")

    if len(results) > 1:
        overall_pcc = float(np.corrcoef(all_pred, all_gt)[0, 1])
        print("-" * len(header))
        print(f"{'Overall (pooled)':<{col}}  {len(all_pred):>6}  {overall_pcc:>8.4f}")


# ---------------------------------------------------------------------------
# Single-dataset mode
# ---------------------------------------------------------------------------

def run_single(csv_path: str, spk2score_path: str, id_col: str, score_col: str,
               verbose: bool):
    if not os.path.exists(csv_path):
        print(f"Error: predictions file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(spk2score_path):
        print(f"Error: ground truth not found: {spk2score_path}", file=sys.stderr)
        sys.exit(1)

    predictions  = load_predictions(csv_path, id_col, score_col)
    ground_truth = load_ground_truth(spk2score_path)

    label = os.path.relpath(spk2score_path)
    pcc   = evaluate(predictions, ground_truth, label, verbose)

    print(f"N:   {len(predictions)}")
    print(f"PCC: {pcc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted scores against PathBench ground truth (spk2score).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results-dir", metavar="DIR",
                      help="Directory of per-dataset CSVs (mirrors datasets directory structure)")
    mode.add_argument("--predictions", metavar="CSV",
                      help="Single predictions CSV file")

    parser.add_argument("--datasets-root", metavar="DIR",
                        help="Root datasets directory (required with --results-dir)")
    parser.add_argument("--ground-truth", metavar="FILE",
                        help="Kaldi spk2score file (required with --predictions)")
    parser.add_argument("--id-column",    default="speaker_id", metavar="NAME",
                        help="CSV column name for IDs (default: speaker_id)")
    parser.add_argument("--score-column", default="score",      metavar="NAME",
                        help="CSV column name for scores (default: score)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-speaker comparison table")

    args = parser.parse_args()

    print(f"Git commit: {get_git_hash()}")

    if args.results_dir:
        if not args.datasets_root:
            parser.error("--datasets-root is required with --results-dir")
        run_batch(args.results_dir, args.datasets_root,
                  args.id_column, args.score_column, args.verbose)
    else:
        if not args.ground_truth:
            parser.error("--ground-truth is required with --predictions")
        run_single(args.predictions, args.ground_truth,
                   args.id_column, args.score_column, args.verbose)


if __name__ == "__main__":
    main()
