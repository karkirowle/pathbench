#!/usr/bin/env python3
"""Generate wav_hash.scp files for dataset audio integrity verification.

For each wav.scp found under the datasets root, this script SHA256-hashes every
referenced audio file and writes a sibling ``wav_hash.scp`` with the format::

    utt_id  sha256_hex_digest

Usage::

    python scripts/generate_wav_hashes.py                     # default: datasets/
    python scripts/generate_wav_hashes.py --datasets-root /path/to/datasets
    python scripts/generate_wav_hashes.py --workers 8         # parallel threads
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav_scp(wav_scp_path: str) -> list[tuple[str, str]]:
    """Return list of (utt_id, audio_path) from a wav.scp file."""
    entries = []
    with open(wav_scp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, audio_path = line.split(None, 1)
            entries.append((utt_id, audio_path))
    return entries


def generate_hashes_for_wav_scp(wav_scp_path: str, workers: int = 4) -> tuple[dict, list]:
    """Hash all audio files in a wav.scp.

    Returns (utt_id_to_hash, missing_files) where missing_files is a list of
    (utt_id, path) tuples for files that could not be read.
    """
    entries = read_wav_scp(wav_scp_path)
    # Deduplicate audio paths (multiple utt_ids can share a file via segments)
    unique_paths = {path for _, path in entries}

    path_to_hash = {}
    missing = []

    def _hash_one(path):
        return path, file_sha256(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for path in unique_paths:
            if not os.path.isfile(path):
                missing.append(path)
                continue
            futures[pool.submit(_hash_one, path)] = path

        for future in as_completed(futures):
            path, digest = future.result()
            path_to_hash[path] = digest

    utt_hashes = {}
    missing_utts = []
    for utt_id, path in entries:
        if path in path_to_hash:
            utt_hashes[utt_id] = path_to_hash[path]
        else:
            missing_utts.append((utt_id, path))

    return utt_hashes, missing_utts


def find_wav_scps(datasets_root: str) -> list[str]:
    """Find all wav.scp files under the datasets root."""
    results = []
    for dirpath, _, filenames in os.walk(datasets_root):
        if "wav.scp" in filenames:
            results.append(os.path.join(dirpath, "wav.scp"))
    results.sort()
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate wav_hash.scp files for dataset integrity checks.")
    parser.add_argument("--datasets-root", default="datasets",
                        help="Root directory to search for wav.scp files (default: datasets/)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel hashing threads (default: 4)")
    args = parser.parse_args()

    wav_scps = find_wav_scps(args.datasets_root)
    if not wav_scps:
        print(f"No wav.scp files found under {args.datasets_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_scps)} wav.scp files under {args.datasets_root}")

    total_missing = 0
    total_hashed = 0

    for wav_scp_path in wav_scps:
        dataset_dir = os.path.dirname(wav_scp_path)
        rel_dir = os.path.relpath(dataset_dir, args.datasets_root)

        utt_hashes, missing = generate_hashes_for_wav_scp(wav_scp_path, workers=args.workers)

        hash_scp_path = os.path.join(dataset_dir, "wav_hash.scp")
        # Write in the same utt_id order as wav.scp
        entries = read_wav_scp(wav_scp_path)
        with open(hash_scp_path, "w") as f:
            for utt_id, _ in entries:
                if utt_id in utt_hashes:
                    f.write(f"{utt_id} {utt_hashes[utt_id]}\n")

        n = len(utt_hashes)
        total_hashed += n
        total_missing += len(missing)

        status = f"  {rel_dir}: {n} hashed"
        if missing:
            status += f", {len(missing)} MISSING"
        print(status)

        for utt_id, path in missing:
            print(f"    MISSING: {utt_id} -> {path}", file=sys.stderr)

    print(f"\nDone. {total_hashed} files hashed across {len(wav_scps)} datasets.")
    if total_missing:
        print(f"WARNING: {total_missing} files missing!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
