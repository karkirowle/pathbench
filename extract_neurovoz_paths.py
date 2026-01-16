import os
import argparse
import glob
import pandas as pd
import numpy as np
import math

# ==========================================
# 1. SCORE PROCESSING (GRBAS / Intelligibility)
# ==========================================

def get_converted_score(raw_val):
    """
    Converts raw GRBAS/Intelligibility scores to the target scale:
    Normal -> 3
    Mild -> 2
    Moderate -> 1
    (Severe -> 0, assumed extension)
    
    If the input does not match any key, it returns the raw value AS-IS.
    """
    # Define mapping (handles both integers and string representations)
    CONVERSION_MAP = {
        # Numeric inputs
        0: 3, "0": 3,
        1: 2, "1": 2,
        2: 1, "2": 1,
        3: 0, "3": 0,
        
        # Text inputs (case insensitive handling done below)
        "normal": 3,
        "nnormal": 3,
        "mild": 2,
        "mild deficiency": 2,
        "2 mild deficiency": 2,
        "milddefieicence": 2, 
        "moderate": 1,
        "moderate deficiency": 1,
        "moderatedeficiency": 1,
        "severe": 0,
        "severe deficiency": 0
    }

    # Normalize input for lookup
    if isinstance(raw_val, str):
        key = raw_val.lower().strip()
    else:
        key = raw_val

    # Return mapped value if exists, otherwise return the raw input as-is
    return CONVERSION_MAP.get(key, raw_val)

def load_intelligibility_scores(grbas_root):
    """
    Reads all CSV files, combines them, converts scores,
    and returns: { filename_no_ext: converted_score }
    """
    print(f"Loading GRBAS scores from: {grbas_root}")
    all_files = glob.glob(os.path.join(grbas_root, "*.csv"))
    
    if not all_files:
        print("Warning: No CSV files found in GRBAS directory.")
        return {}

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        return {}

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = [c.upper() for c in combined_df.columns]
    
    # Identify columns
    name_col = next((c for c in combined_df.columns if 'TEXT_PATIENT_ID' in c), None)
    score_col = 'INTELLIGIBILITY'

    if not name_col or score_col not in combined_df.columns:
        print(f"Error: Missing columns. Found: {combined_df.columns}")
        return {}

    score_map = {}
    for _, row in combined_df.iterrows():
        fname = str(row[name_col])
        fname = os.path.splitext(fname)[0]
        
        raw_score = row[score_col]
        
        # --- APPLY CONVERSION ---
        converted_score = get_converted_score(raw_score)
        
        # Store the score (even if it's "as-is")
        score_map[fname] = converted_score

        # Check if it is HC or PD in the corresponding wav
        corresponding_audio = glob.glob(os.path.join(grbas_root, "..", "audios", f"*{fname}*.wav"), recursive=True)
        
        if any("HC_" in os.path.basename(p).upper() for p in corresponding_audio):
            score_map["HC_" + fname] = converted_score
        elif any("PD_" in os.path.basename(p).upper() for p in corresponding_audio):
            score_map["PD_" + fname] = converted_score
       
    return score_map

def is_valid_score(score):
    """
    Checks if a score is valid (not 'N/A', not None, not NaN).
    """
    if score == "N/A" or score is None:
        return False
    if isinstance(score, float) and (np.isnan(score) or math.isnan(score)):
        return False
    # If it's a string "nan" (case insensitive)
    if isinstance(score, str) and score.lower() == "nan":
        return False
    return True

# ==========================================
# 2. MAIN PROCESSING
# ==========================================

def process_neurovoz(dataset_root, grbas_root, output_dir):
    
    # 1. Load & Convert Scores
    utt_score_map = load_intelligibility_scores(grbas_root)
    
    groups = ["pathological", "control"]
    handles = {}

    # Setup Output Files
    for g in groups:
        out_path = os.path.join(output_dir, "neurovoz", g)
        os.makedirs(out_path, exist_ok=True)
        handles[g] = {
            "wav_scp": open(os.path.join(out_path, "wav.scp"), "w", encoding="utf-8"),
            "text": open(os.path.join(out_path, "text"), "w", encoding="utf-8"),
            "utt2spk": open(os.path.join(out_path, "utt2spk"), "w", encoding="utf-8"),
            "utt2score": open(os.path.join(out_path, "utt2score"), "w", encoding="utf-8"),
            "spk2score": open(os.path.join(out_path, "spk2score"), "w", encoding="utf-8"),
            "spk_scores_accumulator": {} 
        }

    print(f"Scanning directory: {dataset_root}...")
    
    all_wavs = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    
    count = 0
    skipped_exclusion_count = 0
    skipped_nan_count = 0
    
    # Exclude these patterns
    exclusion_patterns = [
        "A1", "A2", "A3", 
        "I2", "E1", "E2", "E3", "PATAKA", 
        "U2", "U1", "U3", "FREE", 
        "O2", "I1", "I3", "O1", "O3"
    ]

    for wav_path in all_wavs:
        filename = os.path.basename(wav_path)
        file_root = os.path.splitext(filename)[0]
        
        # --- FILTER 1: EXCLUSION PATTERNS ---
        if any(ex in file_root for ex in exclusion_patterns):
            skipped_exclusion_count += 1
            continue
            
        parts = file_root.split("_")
        if len(parts) < 3:
            continue

        # Metadata
        type_str = parts[0]
        speaker_id = parts[-1]

        # Group
        if "HC" in type_str.upper():
            group = "control"
        elif "PD" in type_str.upper():
            group = "pathological"
        else:
            continue

        # Get Score (Converted or As-Is)
        # If file not in CSV, default to "N/A"
        final_score = utt_score_map.get(file_root, "N/A")

        # --- FILTER 2: REMOVE NAN / N/A ---
        # If score is invalid, skip this utterance entirely
        if not is_valid_score(final_score):
            print(f"Skipping {file_root} due to invalid score: {final_score}")
            skipped_nan_count += 1
            continue

        # Transcript
        txt_path = os.path.splitext(wav_path.replace("audios", "transcriptions"))[0] + ".txt"
        transcript = "<UNK>"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read().strip()
                if content: transcript = content

        # Write to files
        utt_id = file_root 
        h = handles[group]
        
        h["wav_scp"].write(f"{utt_id} {os.path.abspath(wav_path)}\n")
        h["text"].write(f"{utt_id} {transcript}\n")
        h["utt2spk"].write(f"{utt_id} {speaker_id}\n")
        h["utt2score"].write(f"{utt_id} {final_score}\n")
        
        # Accumulate for spk2score (Average of converted scores)
        if speaker_id not in h["spk_scores_accumulator"]:
            h["spk_scores_accumulator"][speaker_id] = []
        
        try:
            val = float(final_score)
            h["spk_scores_accumulator"][speaker_id].append(val)
        except ValueError:
            pass 

        count += 1

    # Finalize spk2score
    for g in groups:
        h = handles[g]
        accumulator = h["spk_scores_accumulator"]
        
        for spk, scores in sorted(accumulator.items()):
            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)
                h["spk2score"].write(f"{spk} {avg_score:.2f}\n")
            else:
                # Skip writing speakers who have no valid utterances
                pass
        
        for key in h:
            if hasattr(h[key], "close"):
                h[key].close()

    print(f"Processing complete.")
    print(f"Processed {count} valid utterances.")
    print(f"Skipped {skipped_exclusion_count} excluded patterns.")
    print(f"Skipped {skipped_nan_count} due to missing/NaN scores.")
    print(f"Output saved to: {os.path.join(output_dir, 'neurovoz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurovoz_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/audios", 
                        help="Path to Neurovoz WAV directory")
    parser.add_argument("--grbas_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/grbas", 
                        help="Path to directory containing GRBAS CSV files")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_neurovoz(args.neurovoz_root, args.grbas_root, args.output_dir)