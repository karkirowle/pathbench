import os
import argparse
import glob
import pandas as pd
import numpy as np
import math
from collections import defaultdict

# ==========================================
# 0. CONFIGURATION & SPEAKER WHITELIST
# ==========================================

# List provided by user. 
# NOTE: We assume this whitelist applies ONLY to the Pathological speakers 
# to filter out those with sparse data. All Control speakers are kept.
RAW_PD_WHITELIST = """
0007  0009  0011  0013  0015  0017  0019  0021  0023  0025  0027  0029  0031  0033  0037  0039  0041  0043  0046  0058  0067  0077  0079  0111  0115
0008  0010  0012  0014  0016  0018  0020  0022  0024  0026  0028  0030  0032  0035  0038  0040  0042  0044  0047  0066  0070  0078  0109  0113  0117
"""

TARGET_PD_IDS = set(RAW_PD_WHITELIST.split())

def normalize_id(id_str):
    try:
        return int(id_str)
    except ValueError:
        return str(id_str).lower().strip()

TARGET_PD_IDS_NORM = {normalize_id(s) for s in TARGET_PD_IDS}

# ==========================================
# 1. METADATA & SCORE PROCESSING
# ==========================================

def get_converted_score(raw_val):
    CONVERSION_MAP = {
        0: 3, "0": 3,
        1: 2, "1": 2,
        2: 1, "2": 1,
        3: 0, "3": 0,
        "normal": 3, "nnormal": 3,
        "mild": 2, "mild deficiency": 2, "2 mild deficiency": 2, "milddefieicence": 2, 
        "moderate": 1, "moderate deficiency": 1, "moderatedeficiency": 1,
        "severe": 0, "severe deficiency": 0
    }
    if isinstance(raw_val, str):
        key = raw_val.lower().strip()
    else:
        key = raw_val
    return CONVERSION_MAP.get(key, raw_val)

def normalize_gender(val):
    if val == "m": return 'm'
    try:
        val_float = float(val)
        if val_float == 1: return 'm'
        elif val_float == 0: return 'f'
    except: pass
    return "N/A"

def load_metadata(csv_path):
    print(f"Loading metadata from: {csv_path}")
    if not csv_path or not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR reading CSV {csv_path}: {e}")
        return {}

    df.columns = [str(c).upper().strip() for c in df.columns]

    id_col = next((c for c in df.columns if 'ID' in c or 'COD' in c or 'PARTICIPANT' in c), None)
    age_col = next((c for c in df.columns if 'AGE' in c or 'EDAD' in c), None)
    sex_col = next((c for c in df.columns if 'SEX' in c or 'GEN' in c), None)

    if not id_col: return {}
    
    meta_map = {}
    for _, row in df.iterrows():
        raw_id = row[id_col]
        norm_id = normalize_id(raw_id)
        age = row[age_col] if age_col and pd.notna(row[age_col]) else "N/A"
        raw_sex = row[sex_col] if sex_col and pd.notna(row[sex_col]) else "m"
        sex = normalize_gender(str(raw_sex))
        
        meta_map[norm_id] = {"age": age, "sex": sex}
        
    return meta_map

def load_intelligibility_scores(grbas_root):
    print(f"Loading GRBAS scores from: {grbas_root}")
    all_files = glob.glob(os.path.join(grbas_root, "*.csv"))
    if not all_files: return {}

    df_list = []
    for filename in all_files:
        try: df_list.append(pd.read_csv(filename))
        except: pass

    if not df_list: return {}

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = [c.upper() for c in combined_df.columns]
    
    name_col = next((c for c in combined_df.columns if 'TEXT_PATIENT_ID' in c), None)
    score_col = 'INTELLIGIBILITY'

    if not name_col or score_col not in combined_df.columns: return {}

    score_map = {}
    for _, row in combined_df.iterrows():
        fname = str(row[name_col])
        fname = os.path.splitext(fname)[0]
        score_map[fname] = get_converted_score(row[score_col])
        
        # Heuristic mapping
        corresponding_audio = glob.glob(os.path.join(grbas_root, "..", "audios", f"*{fname}*.wav"), recursive=True)
        if any("HC_" in os.path.basename(p).upper() for p in corresponding_audio):
            score_map["HC_" + fname] = score_map[fname]
        elif any("PD_" in os.path.basename(p).upper() for p in corresponding_audio):
            score_map["PD_" + fname] = score_map[fname]
       
    return score_map

def is_valid_score(score):
    if score == "N/A" or score is None: return False
    if isinstance(score, float) and (np.isnan(score) or math.isnan(score)): return False
    if isinstance(score, str) and score.lower() == "nan": return False
    return True

# ==========================================
# 2. SCANNING
# ==========================================

def get_sentence_id(parts):
    # Extracts task ID, e.g. "HC_S15_001" -> "S15"
    if len(parts) > 2:
        return "_".join(parts[1:-1])
    return "UNKNOWN"

def scan_dataset(dataset_root, utt_score_map, combined_metadata):
    print(f"Scanning directory: {dataset_root}...")
    
    all_wavs = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    valid_entries = []
    
    exclusion_patterns = ["A1", "A2", "A3", "I2", "E1", "E2", "E3", "PATAKA", 
                          "U2", "U1", "U3", "FREE", "O2", "I1", "I3", "O1", "O3"]

    for wav_path in all_wavs:
        filename = os.path.basename(wav_path)
        file_root = os.path.splitext(filename)[0]
        
        # 1. Exclusion Patterns
        if any(ex in file_root for ex in exclusion_patterns): continue
            
        parts = file_root.split("_")
        if len(parts) < 3: continue

        type_str = parts[0]
        raw_speaker_id = parts[-1]
        sentence_id = get_sentence_id(parts)

        # 2. Group & Whitelist Logic
        if "HC" in type_str.upper():
            group = "control"
        elif "PD" in type_str.upper():
            group = "pathological"
            if normalize_id(raw_speaker_id) not in TARGET_PD_IDS_NORM: continue
        else:
            continue

        # 3. Score Validation
        final_score = utt_score_map.get(file_root, "N/A")
        if not is_valid_score(final_score): continue

        # 4. Transcript
        txt_path = os.path.splitext(wav_path.replace("audios", "transcriptions"))[0] + ".txt"
        txt_path = txt_path.replace(raw_speaker_id, "0030")  # Fixed speaker lookup
        txt_path = txt_path.replace("HC_", "PD_")  # Use PD prefix lookup
        transcript = "<UNK>"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read().strip()
                if content: transcript = content

        # 5. Metadata Lookup
        norm_spk_id = normalize_id(raw_speaker_id)
        spk_meta = combined_metadata.get(norm_spk_id, {"age": "N/A", "sex": "m"})
        
        valid_entries.append({
            "utt_id": file_root,
            "wav_path": os.path.abspath(wav_path),
            "transcript": transcript,
            "speaker_id": raw_speaker_id,
            "sentence_id": sentence_id, # This acts as our "norm_text" / "key"
            "score": final_score,
            "group": group,
            "age": spk_meta["age"],
            "sex": spk_meta["sex"]
        })

    return valid_entries

# ==========================================
# 3. BALANCING LOGIC
# ==========================================

def get_intersection_sentences(entries, group_filter="pathological"):
    """Sentences spoken by ALL speakers in group."""
    filtered = [e for e in entries if e["group"] == group_filter]
    sent_to_spk = defaultdict(set)
    all_spk = set()
    
    for e in filtered:
        sent_to_spk[e["sentence_id"]].add(e["speaker_id"])
        all_spk.add(e["speaker_id"])
        
    intersection = set()
    for sent, speakers in sent_to_spk.items():
        if speakers == all_spk:
            intersection.add(sent)
    return intersection, all_spk

def get_valid_sentences_by_gender_count(entries, min_per_gender=2):
    """Sentences spoken by >=2 Male and >=2 Female Control speakers."""
    control_entries = [e for e in entries if e["group"] == "control"]
    sent_stats = defaultdict(lambda: {'m': set(), 'f': set()})
    
    for e in control_entries:
        g = e['sex']
        if g not in ['m', 'f']: continue 
        sent_stats[e["sentence_id"]][g].add(e["speaker_id"])
    
    valid = set()
    for sent, stats in sent_stats.items():
        if len(stats['m']) >= min_per_gender and len(stats['f']) >= min_per_gender:
            valid.add(sent)
    return valid

# ==========================================
# 4. WRITING
# ==========================================

def write_kaldi_files(entries, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort entries by utterance ID for consistent ordering
    entries = sorted(entries, key=lambda x: x['utt_id'])
    
    files = {
        "wav": open(os.path.join(output_dir, "wav.scp"), "w", encoding="utf-8"),
        "text": open(os.path.join(output_dir, "text"), "w", encoding="utf-8"),
        "utt2spk": open(os.path.join(output_dir, "utt2spk"), "w", encoding="utf-8"),
        "utt2score": open(os.path.join(output_dir, "utt2score"), "w", encoding="utf-8"),
        "utt2age": open(os.path.join(output_dir, "utt2age"), "w", encoding="utf-8"),
        "utt2gender": open(os.path.join(output_dir, "utt2gender"), "w", encoding="utf-8"),
        "spk2score": open(os.path.join(output_dir, "spk2score"), "w", encoding="utf-8"),
        "spk2age": open(os.path.join(output_dir, "spk2age"), "w", encoding="utf-8"),
        "spk2gender": open(os.path.join(output_dir, "spk2gender"), "w", encoding="utf-8"),
        "spk2utt": open(os.path.join(output_dir, "spk2utt"), "w", encoding="utf-8"), # NEW
        "spk2uttnum": open(os.path.join(output_dir, "spk2uttnum"), "w", encoding="utf-8"), # NEW
        "lang": open(os.path.join(output_dir, "language"), "w", encoding="utf-8")
    }

    files["lang"].write("es\n") 

    scores_acc = defaultdict(list)
    meta_acc = {}
    spk2utt_map = defaultdict(list) # NEW: Aggregate utterance IDs
    count = 0

    for e in entries:
        spk_id = e['speaker_id']
        utt_id = e['utt_id']
        
        files["wav"].write(f"{utt_id} {e['wav_path']}\n")
        files["text"].write(f"{utt_id} {e['transcript']}\n")
        files["utt2spk"].write(f"{utt_id} {spk_id}\n")
        files["utt2score"].write(f"{utt_id} {e['score']}\n")
        files["utt2age"].write(f"{utt_id} {e['age']}\n")
        files["utt2gender"].write(f"{utt_id} {e['sex']}\n")
        
        # Accumulate Data
        spk2utt_map[spk_id].append(utt_id)
        try: scores_acc[spk_id].append(float(e['score']))
        except: pass
        meta_acc[spk_id] = {"age": e['age'], "sex": e['sex']}
        
        count += 1

    # Write Speaker Files
    # Sort by speaker ID for consistency
    sorted_speakers = sorted(meta_acc.keys())
    
    for spk in sorted_speakers:
        # spk2score (Average)
        if spk in scores_acc and scores_acc[spk]:
            avg = sum(scores_acc[spk]) / len(scores_acc[spk])
            files["spk2score"].write(f"{spk} {avg}\n")
        
        # spk2age / spk2gender
        meta = meta_acc[spk]
        files["spk2age"].write(f"{spk} {meta['age']}\n")
        files["spk2gender"].write(f"{spk} {meta['sex']}\n")
        
        # spk2utt
        utts_str = " ".join(spk2utt_map[spk])
        files["spk2utt"].write(f"{spk} {utts_str}\n")
        
        # spk2uttnum
        utt_count = len(spk2utt_map[spk])
        files["spk2uttnum"].write(f"{spk} {utt_count}\n")
        
    for f in files.values():
        f.close()

    return len(meta_acc)

def process_neurovoz(dataset_root, grbas_root, hc_csv, pd_csv, output_dir):
    # 1. Load Data
    print("--- Loading Metadata ---")
    hc_meta = load_metadata(hc_csv)
    pd_meta = load_metadata(pd_csv)
    combined_metadata = {**hc_meta, **pd_meta}
    utt_score_map = load_intelligibility_scores(grbas_root)
    
    entries = scan_dataset(dataset_root, utt_score_map, combined_metadata)
    print(f"Total valid entries found: {len(entries)}")

    # 2. Get Constraints
    # A. Pathological Intersection
    raw_bal_sents, pd_speakers = get_intersection_sentences(entries, "pathological")
    print(f"PD Intersection Sentences: {len(raw_bal_sents)}")
    
    # B. Valid Controls (>=2M, >=2F)
    valid_control_sents = get_valid_sentences_by_gender_count(entries, 2)
    print(f"Valid Control Sentences (>=2M & >=2F): {len(valid_control_sents)}")
    
    # C. Final Balanced Set
    final_bal_sents = raw_bal_sents.intersection(valid_control_sents)
    print(f"Final Balanced Sentences: {len(final_bal_sents)}")

    # 3. Partition
    groups = ["pathological", "control"]
    dtype = "utterances" 

    for group in groups:
        subset = [e for e in entries if e["group"] == group]
        
        # --- Balanced (Intersection + Valid Control + Dedup) ---
        balanced_data = []
        seen_bal = set()
        for e in subset:
            if e["sentence_id"] in final_bal_sents:
                if group == "pathological" and e["speaker_id"] not in pd_speakers:
                    continue
                    
                key = (e["speaker_id"], e["sentence_id"])
                if key not in seen_bal:
                    balanced_data.append(e)
                    seen_bal.add(key)
        
        # Identify Balanced Speakers
        balanced_speakers = set(e["speaker_id"] for e in balanced_data)

        # --- Unbalanced (Same Speakers + Valid Control Sents) ---
        unbalanced_data = []
        for e in subset:
            if e["speaker_id"] in balanced_speakers:
                if e["sentence_id"] in valid_control_sents:
                    unbalanced_data.append(e)

        # --- All (All Speakers, No Text Constraint) ---
        all_data = subset

        # Write
        base_p = os.path.join(output_dir, "neurovoz", group, dtype)
        
        n_bal = write_kaldi_files(balanced_data, os.path.join(base_p, "balanced"))
        n_unbal = write_kaldi_files(unbalanced_data, os.path.join(base_p, "unbalanced"))
        #n_all = write_kaldi_files(all_data, os.path.join(base_p, "all"))
        
        print(f"[{group.upper()}]")
        print(f"    Balanced:   {len(balanced_data)} utts ({n_bal} spks)")
        print(f"    Unbalanced: {len(unbalanced_data)} utts ({n_unbal} spks)")
        #print(f"    All:        {len(all_data)} utts ({n_all} spks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurovoz_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/audios")
    parser.add_argument("--grbas_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/grbas")
    parser.add_argument("--hc_metadata", default="/data/group1/z40484r/datasets/neurovoz_v3/data/metadata/metadata_hc.csv")
    parser.add_argument("--pd_metadata", default="/data/group1/z40484r/datasets/neurovoz_v3/data/metadata/metadata_pd.csv")
    parser.add_argument("--output_dir", default="datasets")
    args = parser.parse_args()
    
    process_neurovoz(args.neurovoz_root, args.grbas_root, args.hc_metadata, args.pd_metadata, args.output_dir)