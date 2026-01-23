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
    """
    Helper to normalize IDs for comparison (e.g. 004 -> 4 -> 0004).
    """
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
    """
    Normalizes gender to 'm' or 'f' for Kaldi compatibility.
    """
    #if not isinstance(val, str):
    #    print(f"warning", val)
    if val == "m":
        return 'm'
    val = float(val.lower().strip())
    
    if val == 1:
        return 'm'
    elif val == 0:
        return 'f'
    else:
        return "N/A"
    

def load_metadata(csv_path):
    """
    Loads Age and Sex from a CSV file.
    Expects columns roughly matching 'ID', 'Age', 'Sex'.
    Returns: dict { normalized_id : {'age': ..., 'sex': ...} }
    """
    print(f"Loading metadata from: {csv_path}")
    if not csv_path or not os.path.exists(csv_path):
        print(f"WARNING: Metadata file not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR reading CSV {csv_path}: {e}")
        return {}

    # Normalize headers to upper case for easier search
    df.columns = [str(c).upper().strip() for c in df.columns]

    # fuzzy column matching
    id_col = next((c for c in df.columns if 'ID' in c or 'COD' in c or 'PARTICIPANT' in c), None)
    age_col = next((c for c in df.columns if 'AGE' in c or 'EDAD' in c), None)
    sex_col = next((c for c in df.columns if 'SEX' in c or 'GEN' in c), None)

    if not id_col:
        print(f"WARNING: Could not identify ID column in {csv_path}. Columns found: {df.columns}")
        return {}
    
    meta_map = {}
    for _, row in df.iterrows():
        raw_id = row[id_col]
        norm_id = normalize_id(raw_id)
        
        age = row[age_col] if age_col and pd.notna(row[age_col]) else "N/A"
        raw_sex = row[sex_col] if sex_col and pd.notna(row[sex_col]) else "m"
        
        sex = normalize_gender(str(raw_sex))
        
        meta_map[norm_id] = {
            "age": age,
            "sex": sex
        }
        
    return meta_map

def load_intelligibility_scores(grbas_root):
    print(f"Loading GRBAS scores from: {grbas_root}")
    all_files = glob.glob(os.path.join(grbas_root, "*.csv"))
    if not all_files: return {}

    df_list = []
    for filename in all_files:
        try:
            df_list.append(pd.read_csv(filename))
        except Exception: pass

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
        
        # Heuristic mapping for HC/PD prefixes
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
# 2. SCANNING & BALANCING LOGIC
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
    
    stats = {
        "exclusion": 0, "nan_score": 0, "pd_not_in_whitelist": 0, "missing_meta": 0
    }
    
    exclusion_patterns = ["A1", "A2", "A3", "I2", "E1", "E2", "E3", "PATAKA", 
                          "U2", "U1", "U3", "FREE", "O2", "I1", "I3", "O1", "O3"]

    for wav_path in all_wavs:
        filename = os.path.basename(wav_path)
        file_root = os.path.splitext(filename)[0]
        
        # 1. Exclusion Patterns
        if any(ex in file_root for ex in exclusion_patterns):
            stats["exclusion"] += 1
            continue
            
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
            if normalize_id(raw_speaker_id) not in TARGET_PD_IDS_NORM:
                stats["pd_not_in_whitelist"] += 1
                continue
        else:
            continue

        # 3. Score Validation
        final_score = utt_score_map.get(file_root, "N/A")
        if not is_valid_score(final_score):
            stats["nan_score"] += 1
            continue

        # 4. Transcript
        txt_path = os.path.splitext(wav_path.replace("audios", "transcriptions"))[0] + ".txt"
        # Always use same speaker because sometimes different speakers have different transcriptions, I noticed

        txt_path = txt_path.replace(raw_speaker_id, "0030")  # Use a fixed speaker for transcript lookup
        txt_path = txt_path.replace("HC_", "PD_")  # Use HC prefix for transcript lookup
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
            "sentence_id": sentence_id,
            "score": final_score,
            "group": group,
            "age": spk_meta["age"],
            "sex": spk_meta["sex"]
        })

    print(f"Scanned {len(valid_entries)} valid utterances.")
    print(f"Skipped {stats['pd_not_in_whitelist']} PD utterances (not in whitelist).")
    print(f"Skipped {stats['exclusion']} excluded patterns.")
    print(f"Skipped {stats['nan_score']} missing scores.")
    
    return valid_entries

def get_balanced_target_sentences(valid_entries):
    """
    1. Looks ONLY at the 'pathological' group.
    2. Finds the intersection of sentences spoken by ALL whitelisted pathological speakers.
    3. Returns this list of Sentence IDs.
    """
    pd_entries = [e for e in valid_entries if e["group"] == "pathological"]
    
    sent_to_speakers = defaultdict(set)
    all_pd_speakers = set()

    for entry in pd_entries:
        sent_to_speakers[entry["sentence_id"]].add(entry["speaker_id"])
        all_pd_speakers.add(entry["speaker_id"])

    # Debug print for specific sentence if needed
    # for sent_id, speakers in sent_to_speakers.items():
    #    if sent_id == "BARBAS":
    #        print(f" [{sent_id}]: {sorted(list(speakers))}")

    balanced_ids = []
    # Find sentences spoken by EVERY Pathological speaker
    for sent_id, speakers in sent_to_speakers.items():
        if speakers == all_pd_speakers:
            balanced_ids.append(sent_id)
            
    return set(balanced_ids), all_pd_speakers

# ==========================================
# 3. OUTPUT GENERATION
# ==========================================

def write_kaldi_files(entries, output_dir, version_name):
    target_dir = os.path.join(output_dir, version_name)
    groups = ["pathological", "control"]
    
    handles = {}
    for g in groups:
        g_dir = os.path.join(target_dir, g)
        os.makedirs(g_dir, exist_ok=True)
        handles[g] = {
            "wav_scp": open(os.path.join(g_dir, "wav.scp"), "w", encoding="utf-8"),
            "text": open(os.path.join(g_dir, "text"), "w", encoding="utf-8"),
            "utt2spk": open(os.path.join(g_dir, "utt2spk"), "w", encoding="utf-8"),
            "utt2score": open(os.path.join(g_dir, "utt2score"), "w", encoding="utf-8"),
            "utt2age": open(os.path.join(g_dir, "utt2age"), "w", encoding="utf-8"),
            "utt2gender": open(os.path.join(g_dir, "utt2gender"), "w", encoding="utf-8"),
            
            "spk2score": open(os.path.join(g_dir, "spk2score"), "w", encoding="utf-8"),
            "spk2age": open(os.path.join(g_dir, "spk2age"), "w", encoding="utf-8"),
            "spk2gender": open(os.path.join(g_dir, "spk2gender"), "w", encoding="utf-8"),
            
            "scores_acc": defaultdict(list),
            "meta_acc": {} # To store speaker level age/sex to write to spk2* later
        }

    count = 0
    unique_sents = set()
    
    for e in entries:
        g = e["group"]
        h = handles[g]
        
        # Write utterance level files
        h["wav_scp"].write(f"{e['utt_id']} {e['wav_path']}\n")
        h["text"].write(f"{e['utt_id']} {e['transcript']}\n")
        h["utt2spk"].write(f"{e['utt_id']} {e['speaker_id']}\n")
        h["utt2score"].write(f"{e['utt_id']} {e['score']}\n")
        h["utt2age"].write(f"{e['utt_id']} {e['age']}\n")
        h["utt2gender"].write(f"{e['utt_id']} {e['sex']}\n")
        
        # Accumulate for speaker level files
        try:
            h["scores_acc"][e['speaker_id']].append(float(e['score']))
        except ValueError: pass
        
        # Store metadata for speaker (last seen valid value overwrites)
        h["meta_acc"][e['speaker_id']] = {"age": e['age'], "sex": e['sex']}
            
        unique_sents.add(e["sentence_id"])
        count += 1

    # Write speaker level files
    for g in groups:
        h = handles[g]
        
        # spk2score (Average)
        for spk, scores in sorted(h["scores_acc"].items()):
            if scores:
                avg = sum(scores) / len(scores)
                h["spk2score"].write(f"{spk} {avg}\n")
        
        # spk2age and spk2gender
        for spk, meta in sorted(h["meta_acc"].items()):
            h["spk2age"].write(f"{spk} {meta['age']}\n")
            h["spk2gender"].write(f"{spk} {meta['sex']}\n")
        
        # Close all handles
        for k, v in h.items():
            if hasattr(v, 'close'):
                v.close()

    print(f"[{version_name}] Written {count} utterances.")
    return unique_sents

# ==========================================
# 4. MAIN
# ==========================================

def process_neurovoz(dataset_root, grbas_root, hc_csv, pd_csv, output_dir):
    
    # 1. Load Metadata
    print("--- Loading Metadata ---")
    hc_meta = load_metadata(hc_csv)
    pd_meta = load_metadata(pd_csv)
    # Combine (if ID collision, PD overwrites HC, though unlikely to overlap)
    combined_metadata = {**hc_meta, **pd_meta}
    print(f"Loaded metadata for {len(combined_metadata)} unique speakers.")

    # 2. Load Scores
    utt_score_map = load_intelligibility_scores(grbas_root)
    
    # 3. Scan (Apply PD Whitelist Only, Keep All Controls, Integrate Metadata)
    all_entries = scan_dataset(dataset_root, utt_score_map, combined_metadata)
    if not all_entries: return

    # 4. Determine "Balanced Sentences" based on Pathological Intersection
    target_sentences, pd_speakers = get_balanced_target_sentences(all_entries)
    
    # 5. Filter for Balanced Version
    balanced_entries = [
        e for e in all_entries 
        if e["sentence_id"] in target_sentences
    ]

    # 6. Write
    print("\n--- Generating 'neurovoz_all' (Whitelisted PDs + All Controls) ---")
    write_kaldi_files(all_entries, output_dir, "neurovoz_all")
    
    print("\n--- Generating 'neurovoz_balanced' (Strict Sentence Intersection) ---")
    final_sents = write_kaldi_files(balanced_entries, output_dir, "neurovoz_balanced")
    
    # 7. Stats
    print("\n" + "="*40)
    print("BALANCED DATASET STATISTICS")
    print("="*40)
    print(f"Pathological Speakers (Intersection Base): {len(pd_speakers)}")
    print(f"Sentences in intersection: {len(target_sentences)}")
    
    if not target_sentences:
        print("WARNING: The selected Pathological speakers have NO common sentences.")
    else:
        print("-" * 20)
        id_to_trans = {e['sentence_id']: e['transcript'] for e in balanced_entries}
        print("Final Common Sentences (Sample):")
        for sid in sorted(list(final_sents))[:10]: # Print first 10
            print(f" [{sid}]: {id_to_trans.get(sid, '')}")

    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurovoz_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/audios", 
                        help="Path to Neurovoz WAV directory")
    parser.add_argument("--grbas_root", default="/data/group1/z40484r/datasets/neurovoz_v3/data/grbas", 
                        help="Path to directory containing GRBAS CSV files")
    parser.add_argument("--hc_metadata", default="/data/group1/z40484r/datasets/neurovoz_v3/data/metadata/metadata_hc.csv", 
                        help="Path to HC metadata CSV (columns: ID, Age")
    parser.add_argument("--pd_metadata", default="/data/group1/z40484r/datasets/neurovoz_v3/data/metadata/metadata_pd.csv",
                        help="Path to PD metadata CSV (columns: ID, Age, Sex)")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_neurovoz(args.neurovoz_root, args.grbas_root, args.hc_metadata, args.pd_metadata, args.output_dir)