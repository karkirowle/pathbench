import os
import argparse
import glob
import re
import string
from collections import defaultdict

# ==========================================
# 0. CONFIGURATION
# ==========================================

SENTENCE_TEXTS = {
    "S1": "Wil je liever de thee of de borrel?", 
    "S2": "Na nieuwjaar was hij weeral hier." 
}

IGNORE_LABELS = ["", "sil", "<sil>", "sp", "#", "nsp"]

# ==========================================
# 1. HELPERS
# ==========================================

def get_speaker_from_filename(filename):
    """Extracts speaker ID (first part of filename before underscore)."""
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0].upper()
    return None

def normalize_text(text):
    if not text: return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()

def _clean(line):
    return line.strip().replace('"', '')

def parse_textgrid_intervals(textgrid_path):
    """
    Robust parser for Praat TextGrids.
    """
    intervals = []
    try:
        with open(textgrid_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines()]
        if not lines: return []

        is_long_format = any("xmin =" in l for l in lines[:20])
        target_intervals = []
        fallback_intervals = []
        found_target = False

        if is_long_format:
            current_tier_name = ""
            in_interval = False
            xmin, xmax, text = None, None, None
            for line in lines:
                if line.startswith("name ="):
                    current_tier_name = _clean(line.split("=")[1])
                elif line.startswith("intervals ["):
                    in_interval = True
                    xmin, xmax, text = None, None, None
                
                if in_interval:
                    if line.startswith("xmin ="): xmin = float(line.split("=")[1].strip())
                    elif line.startswith("xmax ="): xmax = float(line.split("=")[1].strip())
                    elif line.startswith("text ="):
                        match = re.search(r'text = "(.*)"', line)
                        if match: text = match.group(1).strip()
                    
                    if xmin is not None and xmax is not None and text is not None:
                        if text.lower() not in IGNORE_LABELS:
                            data = (xmin, xmax, text)
                            if current_tier_name.lower() == "target":
                                target_intervals.append(data)
                                found_target = True
                            elif not found_target:
                                fallback_intervals.append(data)
                        in_interval = False
                        xmin, xmax, text = None, None, None
        else:
            iterator = iter(lines)
            try:
                while True:
                    if "<exists>" in next(iterator): break
                num_tiers = int(_clean(next(iterator)))
                for _ in range(num_tiers):
                    tier_type = _clean(next(iterator))
                    tier_name = _clean(next(iterator))
                    next(iterator); next(iterator) 
                    num_items = int(_clean(next(iterator)))
                    if tier_type == "IntervalTier":
                        current_tier_data = []
                        for _ in range(num_items):
                            t_min = float(_clean(next(iterator)))
                            t_max = float(_clean(next(iterator)))
                            t_text = _clean(next(iterator))
                            if t_text.lower() not in IGNORE_LABELS:
                                current_tier_data.append((t_min, t_max, t_text))
                        if tier_name.lower() == "target":
                            target_intervals = current_tier_data
                            found_target = True
                        elif not found_target and not fallback_intervals:
                            fallback_intervals = current_tier_data
                    else:
                        for _ in range(num_items): next(iterator); next(iterator)
            except StopIteration: pass
            except ValueError: pass
        return target_intervals if found_target else fallback_intervals
    except Exception as e:
        print(f"Warning: Error parsing {textgrid_path}: {e}")
        return []

# ==========================================
# 2. METADATA LOADING
# ==========================================

def load_metadata_scores(metadata_path):
    scores_std = {}
    scores_a18 = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                f.readline() # Skip header
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 5:
                        spk = parts[0].strip().upper()
                        scores_a18[spk] = parts[1].strip() # A18 Specific
                        scores_std[spk] = parts[4].strip() # Standard
        except Exception as e: print(f"Meta Error: {e}")
    return scores_std, scores_a18

def load_metadata_age(age_path):
    ages = {}
    if age_path and os.path.exists(age_path):
        try:
            with open(age_path, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 2:
                        spk = parts[0].strip().upper()
                        ages[spk] = parts[1].strip()
        except Exception as e: print(f"Age Error: {e}")
    return ages

def load_metadata_gender(gender_path):
    genders = {}
    if gender_path and os.path.exists(gender_path):
        try:
            with open(gender_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 2:
                        spk = parts[0].strip().upper()
                        gen_val = parts[1].strip().lower()
                        if gen_val.startswith('f') or gen_val == 'woman' or gen_val == 'female':
                            norm_gen = 'f'
                        else:
                            norm_gen = 'm'
                        genders[spk] = norm_gen
        except Exception as e: print(f"Gender Error: {e}")
    return genders

# ==========================================
# 3. SCANNING
# ==========================================

def scan_copas(copas_root, scores_std, scores_a18, ages, genders):
    dir_s1 = os.path.join(copas_root, "S1")
    dir_s2 = os.path.join(copas_root, "S2")
    dir_dia = os.path.join(copas_root, "DIA")
    all_entries = []

    # --- Sentences ---
    print("Scanning Sentences (S1/S2)...")
    for folder_path, folder_name in [(dir_s1, "S1"), (dir_s2, "S2")]:
        if not os.path.isdir(folder_path): continue
        for wav_path in glob.glob(os.path.join(folder_path, "*.wav")):
            filename = os.path.basename(wav_path)
            speaker_id = get_speaker_from_filename(filename)
            if not speaker_id: continue
            
            group = "control" if speaker_id.startswith("N") else "pathological"
            transcription = SENTENCE_TEXTS.get(folder_name, "")
            sc = "N/A" if group == "control" else scores_std.get(speaker_id, "N/A")
            
            all_entries.append({
                "utt_id": f"{speaker_id}_{folder_name}",
                "wav_id": f"{speaker_id}_{folder_name}",
                "wav_path": os.path.abspath(wav_path),
                "transcript": transcription,
                "norm_text": normalize_text(transcription),
                "speaker": speaker_id,
                "gender": genders.get(speaker_id, "m"),
                "group": group,
                "dtype": "utterances",
                "sentence_label": folder_name,
                "score": sc,
                "age": ages.get(speaker_id, "N/A"),
                "has_segments": False,
                "is_a18": False
            })

    # --- Words ---
    print("Scanning Words (DIA)...")
    if os.path.isdir(dir_dia):
        for tg_path in glob.glob(os.path.join(dir_dia, "*.TextGrid")):
            base_name = os.path.splitext(os.path.basename(tg_path))[0]
            wav_path = os.path.join(dir_dia, base_name + ".wav")
            if not os.path.exists(wav_path): continue

            speaker_id = get_speaker_from_filename(base_name)
            if not speaker_id: continue

            group = "control" if speaker_id.startswith("N") else "pathological"
            intervals = parse_textgrid_intervals(tg_path)
            is_a18 = "_A18" in base_name.upper()
            
            if group == "control": sc = "N/A"
            elif is_a18: sc = scores_a18.get(speaker_id, "N/A")
            else: sc = scores_std.get(speaker_id, "N/A")

            count = 0
            for start, end, text in intervals:
                count += 1
                utt_id = f"{base_name}_{count:03d}"
                all_entries.append({
                    "utt_id": utt_id,
                    "wav_id": base_name,
                    "wav_path": os.path.abspath(wav_path),
                    "start": start,
                    "end": end,
                    "transcript": text,
                    "norm_text": normalize_text(text),
                    "speaker": speaker_id,
                    "gender": genders.get(speaker_id, "m"),
                    "group": group,
                    "dtype": "word",
                    "score": sc,
                    "age": ages.get(speaker_id, "N/A"),
                    "has_segments": True,
                    "is_a18": is_a18
                })
    return all_entries

# ==========================================
# 4. LOGIC: VALIDATION & FILTERING
# ==========================================

def get_complete_sentence_speakers(entries):
    spk_sents = defaultdict(set)
    for e in entries:
        if e["dtype"] == "utterances":
            spk_sents[e["speaker"]].add(e["sentence_label"])
    complete = set()
    for spk, labels in spk_sents.items():
        if "S1" in labels and "S2" in labels:
            complete.add(spk)
    return complete

def get_valid_texts_by_gender_count(entries, dtype_filter, min_per_gender=2):
    """
    Returns texts spoken by >= min Males AND >= min Females (Controls).
    """
    control_entries = [e for e in entries if e["group"] == "control" and e["dtype"] == dtype_filter]
    text_stats = defaultdict(lambda: {'m': set(), 'f': set()})
    
    for e in control_entries:
        g = e['gender']
        text_stats[e["norm_text"]][g].add(e["speaker"])
    
    valid_texts = set()
    for text, stats in text_stats.items():
        if len(stats['m']) >= min_per_gender and len(stats['f']) >= min_per_gender:
            valid_texts.add(text)
    return valid_texts

# ==========================================
# 5. WRITING OUTPUTS
# ==========================================

def write_kaldi_subset(entries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "wav": open(os.path.join(out_dir, "wav.scp"), "w"),
        "text": open(os.path.join(out_dir, "text"), "w"),
        "utt2spk": open(os.path.join(out_dir, "utt2spk"), "w"),
        "segments": open(os.path.join(out_dir, "segments"), "w"),
        "lang": open(os.path.join(out_dir, "language"), "w"),
        "score": open(os.path.join(out_dir, "spk2score"), "w"),
        "age": open(os.path.join(out_dir, "spk2age"), "w"),
        "gender": open(os.path.join(out_dir, "spk2gender"), "w")
    }
    
    files["lang"].write("nl\n")
    written_wavs = set()
    spk_data = {}
    count_segments = 0

    for e in entries:
        if e["wav_id"] not in written_wavs:
            files["wav"].write(f"{e['wav_id']} {e['wav_path']}\n")
            written_wavs.add(e["wav_id"])
        
        if e["has_segments"]:
            files["segments"].write(f"{e['utt_id']} {e['wav_id']} {e['start']:.3f} {e['end']:.3f}\n")
            count_segments += 1
        
        files["text"].write(f"{e['utt_id']} {e['transcript']}\n")
        files["utt2spk"].write(f"{e['utt_id']} {e['speaker']}\n")
        
        spk_data[e['speaker']] = {"score": e['score'], "age": e['age'], "gender": e['gender']}

    for spk in sorted(spk_data.keys()):
        files["score"].write(f"{spk} {spk_data[spk]['score']}\n")
        files["age"].write(f"{spk} {spk_data[spk]['age']}\n")
        files["gender"].write(f"{spk} {spk_data[spk]['gender']}\n")

    files["segments"].close()
    if count_segments == 0:
        os.remove(os.path.join(out_dir, "segments"))
        
    for f in files.values():
        if not f.closed: f.close()
        
    return len(spk_data)

def process_copas(copas_root, metadata_path, age_path, gender_path, output_dir):
    # 1. Load Meta & Scan
    scores_std, scores_a18 = load_metadata_scores(metadata_path)
    ages = load_metadata_age(age_path)
    genders = load_metadata_gender(gender_path)
    
    entries = scan_copas(copas_root, scores_std, scores_a18, ages, genders)
    print(f"Total entries found: {len(entries)}")
    
    # 2. Pre-calculate Criteria
    complete_sentence_speakers = get_complete_sentence_speakers(entries)
    
    groups = ["pathological", "control"]
    dtypes = ["word", "utterances"]

    for dtype in dtypes:
        # --- VALIDATION: Identify Valid Texts (>=2M & >=2F Controls) ---
        # Used for filtering both Balanced and Unbalanced to ensure control availability
        valid_control_texts = get_valid_texts_by_gender_count(entries, dtype, 2)
        print(f"\n--- {dtype.upper()} Valid Texts (>=2M, >=2F Controls): {len(valid_control_texts)} ---")

        # --- Balanced Set Construction (Pathological Reference) ---
        # We need to identify the "Balanced Speakers" (Pathological) first.
        # These are the speakers we must match in the Unbalanced Pathological set.
        
        pathological_entries = [e for e in entries if e["group"] == "pathological" and e["dtype"] == dtype]
        
        balanced_pathological_speakers = set()
        
        # 1. Determine which Pathological speakers qualify for Balanced
        if dtype == "utterances":
            for e in pathological_entries:
                if e["speaker"] in complete_sentence_speakers:
                     balanced_pathological_speakers.add(e["speaker"])
        else: # Words
            for e in pathological_entries:
                if e["is_a18"] and e["norm_text"] in valid_control_texts:
                    balanced_pathological_speakers.add(e["speaker"])

        # Now process both groups
        for group in groups:
            subset_entries = [e for e in entries if e["group"] == group and e["dtype"] == dtype]
            
            # --- 1. Balanced Partition ---
            balanced_set = []
            seen_bal = set()
            
            if dtype == "utterances":
                # Logic: S1 & S2 + Valid Text Check
                for e in subset_entries:
                    # Strict Speaker Check (Pathological must be complete, Control just needs to exist)
                    # Actually, for Balanced, we usually take *corresponding* data.
                    # For S1/S2, everyone who has it is included.
                    if e["speaker"] in complete_sentence_speakers: # or group==control? 
                        # Actually S1/S2 are fixed texts, so valid_control_texts check is implicit/pass
                        key = (e["speaker"], e["sentence_label"])
                        if key not in seen_bal:
                            balanced_set.append(e)
                            seen_bal.add(key)
            else: # Words
                # Logic: Only A18 Task + Valid Control Text
                for e in subset_entries:
                    if e["is_a18"]:
                        if e["norm_text"] in valid_control_texts:
                            key = e["wav_id"]
                            if key not in seen_bal:
                                balanced_set.append(e)
                                seen_bal.add(key)

            # --- 2. Unbalanced Partition ---
            # Logic: 
            #   Pathological: Must be in balanced_pathological_speakers
            #   Control: ALL controls (relaxed speaker constraint to ensure 'gil' appears)
            #   Both: Must be Valid Control Text
            
            unbalanced_set = []
            for e in subset_entries:
                # Speaker Constraint
                valid_speaker = True
                if group == "pathological":
                    if e["speaker"] not in balanced_pathological_speakers:
                        valid_speaker = False
                # else: Control -> All controls allowed
                
                if valid_speaker:
                    # Text Constraint (Strict)
                    if e["norm_text"] in valid_control_texts:
                        
                        # Score correction
                        final_score = e["score"]
                        if dtype == "word" and e["is_a18"] and group == "pathological":
                            final_score = scores_std.get(e["speaker"], "N/A")
                        
                        e_copy = e.copy()
                        e_copy["score"] = final_score
                        unbalanced_set.append(e_copy)

            # --- 3. All Partition ---
            # Logic: All Speakers + All Data (No Control Text Requirement)
            all_set = []
            for e in subset_entries:
                final_score = e["score"]
                if dtype == "word" and e["is_a18"] and group == "pathological":
                    final_score = scores_std.get(e["speaker"], "N/A")
                
                e_copy = e.copy()
                e_copy["score"] = final_score
                all_set.append(e_copy)

            # Write
            base_p = os.path.join(output_dir, "copas", group, dtype)
            
            n_bal = write_kaldi_subset(balanced_set, os.path.join(base_p, "balanced"))
            n_unbal = write_kaldi_subset(unbalanced_set, os.path.join(base_p, "unbalanced"))
            n_all = write_kaldi_subset(all_set, os.path.join(base_p, "all"))
            
            print(f"[{group.upper()} - {dtype.upper()}]")
            print(f"    Balanced:   {len(balanced_set)} utts ({n_bal} spks)")
            print(f"    Unbalanced: {len(unbalanced_set)} utts ({n_unbal} spks) [ValidCtrl]")
            print(f"    All:        {len(all_set)} utts ({n_all} spks) [No Control Req]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copas_root", default="/data/group1/z40484r/datasets/COPAS_1.0.1/Data/Data")
    parser.add_argument("--metadata_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/severity_data_2_test.csv")
    parser.add_argument("--age_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/age_test.csv")
    parser.add_argument("--gender_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/genders.csv")
    parser.add_argument("--output_dir", default="datasets")
    args = parser.parse_args()
    
    process_copas(args.copas_root, args.metadata_path, args.age_path, args.gender_path, args.output_dir)