import os
import argparse
import csv
import glob
import string
from collections import defaultdict

# ==========================================
# 1. SCORING, TEXT & GENDER HELPERS
# ==========================================

def map_letter_to_score(letter):
    """
    Maps Frenchay letter grades to numerical values.
    """
    key = letter.lower().replace(" ", "").strip()
    mapping = {
        "a": "5", "a/b": "4.5",
        "b": "4", "b/c": "3.5",
        "c": "3", "c/d": "2.5",
        "d": "2", "d/e": "1.5",
        "e": "1"
    }
    return mapping.get(key, letter)

def normalize_text(text):
    """
    Normalizes text for comparison:
    - Lowercase
    - Strip punctuation
    - Strip whitespace
    """
    if not text: return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()

def get_gender(speaker_id):
    """
    Infers gender from TORGO speaker ID.
    F* / FC* -> female ('f')
    M* / MC* -> male ('m')
    """
    if speaker_id.upper().startswith("F"):
        return "f"
    elif speaker_id.upper().startswith("M"):
        return "m"
    return "m" # Fallback

def get_speaker_scores(speaker_dir):
    """
    Parses the .csv file in the speaker's 'Notes' directory.
    """
    notes_dir = os.path.join(speaker_dir, "Notes")
    if not os.path.isdir(notes_dir): return None

    csv_files = glob.glob(os.path.join(notes_dir, "*.csv"))
    if not csv_files: return None
    
    csv_path = csv_files[0]
    scores = {"word": "N/A", "utterances": "N/A"}
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                cleaned_row = [x.strip() for x in row]
                
                if "Words" in cleaned_row:
                    try:
                        idx = cleaned_row.index("Words")
                        if idx + 1 < len(cleaned_row) and cleaned_row[idx+1]:
                            scores["word"] = map_letter_to_score(cleaned_row[idx + 1])
                    except ValueError: pass

                if "Sentences" in cleaned_row:
                    try:
                        idx = cleaned_row.index("Sentences")
                        if idx + 1 < len(cleaned_row) and cleaned_row[idx+1]:
                            scores["utterances"] = map_letter_to_score(cleaned_row[idx + 1])
                    except ValueError: pass
        return scores
    except Exception as e:
        print(f"Warning: Could not parse CSV {csv_path}: {e}")
        return None

# ==========================================
# 2. SCANNING & BALANCING
# ==========================================

def scan_torgo(torgo_root):
    pathological_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    control_speakers = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]
    
    # Pre-fetch scores
    spk_scores = {}
    for spk in pathological_speakers:
        s_dir = os.path.join(torgo_root, spk)
        if os.path.isdir(s_dir):
            spk_scores[spk] = get_speaker_scores(s_dir) or {"word": "N/A", "utterances": "N/A"}
    
    all_entries = []
    full_list = pathological_speakers + control_speakers
    
    print("Scanning speakers...")
    for speaker in full_list:
        speaker_dir = os.path.join(torgo_root, speaker)
        if not os.path.isdir(speaker_dir): continue

        if speaker in pathological_speakers:
            group = "pathological"
            scores = spk_scores.get(speaker, {"word": "N/A", "utterances": "N/A"})
        else:
            group = "control"
            scores = {"word": "N/A", "utterances": "N/A"}

        for session in os.listdir(speaker_dir):
            if not session.startswith("Session"): continue
            
            session_dir = os.path.join(speaker_dir, session)
            prompts_dir = os.path.join(session_dir, "prompts")
            wav_dir = os.path.join(session_dir, "wav_arrayMic")

            if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir): continue

            for prompt_file in os.listdir(prompts_dir):
                if not prompt_file.endswith(".txt"): continue
                
                # 1. Read Transcript
                prompt_path = os.path.join(prompts_dir, prompt_file)
                try:
                    with open(prompt_path, "r", encoding="utf-8", errors='ignore') as f:
                        transcription = f.read().strip()
                except Exception: continue

                if not transcription: continue
                if "[" in transcription or "/" in transcription: continue 

                # 2. Determine Type (Word vs Utterance)
                if " " in transcription:
                    dtype = "utterances"
                else:
                    dtype = "word"

                # 3. Check Audio Existence
                prompt_basename = os.path.splitext(prompt_file)[0]
                wav_name = f"{prompt_basename}.wav"
                wav_path = os.path.join(wav_dir, wav_name)
                
                if not os.path.exists(wav_path):
                    wav_name_alt = f"{speaker}_{session}_{prompt_basename}.wav"
                    wav_path = os.path.join(wav_dir, wav_name_alt)
                    if not os.path.exists(wav_path): continue

                # 4. Create Entry
                all_entries.append({
                    "utt_id": f"{speaker}_{session}_{prompt_basename}",
                    "wav_path": os.path.abspath(wav_path),
                    "transcript": transcription,
                    "norm_text": normalize_text(transcription),
                    "speaker": speaker,
                    "gender": get_gender(speaker),
                    "group": group,
                    "dtype": dtype,
                    "score": scores[dtype]
                })

    return all_entries

def get_intersection_texts(entries, group_filter="pathological", type_filter="word"):
    """
    Returns the set of NORMALIZED TEXTS spoken by ALL speakers in the target group.
    """
    filtered = [e for e in entries if e["group"] == group_filter and e["dtype"] == type_filter]
    
    text_to_spk = defaultdict(set)
    all_spk = set()
    
    for e in filtered:
        text_to_spk[e["norm_text"]].add(e["speaker"])
        all_spk.add(e["speaker"])
        
    intersection = set()
    for text, speakers in text_to_spk.items():
        if speakers == all_spk:
            intersection.add(text)
            
    return intersection, all_spk

def get_valid_texts_by_gender_count(entries, dtype_filter="word", min_per_gender=2):
    """
    Returns a set of normalized texts that have been spoken by:
    - At least 'min_per_gender' MALE Control speakers
    - AND
    - At least 'min_per_gender' FEMALE Control speakers
    """
    # Filter for Control entries of the specific type
    control_entries = [e for e in entries if e["group"] == "control" and e["dtype"] == dtype_filter]
    
    # Map text -> {'m': set(speakers), 'f': set(speakers)}
    text_stats = defaultdict(lambda: {'m': set(), 'f': set()})
    
    for e in control_entries:
        g = e['gender']
        text_stats[e["norm_text"]][g].add(e["speaker"])
    
    valid_texts = set()
    for text, stats in text_stats.items():
        count_m = len(stats['m'])
        count_f = len(stats['f'])
        
        # STRICT CONDITION: Must have enough of BOTH genders
        if count_m >= min_per_gender and count_f >= min_per_gender:
            valid_texts.add(text)
            
    return valid_texts

# ==========================================
# 3. WRITING OUTPUTS
# ==========================================

def write_kaldi_subset(entries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Write standard Kaldi files
    with open(os.path.join(out_dir, "wav.scp"), "w") as f_wav, \
         open(os.path.join(out_dir, "text"), "w") as f_text, \
         open(os.path.join(out_dir, "utt2spk"), "w") as f_u2s, \
         open(os.path.join(out_dir, "language"), "w") as f_lang: 
         
        # Write language file content
        f_lang.write("en\n")
         
        for e in entries:
            f_wav.write(f"{e['utt_id']} {e['wav_path']}\n")
            f_text.write(f"{e['utt_id']} {e['transcript']}\n")
            f_u2s.write(f"{e['utt_id']} {e['speaker']}\n")

    # Speaker Level Files
    spk_map = {e['speaker']: e['score'] for e in entries}
    
    with open(os.path.join(out_dir, "spk2score"), "w") as f_s2sc, \
         open(os.path.join(out_dir, "spk2gender"), "w") as f_s2gen:
         
        for spk in sorted(spk_map.keys()):
            # Write Score
            f_s2sc.write(f"{spk} {spk_map[spk]}\n")
            
            # Write Gender
            gender = get_gender(spk)
            f_s2gen.write(f"{spk} {gender}\n")

def process_torgo(torgo_root, output_dir):
    # 1. Scan everything
    entries = scan_torgo(torgo_root)
    print(f"Total utterances found: {len(entries)}")
    
    # --- NEW: Control Speaker Statistics ---
    control_speakers = set(e['speaker'] for e in entries if e['group'] == 'control')
    n_controls = len(control_speakers)
    n_male_ctrl = len([s for s in control_speakers if get_gender(s) == 'm'])
    n_female_ctrl = len([s for s in control_speakers if get_gender(s) == 'f'])
    
    print("-" * 40)
    print(f"CONTROL SPEAKER STATISTICS")
    print(f"Total Controls: {n_controls}")
    print(f"  Male:   {n_male_ctrl}")
    print(f"  Female: {n_female_ctrl}")
    
    # 2. Determine "Valid Texts" (Strict Control Coverage)
    valid_word_texts = get_valid_texts_by_gender_count(entries, "word", 2)
    valid_utt_texts = get_valid_texts_by_gender_count(entries, "utterances", 2)
    
    print("-" * 40)
    print(f"STRICT CONTROL VALIDATION (>=2 Male AND >=2 Female Controls)")
    print(f"Valid 'All' Texts - Words: {len(valid_word_texts)}")
    print(f"Valid 'All' Texts - Utterances: {len(valid_utt_texts)}")
    print("-" * 40)
    
    # 3. Determine Balanced Texts (Intersection of PD speakers)
    raw_bal_word_texts, pd_word_spks = get_intersection_texts(entries, "pathological", "word")
    raw_bal_utt_texts, pd_utt_spks = get_intersection_texts(entries, "pathological", "utterances")
    
    # ENFORCE VALIDATION ON BALANCED
    final_bal_word_texts = raw_bal_word_texts.intersection(valid_word_texts)
    final_bal_utt_texts = raw_bal_utt_texts.intersection(valid_utt_texts)

    print(f"Balanced Words (Intersection of PDs AND Valid Controls): {len(final_bal_word_texts)} (dropped {len(raw_bal_word_texts) - len(final_bal_word_texts)})")
    print(f"Balanced Utterances (Intersection of PDs AND Valid Controls): {len(final_bal_utt_texts)} (dropped {len(raw_bal_utt_texts) - len(final_bal_utt_texts)})")

    # 4. Partition and Write
    groups = ["pathological", "control"]
    dtypes = ["word", "utterances"]
    
    for group in groups:
        for dtype in dtypes:
            # Select relevant Texts sets
            target_balanced = final_bal_word_texts if dtype == "word" else final_bal_utt_texts
            target_all_valid = valid_word_texts if dtype == "word" else valid_utt_texts
            
            # Filter entries for this group/dtype
            subset_entries = [e for e in entries if e["group"] == group and e["dtype"] == dtype]
            
            # --- 1. Balanced Partition (Strict Intersection + Deduplication + Valid Control Coverage) ---
            balanced_subset = []
            seen_balanced_items = set() # Key: (speaker_id, normalized_text)

            for e in subset_entries:
                # Must be in the intersection set (which is already filtered for control validity)
                if e["norm_text"] in target_balanced:
                    key = (e["speaker"], e["norm_text"])
                    if key not in seen_balanced_items:
                        balanced_subset.append(e)
                        seen_balanced_items.add(key)
            
            # --- 2. Unbalanced Partition (All Valid Data) ---
            # Must satisfy the strict control coverage requirement
            unbalanced_subset = [e for e in subset_entries if e["norm_text"] in target_all_valid]
            
            # Write Balanced
            out_path_bal = os.path.join(output_dir, "torgo", group, dtype, "balanced")
            write_kaldi_subset(balanced_subset, out_path_bal)
            
            # Write Unbalanced (All)
            out_path_unbal = os.path.join(output_dir, "torgo", group, dtype, "unbalanced")
            write_kaldi_subset(unbalanced_subset, out_path_unbal)
            
            print(f"[{group.upper()} - {dtype.upper()}] Balanced: {len(balanced_subset)}, Unbalanced (All Valid): {len(unbalanced_subset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torgo_root", default="/data/group1/z40484r/datasets/TORGO", 
                        help="Path to TORGO root directory")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_torgo(args.torgo_root, args.output_dir)