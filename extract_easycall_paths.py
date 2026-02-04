import os
import argparse
import glob
import re
import string
from collections import defaultdict

# ==========================================
# 1. SCORES & CONFIG
# ==========================================
SPEAKER_SCORES = {
    "f01": "1", "f02": "3", "f03": "1", "f05": "1", "f06": "1",
    "f07": "1", "f08": "1", "f09": "1", "f10": "5", "f11": "2",
    "m01": "3", "m02": "1", "m03": "3", "m04": "1", "m05": "4",
    "m06": "4", "m07": "4", "m08": "3", "m09": "1", "m10": "3",
    "m11": "5", "m12": "1", "m13": "1", "m14": "5", "m15": "1",
    "m16": "3", "m17": "1", "m18": "1", "m19": "3", "m20": "1",
    "fc01": "N/A", "fc02": "N/A", "mc01": "N/A", "mc02": "N/A"
}

# Invert scores: 1->5, 2->4, 3->3, 4->2, 5->1
for spk in SPEAKER_SCORES:
    if SPEAKER_SCORES[spk] != "N/A":
        old_score = int(SPEAKER_SCORES[spk])
        new_score = 6 - old_score
        SPEAKER_SCORES[spk] = str(new_score)

OFFICIAL_TRANSCRIPTS = [
    "Aggiungi ai preferiti", "Aggiungi", "Apri rubrica", "Apri",
    "Attiva vivavoce", "Bue", "Cancella contatto", "Cancella tutto",
    "Cancella", "Cella", "Chiama emergenza", "Chiama ultimo numero",
    "Chiama", "Chiamata", "Chiudi applicazione", "Chiudi rubrica",
    "Chiudi", "Cinque", "Deseleziona", "Disattiva vivavoce", "Due",
    "Fai una telefonata", "Indietro", "Mali", "Muovi", "Muto",
    "Nave", "No", "Nove", "Nuovo contatto", "Otto", "Preferiti",
    "Quattro", "Raggiungi", "Richiama", "Rimuovi", "Rubrica",
    "Sali", "Salva", "Scendi", "Scesi", "Scorri verso il basso",
    "Scorri verso l’alto", "Sei", "Seleziona", "Sette", "Sezione",
    "Si", "Sopra", "Sotto", "Stop", "Tastiera", "Termina chiamata",
    "Terminare", "Top", "Tra", "Tre", "Uno", "Vai alla pagina principale",
    "Vai alla tastiera", "Vai nei preferiti", "Vai nel registro chiamate",
    "Vai nella rubrica", "Vivavoce", "Zelo", "Zero"
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):
    """
    Standard normalization for lookup keys: lowercase, no spaces, no punctuation.
    """
    t = text.lower()
    t = t.replace("’", "").replace("'", "")
    t = t.replace(" ", "").replace("_", "")
    return t.strip()

def clean_filename_stem(stem):
    """
    Removes repetition suffixes like '_1', '_2' or just '1', '2' from the end of the filename.
    Example: 'Apri_rubrica_1' -> 'Apri_rubrica'
    """
    # Regex to remove trailing underscore + digits (e.g. _1, _02)
    s = re.sub(r'_\d+$', '', stem)
    # Also handle trailing digits without underscore if necessary (careful not to kill '5' if word is 5)
    # But EasyCall repetition usually follows underscore.
    return s

def build_lookup_table(official_list):
    table = {}
    for phrase in official_list:
        key = normalize_text(phrase)
        table[key] = phrase
    return table

def get_gender(speaker_id):
    if speaker_id.lower().startswith("f"):
        return "f"
    return "m"

# ==========================================
# 3. SCANNING
# ==========================================

def scan_easycall(dataset_root):
    lookup_table = build_lookup_table(OFFICIAL_TRANSCRIPTS)
    all_entries = []
    
    print(f"Scanning directory: {dataset_root}...")
    all_wavs = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    
    for wav_path in all_wavs:
        filename = os.path.basename(wav_path)
        
        # Expecting format: m05_01_apri.wav
        if filename.count("_") < 2: continue

        try:
            parts = filename.split("_", 2)
            speaker_id = parts[0].lower()
            session_id = parts[1]
            raw_text_part = parts[2]
            
            is_control = "c" in speaker_id
            
            # Filter Check
            if (speaker_id not in SPEAKER_SCORES) and (not is_control): 
                continue

            # --- CLEANING LOGIC ---
            raw_text_no_ext = os.path.splitext(raw_text_part)[0]
            
            # 1. Strip repetition markers (Apri_rubrica_1 -> Apri_rubrica)
            cleaned_text = clean_filename_stem(raw_text_no_ext)
            
            # 2. Normalize for lookup (Apri_rubrica -> aprirubrica)
            file_key = normalize_text(cleaned_text)
            
            # 3. Lookup
            official_transcript = lookup_table.get(file_key)
            if not official_transcript: 
                # Fallback: try looking up without the cleaning (edge cases)
                file_key_raw = normalize_text(raw_text_no_ext)
                official_transcript = lookup_table.get(file_key_raw)
                
            if not official_transcript: continue

            # Determine Group
            if is_control:
                group = "control"
            else:
                group = "pathological"
            
            # Determine Type (Word vs Utterance)
            if " " in official_transcript:
                dtype = "utterances"
            else:
                dtype = "word"

            utt_id = f"{speaker_id}_{session_id}_{file_key}" # Keep unique ID robust
            # Add suffix back to ID if it was stripped, to ensure uniqueness of repeated files?
            # Actually, we rely on the filename being unique on disk. 
            # If we map 'Apri_rubrica' and 'Apri_rubrica_1' to the same utt_id, Kaldi will complain.
            # Let's make utt_id unique by including the RAW filename stem part.
            unique_suffix = normalize_text(raw_text_no_ext)
            utt_id = f"{speaker_id}_{session_id}_{unique_suffix}"
            
            score = SPEAKER_SCORES.get(speaker_id, "N/A")
            
            all_entries.append({
                "utt_id": utt_id,
                "wav_path": os.path.abspath(wav_path),
                "transcript": official_transcript,
                "norm_text": file_key, # Use the CLEANED key for intersection logic
                "speaker": speaker_id,
                "gender": get_gender(speaker_id),
                "group": group,
                "dtype": dtype,
                "score": score
            })

        except Exception: pass
            
    return all_entries

# ==========================================
# 4. LOGIC: INTERSECTION & VALIDATION
# ==========================================

def get_intersection_texts(entries, group_filter="pathological", dtype_filter="word"):
    filtered = [e for e in entries if e["group"] == group_filter and e["dtype"] == dtype_filter]
    
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

def get_valid_texts_by_gender_count(entries, dtype_filter, min_per_gender=2):
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
    
    entries = sorted(entries, key=lambda x: x['utt_id'])
    
    spk2utt_map = defaultdict(list)
    spk_map = {} 

    with open(os.path.join(out_dir, "wav.scp"), "w") as f_wav, \
         open(os.path.join(out_dir, "text"), "w") as f_text, \
         open(os.path.join(out_dir, "utt2spk"), "w") as f_u2s, \
         open(os.path.join(out_dir, "language"), "w") as f_lang:
         
        f_lang.write("it\n")
         
        for e in entries:
            utt_id = e['utt_id']
            spk_id = e['speaker']
            
            f_wav.write(f"{utt_id} {e['wav_path']}\n")
            f_text.write(f"{utt_id} {e['transcript']}\n")
            f_u2s.write(f"{utt_id} {spk_id}\n")
            
            spk2utt_map[spk_id].append(utt_id)
            spk_map[spk_id] = {'score': e['score'], 'gender': e['gender']}

    with open(os.path.join(out_dir, "spk2utt"), "w") as f_s2u, \
         open(os.path.join(out_dir, "spk2uttnum"), "w") as f_s2un:
         
        for spk in sorted(spk2utt_map.keys()):
            utts_str = " ".join(spk2utt_map[spk])
            count = len(spk2utt_map[spk])
            
            f_s2u.write(f"{spk} {utts_str}\n")
            f_s2un.write(f"{spk} {count}\n")

    with open(os.path.join(out_dir, "spk2score"), "w") as f_s2sc, \
         open(os.path.join(out_dir, "spk2gender"), "w") as f_s2gen:
         
        for spk in sorted(spk_map.keys()):
            f_s2sc.write(f"{spk} {spk_map[spk]['score']}\n")
            f_s2gen.write(f"{spk} {spk_map[spk]['gender']}\n")

    return len(spk_map)

def process_easycall(dataset_root, output_dir):
    entries = scan_easycall(dataset_root)
    print(f"Total utterances found: {len(entries)}")
    
    groups = ["pathological", "control"]
    dtypes = ["word", "utterances"]
    
    for dtype in dtypes:
        print(f"\n=== Processing {dtype.upper()} ===")
        
        valid_control_texts = get_valid_texts_by_gender_count(entries, dtype, 2)
        print(f"  Valid Control Texts (>=2M & >=2F): {len(valid_control_texts)}")

        raw_bal_texts, pd_spks = get_intersection_texts(entries, "pathological", dtype)
        
        final_balanced_texts = raw_bal_texts.intersection(valid_control_texts)
        print(f"  Final Balanced Texts (Intersection + Control Valid): {len(final_balanced_texts)}")
        
        for group in groups:
            subset_entries = [e for e in entries if e["group"] == group and e["dtype"] == dtype]
            
            balanced_set = []
            seen_bal = set()
            for e in subset_entries:
                if e["norm_text"] in final_balanced_texts:
                    key = (e["speaker"], e["norm_text"])
                    if key not in seen_bal:
                        balanced_set.append(e)
                        seen_bal.add(key)
            
            balanced_speakers = set(e["speaker"] for e in balanced_set)

            unbalanced_set = []
            for e in subset_entries:
                if e["speaker"] in balanced_speakers:
                    if e["norm_text"] in valid_control_texts:
                        unbalanced_set.append(e)

            base_p = os.path.join(output_dir, "easycall", group, dtype)
            
            n_bal = write_kaldi_subset(balanced_set, os.path.join(base_p, "balanced"))
            n_unbal = write_kaldi_subset(unbalanced_set, os.path.join(base_p, "unbalanced"))
            
            print(f"  [{group.upper()}]")
            print(f"    Balanced:   {len(balanced_set)} utts ({n_bal} spks)")
            print(f"    Unbalanced: {len(unbalanced_set)} utts ({n_unbal} spks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--easycall_root", default="/data/group1/z40484r/datasets/EasyCall", 
                        help="Path to EasyCall root directory")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_easycall(args.easycall_root, args.output_dir)