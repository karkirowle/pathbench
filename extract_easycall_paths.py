import os
import argparse
import glob
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
    # Controls
    "fc01": "N/A", "fc02": "N/A", "mc01": "N/A", "mc02": "N/A"
}

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
    t = text.lower()
    t = t.replace("’", "").replace("'", "")
    t = t.replace(" ", "").replace("_", "")
    return t.strip()

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
        
        # Expecting: m05_01_apri.wav
        if filename.count("_") < 2: continue

        try:
            parts = filename.split("_", 2)
            speaker_id = parts[0].lower()
            session_id = parts[1]
            raw_text_part = parts[2]
            
            # Filter Check
            if speaker_id not in SPEAKER_SCORES: continue

            raw_text_no_ext = os.path.splitext(raw_text_part)[0]
            
            # Lookup Transcript
            file_key = normalize_text(raw_text_no_ext)
            official_transcript = lookup_table.get(file_key)
            if not official_transcript: continue

            # Determine Group
            if "c" in speaker_id:
                group = "control"
            else:
                group = "pathological"
            
            # Determine Type (Word vs Utterance)
            if " " in official_transcript:
                dtype = "utterances"
            else:
                dtype = "word"

            utt_id = f"{speaker_id}_{session_id}_{file_key}"
            
            all_entries.append({
                "utt_id": utt_id,
                "wav_path": os.path.abspath(wav_path),
                "transcript": official_transcript,
                "norm_text": file_key, 
                "speaker": speaker_id,
                "gender": get_gender(speaker_id),
                "group": group,
                "dtype": dtype,
                "score": SPEAKER_SCORES[speaker_id]
            })

        except Exception: pass
            
    return all_entries

# ==========================================
# 4. LOGIC: INTERSECTION & VALIDATION
# ==========================================

def get_intersection_texts(entries, group_filter="pathological", dtype_filter="word"):
    """
    Returns normalized texts spoken by ALL speakers in the target group.
    """
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
    
    with open(os.path.join(out_dir, "wav.scp"), "w") as f_wav, \
         open(os.path.join(out_dir, "text"), "w") as f_text, \
         open(os.path.join(out_dir, "utt2spk"), "w") as f_u2s, \
         open(os.path.join(out_dir, "language"), "w") as f_lang:
         
        f_lang.write("it\n")
         
        for e in entries:
            f_wav.write(f"{e['utt_id']} {e['wav_path']}\n")
            f_text.write(f"{e['utt_id']} {e['transcript']}\n")
            f_u2s.write(f"{e['utt_id']} {e['speaker']}\n")

    spk_map = {e['speaker']: e['score'] for e in entries}
    
    with open(os.path.join(out_dir, "spk2score"), "w") as f_s2sc, \
         open(os.path.join(out_dir, "spk2gender"), "w") as f_s2gen:
         
        for spk in sorted(spk_map.keys()):
            f_s2sc.write(f"{spk} {spk_map[spk]}\n")
            f_s2gen.write(f"{spk} {get_gender(spk)}\n")

    return len(spk_map) # Return number of unique speakers

def process_easycall(dataset_root, output_dir):
    # 1. Scan
    entries = scan_easycall(dataset_root)
    print(f"Total utterances found: {len(entries)}")
    
    groups = ["pathological", "control"]
    dtypes = ["word", "utterances"]
    
    for dtype in dtypes:
        print(f"\n=== Processing {dtype.upper()} ===")
        
        # A. Strict Control Validation (Used for Balanced/Unbalanced only)
        valid_control_texts = get_valid_texts_by_gender_count(entries, dtype, 2)
        print(f"  Valid Control Texts (>=2M & >=2F): {len(valid_control_texts)}")

        # B. Pathological Intersection
        raw_bal_texts, pd_spks = get_intersection_texts(entries, "pathological", dtype)
        
        # C. Target Texts for Balanced/Unbalanced (Intersection + Valid Controls)
        final_balanced_texts = raw_bal_texts.intersection(valid_control_texts)
        print(f"  Final Balanced Texts (Intersection + Control Valid): {len(final_balanced_texts)}")
        
        for group in groups:
            subset_entries = [e for e in entries if e["group"] == group and e["dtype"] == dtype]
            
            # --- 1. Balanced (Intersection + Valid Control + Deduplication) ---
            balanced_set = []
            seen_bal = set()
            for e in subset_entries:
                if e["norm_text"] in final_balanced_texts:
                    key = (e["speaker"], e["norm_text"])
                    if key not in seen_bal:
                        balanced_set.append(e)
                        seen_bal.add(key)
            
            # --- 2. Unbalanced (Same texts as Balanced + All Recordings) ---
            unbalanced_set = []
            for e in subset_entries:
                if e["norm_text"] in final_balanced_texts:
                    unbalanced_set.append(e)

            # --- 3. All (No Text Filtering) ---
            all_set = subset_entries

            # Write Outputs
            base_p = os.path.join(output_dir, "easycall", group, dtype)
            
            n_bal = write_kaldi_subset(balanced_set, os.path.join(base_p, "balanced"))
            n_unbal = write_kaldi_subset(unbalanced_set, os.path.join(base_p, "unbalanced"))
            n_all = write_kaldi_subset(all_set, os.path.join(base_p, "all"))
            
            print(f"  [{group.upper()}]")
            print(f"    Balanced:   {len(balanced_set)} utts ({n_bal} spks)")
            print(f"    Unbalanced: {len(unbalanced_set)} utts ({n_unbal} spks)")
            print(f"    All:        {len(all_set)} utts ({n_all} spks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--easycall_root", default="/data/group1/z40484r/datasets/EasyCall", 
                        help="Path to EasyCall root directory")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_easycall(args.easycall_root, args.output_dir)