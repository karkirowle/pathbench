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
# 1. HELPERS: PARSING & NORMALIZATION
# ==========================================

def get_speaker_from_filename(filename):
    """Extracts speaker ID (first part of filename before underscore)."""
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0].upper()
    return None

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

def _clean(line):
    return line.strip().replace('"', '')

def parse_textgrid_intervals(textgrid_path):
    """
    Robust parser for Praat TextGrids (Long and Short formats).
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
            # Short format parser
            iterator = iter(lines)
            try:
                while True:
                    if "<exists>" in next(iterator): break
                num_tiers = int(_clean(next(iterator)))
                for _ in range(num_tiers):
                    tier_type = _clean(next(iterator))
                    tier_name = _clean(next(iterator))
                    next(iterator); next(iterator) # Skip times
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
    scores = {}
    if not metadata_path or not os.path.exists(metadata_path): return scores
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 5:
                    scores[parts[0].strip().upper()] = parts[4].strip()
    except Exception: pass
    return scores

def load_metadata_age(age_path):
    ages = {}
    if not age_path or not os.path.exists(age_path): return ages
    try:
        with open(age_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    ages[parts[0].strip().upper()] = parts[1].strip()
    except Exception: pass
    return ages

# ==========================================
# 3. SCANNING & BALANCING LOGIC
# ==========================================

def scan_copas(copas_root, metadata_scores, metadata_ages):
    dir_s1 = os.path.join(copas_root, "S1")
    dir_s2 = os.path.join(copas_root, "S2")
    dir_dia = os.path.join(copas_root, "DIA")
    
    all_entries = []

    # --- Process Sentences (S1 & S2) ---
    print("Scanning Sentences (S1/S2)...")
    for folder_path, folder_name in [(dir_s1, "S1"), (dir_s2, "S2")]:
        if not os.path.isdir(folder_path): continue
        for wav_path in glob.glob(os.path.join(folder_path, "*.wav")):
            filename = os.path.basename(wav_path)
            speaker_id = get_speaker_from_filename(filename)
            if not speaker_id: continue
            
            group = "control" if speaker_id.startswith("N") else "pathological"
            transcription = SENTENCE_TEXTS.get(folder_name, "")
            
            all_entries.append({
                "utt_id": f"{speaker_id}_{folder_name}",
                "wav_id": f"{speaker_id}_{folder_name}",
                "wav_path": os.path.abspath(wav_path),
                "transcript": transcription,
                "norm_text": normalize_text(transcription),
                "speaker": speaker_id,
                "group": group,
                "dtype": "utterances",
                "sentence_label": folder_name, # Track S1 vs S2 explicitly
                "score": "Control" if group == "control" else metadata_scores.get(speaker_id, "N/A"),
                "age": metadata_ages.get(speaker_id, "N/A"),
                "has_segments": False
            })

    # --- Process Words (DIA) ---
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
                    "group": group,
                    "dtype": "word",
                    "score": "Control" if group == "control" else metadata_scores.get(speaker_id, "N/A"),
                    "age": metadata_ages.get(speaker_id, "N/A"),
                    "has_segments": True
                })

    return all_entries

def get_complete_sentence_speakers(entries):
    """
    Returns a set of speakers who have BOTH 'S1' and 'S2' recordings.
    """
    spk_sents = defaultdict(set)
    for e in entries:
        if e["dtype"] == "utterances":
            spk_sents[e["speaker"]].add(e["sentence_label"])
            
    # Filter speakers who have exactly {"S1", "S2"}
    complete_speakers = set()
    for spk, labels in spk_sents.items():
        if "S1" in labels and "S2" in labels:
            complete_speakers.add(spk)
            
    return complete_speakers

def get_intersection_words(entries, group_filter="pathological"):
    """
    Returns the set of NORMALIZED WORD TEXTS spoken by ALL speakers in the target group.
    """
    filtered = [e for e in entries if e["group"] == group_filter and e["dtype"] == "word"]
    
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

# ==========================================
# 4. WRITING OUTPUTS
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
        "age": open(os.path.join(out_dir, "spk2age"), "w")
    }
    
    files["lang"].write("nl\n")
    
    written_wavs = set()
    spk_data = {}

    count_segments = 0

    for e in entries:
        # wav.scp
        if e["wav_id"] not in written_wavs:
            files["wav"].write(f"{e['wav_id']} {e['wav_path']}\n")
            written_wavs.add(e["wav_id"])
        
        # segments
        if e["has_segments"]:
            files["segments"].write(f"{e['utt_id']} {e['wav_id']} {e['start']:.3f} {e['end']:.3f}\n")
            count_segments += 1
        
        # text & utt2spk
        files["text"].write(f"{e['utt_id']} {e['transcript']}\n")
        files["utt2spk"].write(f"{e['utt_id']} {e['speaker']}\n")
        
        spk_data[e['speaker']] = {"score": e['score'], "age": e['age']}

    # Speaker Files
    for spk in sorted(spk_data.keys()):
        files["score"].write(f"{spk} {spk_data[spk]['score']}\n")
        files["age"].write(f"{spk} {spk_data[spk]['age']}\n")

    files["segments"].close()
    if count_segments == 0:
        os.remove(os.path.join(out_dir, "segments"))
        
    for f in files.values():
        if not f.closed: f.close()

def process_copas(copas_root, metadata_path, age_path, output_dir):
    # 1. Load Meta & Scan
    scores = load_metadata_scores(metadata_path)
    ages = load_metadata_age(age_path)
    entries = scan_copas(copas_root, scores, ages)
    print(f"Total entries found: {len(entries)}")
    
    # 2. Get Balanced Criteria
    # A. Sentences: Find speakers who have BOTH S1 and S2
    complete_sentence_speakers = get_complete_sentence_speakers(entries)
    print(f"Speakers with complete sentence set (S1+S2): {len(complete_sentence_speakers)}")

    # B. Words: Find intersection texts
    bal_word_texts, _ = get_intersection_words(entries, "pathological")

    # 3. Partition & Write
    groups = ["pathological", "control"]
    dtypes = ["word", "utterances"]

    for group in groups:
        for dtype in dtypes:
            subset_entries = [e for e in entries if e["group"] == group and e["dtype"] == dtype]
            balanced_subset = []
            
            # --- Balanced Partition Logic ---
            if dtype == "utterances":
                # STRICT RULE: Only include speakers who have BOTH S1 and S2
                # Deduplication logic usually not needed for S1/S2 as duplicates are rare, 
                # but if they exist, take the first one.
                seen_sent = set()
                for e in subset_entries:
                    if e["speaker"] in complete_sentence_speakers:
                        key = (e["speaker"], e["sentence_label"])
                        if key not in seen_sent:
                            balanced_subset.append(e)
                            seen_sent.add(key)
                            
            else: # Words
                # INTERSECTION RULE: Text must be in intersection set + Deduplication
                seen_word = set()
                for e in subset_entries:
                    if e["norm_text"] in bal_word_texts:
                        key = (e["speaker"], e["norm_text"])
                        if key not in seen_word:
                            balanced_subset.append(e)
                            seen_word.add(key)

            # --- All (Unbalanced) Logic ---
            unbalanced_subset = subset_entries

            # Write
            write_kaldi_subset(balanced_subset, os.path.join(output_dir, "copas", group, dtype, "balanced"))
            write_kaldi_subset(unbalanced_subset, os.path.join(output_dir, "copas", group, dtype, "unbalanced"))
            
            print(f"[{group.upper()} - {dtype.upper()}] Balanced: {len(balanced_subset)}, All: {len(unbalanced_subset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copas_root", default="/data/group1/z40484r/datasets/COPAS_1.0.1/Data/Data")
    parser.add_argument("--metadata_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/severity_data_2_test.csv")
    parser.add_argument("--age_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/age_test.csv")
    parser.add_argument("--output_dir", default="datasets")
    args = parser.parse_args()
    
    process_copas(args.copas_root, args.metadata_path, args.age_path, args.output_dir)