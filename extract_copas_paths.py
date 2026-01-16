import os
import argparse
import glob
import re

# --- CONFIGURATION: S1 and S2 Texts ---
SENTENCE_TEXTS = {
    "S1": "Wil je liever de thee of de borrel?", 
    "S2": "Na nieuwjaar was hij weeral hier." 
}

# Add any labels you want to ignore here (e.g. silence, breath, noise)
IGNORE_LABELS = ["", "sil", "<sil>", "sp", "#", "nsp"]

def get_speaker_from_filename(filename):
    """Extracts speaker ID (first part of filename before underscore)."""
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0].upper()
    return None

def _clean(line):
    """Helper to remove quotes and whitespace."""
    return line.strip().replace('"', '')

def parse_textgrid_intervals(textgrid_path):
    """
    Robust parser for Praat TextGrids.
    Handles 'Long' (verbose key=value) and 'Short' (positional) formats.
    Prioritizes the 'target' tier if multiple tiers exist.
    """
    intervals = []
    
    try:
        with open(textgrid_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines()]

        if not lines:
            return []

        # --- 1. DETECT FORMAT ---
        is_long_format = any("xmin =" in l for l in lines[:20])

        target_intervals = []
        fallback_intervals = []
        found_target = False

        if is_long_format:
            # === LONG FORMAT PARSER ===
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
                    if line.startswith("xmin ="):
                        xmin = float(line.split("=")[1].strip())
                    elif line.startswith("xmax ="):
                        xmax = float(line.split("=")[1].strip())
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
            # === SHORT / POSITIONAL PARSER ===
            iterator = iter(lines)
            try:
                while True:
                    line = next(iterator)
                    if "<exists>" in line: break
                
                num_tiers = int(_clean(next(iterator)))

                for _ in range(num_tiers):
                    tier_type = _clean(next(iterator))
                    tier_name = _clean(next(iterator))
                    
                    next(iterator); next(iterator) # Skip tier times
                    
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
                        for _ in range(num_items):
                            next(iterator); next(iterator)

            except StopIteration:
                pass
            except ValueError:
                print(f"Warning: parsing structure failed for {textgrid_path}")

        if found_target: return target_intervals
        return fallback_intervals

    except Exception as e:
        print(f"Warning: Error parsing {textgrid_path}: {e}")
        return []

def load_metadata_scores(metadata_path):
    """Loads scores from CSV. Assumes Speaker ID is Col 0, Score is Col 4."""
    scores = {}
    if not metadata_path or not os.path.exists(metadata_path):
        return scores
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 5: # Ensure enough columns exist
                    scores[parts[0].strip().upper()] = parts[4].strip()
    except Exception as e:
        print(f"Warning reading scores: {e}")
    return scores

def load_metadata_age(age_path):
    """Loads age from CSV. Assumes Speaker ID is Col 0, Age is Col 1."""
    ages = {}
    if not age_path or not os.path.exists(age_path):
        print(f"Warning: Age file not found at {age_path}")
        return ages
    try:
        with open(age_path, 'r', encoding='utf-8') as f:
            f.readline() # Skip header if present
            for line in f:
                # Handle comma or space separation
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    spk_id = parts[0].strip().upper()
                    age_val = parts[1].strip()
                    ages[spk_id] = age_val
    except Exception as e:
        print(f"Warning reading ages: {e}")
    return ages

def extract_copas_paths(copas_root, metadata_path, age_path, base_output_dir):
    
    # --- Setup Directories ---
    dir_s1 = os.path.join(copas_root, "S1")
    dir_s2 = os.path.join(copas_root, "S2")
    dir_dia = os.path.join(copas_root, "DIA")

    speaker_groups = ["pathological", "control"]
    data_types = ["word", "utterances"]
    
    # Load Metadata
    metadata_scores = load_metadata_scores(metadata_path)
    metadata_ages = load_metadata_age(age_path)
    
    handles = {}
    speakers_found = {g: {d: set() for d in data_types} for g in speaker_groups}

    try:
        # --- 1. Open Output Files ---
        for group in speaker_groups:
            handles[group] = {}
            for dtype in data_types:
                out_path = os.path.join(base_output_dir, "copas", group, dtype)
                os.makedirs(out_path, exist_ok=True)
                
                files = {
                    "wav_scp": open(os.path.join(out_path, "wav.scp"), "w"),
                    "text": open(os.path.join(out_path, "text"), "w"),
                    "utt2spk": open(os.path.join(out_path, "utt2spk"), "w")
                }
                if dtype == "word":
                    files["segments"] = open(os.path.join(out_path, "segments"), "w")
                
                handles[group][dtype] = files

        # --- 2. Process SENTENCES (S1 & S2) ---
        print("Processing Sentences (S1 and S2)...")
        for folder_path, folder_name in [(dir_s1, "S1"), (dir_s2, "S2")]:
            if not os.path.isdir(folder_path): continue

            wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
            for wav_path in wav_files:
                filename = os.path.basename(wav_path)
                speaker_id = get_speaker_from_filename(filename)
                if not speaker_id: continue
                
                group = "control" if speaker_id.startswith("N") else "pathological"
                transcription = SENTENCE_TEXTS.get(folder_name, "")
                if not transcription: continue 

                utt_id = f"{speaker_id}_{folder_name}"
                
                h = handles[group]["utterances"]
                h["wav_scp"].write(f"{utt_id} {wav_path}\n")
                h["text"].write(f"{utt_id} {transcription}\n")
                h["utt2spk"].write(f"{utt_id} {speaker_id}\n")
                speakers_found[group]["utterances"].add(speaker_id)

        # --- 3. Process WORDS (DIA) ---
        print("Processing Words (DIA)...")
        if os.path.isdir(dir_dia):
            textgrid_files = glob.glob(os.path.join(dir_dia, "*.TextGrid"))
            
            for tg_path in textgrid_files:
                filename = os.path.basename(tg_path)
                base_name = os.path.splitext(filename)[0]
                
                wav_path = os.path.join(dir_dia, base_name + ".wav")
                if not os.path.exists(wav_path): continue

                speaker_id = get_speaker_from_filename(filename)
                if not speaker_id: continue

                group = "control" if speaker_id.startswith("N") else "pathological"

                intervals = parse_textgrid_intervals(tg_path)
                if not intervals: 
                    print(f"Warning: No intervals found in {tg_path}")
                    continue

                recording_id = base_name 
                h = handles[group]["word"]
                h["wav_scp"].write(f"{recording_id} {wav_path}\n")
                speakers_found[group]["word"].add(speaker_id)

                count = 0
                for start, end, text in intervals:
                    count += 1
                    utt_id = f"{recording_id}_{count:03d}"
                    h["segments"].write(f"{utt_id} {recording_id} {start:.3f} {end:.3f}\n")
                    h["text"].write(f"{utt_id} {text}\n")
                    h["utt2spk"].write(f"{utt_id} {speaker_id}\n")

    finally:
        for group in handles:
            for dtype in handles[group]:
                for f in handles[group][dtype].values():
                    f.close()

    # --- 4. Write Scores & Age ---
    print("Writing scores and ages...")
    for group in speaker_groups:
        for dtype in data_types:
            out_path = os.path.join(base_output_dir, "copas", group, dtype)
            
            spk2score_path = os.path.join(out_path, "spk2score")
            spk2age_path = os.path.join(out_path, "spk2age")
            
            with open(spk2score_path, "w") as f_score, open(spk2age_path, "w") as f_age:
                unique_speakers = sorted(list(speakers_found[group][dtype]))
                for spk in unique_speakers:
                    # Write Score
                    score = "Control" if group == "control" else metadata_scores.get(spk, "N/A")
                    f_score.write(f"{spk} {score}\n")
                    
                    # Write Age (Applicable to both Control and Pathological)
                    age = metadata_ages.get(spk, "N/A")
                    f_age.write(f"{spk} {age}\n")

    print(f"Processing complete. Data saved to: {os.path.join(base_output_dir, 'copas')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copas_root", default="/data/group1/z40484r/datasets/COPAS_1.0.1/Data/Data")
    parser.add_argument("--metadata_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/severity_data_2_test.csv")
    parser.add_argument("--age_path", default="/data/group1/z40484r/datasets/COPAS_1.0.1/age_test.csv")
    parser.add_argument("--output_dir", default="datasets")
    args = parser.parse_args()
    
    extract_copas_paths(args.copas_root, args.metadata_path, args.age_path, args.output_dir)