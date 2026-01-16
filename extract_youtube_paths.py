import os
import argparse
import glob
import pandas as pd
import numpy as np
import math

# ==========================================
# 1. SCORE PROCESSING
# ==========================================

def load_utterance_scores(csv_path):
    """
    Reads the utterance-level score CSV.
    Expected format: utterance_id, score
    Returns: { utterance_id (str): score (float) }
    """
    print(f"Loading utterance scores from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: Score file not found at {csv_path}")
        return {}

    try:
        # Assumes header exists. Adjust if no header.
        df = pd.read_csv(csv_path)
        
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Identify columns
        # We look for 'utterance_id' and 'score'
        id_col = next((c for c in df.columns if 'utterance' in c or 'id' in c), None)
        score_col = next((c for c in df.columns if 'score' in c), None)

        if not id_col or not score_col:
            print(f"Error: Could not identify ID or Score columns. Found: {df.columns}")
            return {}

        score_map = {}
        for _, row in df.iterrows():
            utt_id = str(row[id_col]).strip()
            # If utterance ID in CSV has extension, remove it
            utt_id = os.path.splitext(utt_id)[0]
            
            try:
                val = float(row[score_col])
                score_map[utt_id] = val
            except ValueError:
                continue # Skip non-numeric scores
                
        print(f"Loaded scores for {len(score_map)} utterances.")
        return score_map

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

def load_transcriptions(transcription_path):
    """
    Reads a tab-separated transcription file.
    Format: utterance_id [TAB] text
    Returns: { utterance_id: text }
    """
    print(f"Loading transcriptions from: {transcription_path}")
    trans_map = {}
    
    if not os.path.exists(transcription_path):
        print("Warning: Transcription file not found.")
        return {}
        
    try:
        with open(transcription_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    utt_id = parts[0].strip()
                    text = parts[1].strip()
                    trans_map[utt_id] = text
                elif len(parts) == 1:
                    # Handle cases with empty text or missing tab
                    utt_id = parts[0].strip()
                    trans_map[utt_id] = "<UNK>"
    except Exception as e:
        print(f"Error reading transcriptions: {e}")
        
    return trans_map

def extract_speaker_id(filename):
    """
    Extracts speaker ID from filename based on fixed position.
    Format example: 10030012.wav
    Speaker ID is the 3rd and 4th digits (0-indexed indices 2 and 3).
    Example: 10030012 -> '03'
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Validation: Ensure it's long enough and numeric
    if len(name) >= 4 and name[:4].isdigit():
        return name[2:4]
    else:
        # Fallback or error
        return None

# ==========================================
# 2. MAIN PROCESSING
# ==========================================

def process_spon(audio_root, score_root, trans_root, output_dir):
    
    # 1. Load Data
    utt_csv_path = os.path.join(score_root, "naive_scores.csv")
    utt_scores = load_utterance_scores(utt_csv_path)
    
    trans_file = os.path.join(trans_root, "new_transcription_2020_09_15_v4") 
    # If the file has a .txt extension or similar, adjust here. Assuming no ext based on prompt.
    if not os.path.exists(trans_file):
        # Try adding .txt just in case
        if os.path.exists(trans_file + ".txt"):
            trans_file = trans_file + ".txt"
            
    transcriptions = load_transcriptions(trans_file)

    # 2. Setup Output
    # The prompt implies a single dataset, but we usually separate by subset if applicable.
    # Here we will put everything into a 'spon' folder.
    
    out_path = os.path.join(output_dir, "youtube")
    os.makedirs(out_path, exist_ok=True)
    
    files = {
        "wav_scp": open(os.path.join(out_path, "wav.scp"), "w", encoding="utf-8"),
        "text": open(os.path.join(out_path, "text"), "w", encoding="utf-8"),
        "utt2spk": open(os.path.join(out_path, "utt2spk"), "w", encoding="utf-8"),
        "utt2score": open(os.path.join(out_path, "utt2score"), "w", encoding="utf-8"),
        "spk2score": open(os.path.join(out_path, "spk2score"), "w", encoding="utf-8"),
    }
    
    # Accumulator for speaker averaging
    spk_score_accumulator = {}

    print(f"Scanning audio directory: {audio_root}...")
    all_wavs = glob.glob(os.path.join(audio_root, "*.wav"))
    
    count = 0
    missing_score_count = 0

    for wav_path in sorted(all_wavs):
        filename = os.path.basename(wav_path)
        utt_id = os.path.splitext(filename)[0]
        
        # 1. Get Speaker ID
        spk_id = extract_speaker_id(filename)
        if not spk_id:
            print(f"Skipping malformed filename: {filename}")
            continue
            
        # 2. Get Utterance Score
        # Check raw ID
        score = utt_scores.get(utt_id)
        
        if score is None:
            # Check if CSV might have used int ID vs string ID mismatch
            # or try stripping leading zeros if CSV format differs
            # For now, we assume exact match or skip
            missing_score_count += 1
            continue # Skip if no score found (as requested: "Remove utterances with N/A")

        # 3. Get Transcript
        text = transcriptions.get(utt_id, "<UNK>")
        
        # 4. Write Entry
        files["wav_scp"].write(f"{utt_id} {os.path.abspath(wav_path)}\n")
        files["text"].write(f"{utt_id} {text}\n")
        files["utt2spk"].write(f"{utt_id} {spk_id}\n")
        files["utt2score"].write(f"{utt_id} {score}\n")
        
        # 5. Accumulate for Speaker Mean
        if spk_id not in spk_score_accumulator:
            spk_score_accumulator[spk_id] = []
        spk_score_accumulator[spk_id].append(score)
        
        count += 1

    # Finalize Speaker Scores
    for spk_id in sorted(spk_score_accumulator.keys()):
        scores = spk_score_accumulator[spk_id]
        if scores:
            avg = sum(scores) / len(scores)
            files["spk2score"].write(f"{spk_id} {avg:.2f}\n")
        else:
            # Should not happen given logic above
            pass

    # Cleanup
    for f in files.values():
        f.close()

    print(f"Processing complete.")
    print(f"Processed {count} utterances.")
    print(f"Skipped {missing_score_count} utterances missing scores.")
    print(f"Output saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default paths based on prompt
    parser.add_argument("--audio_root", default="../spon_severity/data/spon_youtube_clean/transcription_subset", 
                        help="Path to directory containing WAV files")
    parser.add_argument("--score_root", default="../spon_severity/data/spon_youtube_clean", 
                        help="Path to directory containing CSV score files")
    parser.add_argument("--trans_root", default="../spon_severity/data/spon_youtube_clean/", 
                        help="Path to directory containing transcription file")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    
    args = parser.parse_args()
    
    process_spon(args.audio_root, args.score_root, args.trans_root, args.output_dir)