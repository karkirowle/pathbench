import os
import argparse
import csv
import glob


def map_letter_to_score(letter):
    """
    Maps Frenchay letter grades to numerical values:
    a -> 1, e -> 5, with intermediates (c/d -> 3.5).
    """
    # Normalize: lowercase, remove all spaces (handles "c / d")
    key = letter.lower().replace(" ", "").strip()
    
    mapping = {
        "a": "1",
        "a/b": "1.5",
        "b": "2",
        "b/c": "2.5",
        "c": "3",
        "c/d": "3.5",
        "d": "4",
        "d/e": "4.5",
        "e": "5"
    }
    # Return mapped value, or original if not found
    return mapping.get(key, letter)

def get_speaker_scores(speaker_dir):
    """
    Parses the .csv file in the speaker's 'Notes' directory to find
    intelligibility scores and converts them to numbers.
    """
    notes_dir = os.path.join(speaker_dir, "Notes")
    
    if not os.path.isdir(notes_dir):
        return None

    csv_files = glob.glob(os.path.join(notes_dir, "*.csv"))
    
    if not csv_files:
        return None
    
    csv_path = csv_files[0]
    scores = {"word": "N/A", "utterances": "N/A"}
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                cleaned_row = [x.strip() for x in row]
                
                # Logic: Find "Words", map the letter after it
                if "Words" in cleaned_row:
                    try:
                        idx = cleaned_row.index("Words")
                        if idx + 1 < len(cleaned_row) and cleaned_row[idx+1]:
                            scores["word"] = map_letter_to_score(cleaned_row[idx + 1])
                    except ValueError:
                        pass

                # Logic: Find "Sentences", map the letter after it
                if "Sentences" in cleaned_row:
                    try:
                        idx = cleaned_row.index("Sentences")
                        if idx + 1 < len(cleaned_row) and cleaned_row[idx+1]:
                            scores["utterances"] = map_letter_to_score(cleaned_row[idx + 1])
                    except ValueError:
                        pass
                        
        return scores
    except Exception as e:
        print(f"Warning: Could not parse CSV {csv_path}: {e}")
        return None

def extract_torgo_paths(torgo_root, base_output_dir):
    """
    Extracts sentence and word audio paths from TORGO, filters bad transcripts,
    and extracts intelligibility scores from clinical notes.
    """
    
    # --- Configuration ---
    pathological_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    control_speakers = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]

    speaker_groups = {
        "pathological": pathological_speakers,
        "control": control_speakers
    }

    data_types = ["word", "utterances"]
    
    # Structure: handles[group][data_type][file_key]
    handles = {}
    
    # Structure: speaker_scores[group][data_type][speaker_id] = score
    speaker_scores = {}

    try:
        # --- 1. Setup Output Directories and Files ---
        for group in speaker_groups:
            handles[group] = {}
            speaker_scores[group] = {"word": {}, "utterances": {}}
            
            for dtype in data_types:
                out_path = os.path.join(base_output_dir, "torgo", group, dtype)
                os.makedirs(out_path, exist_ok=True)
                
                handles[group][dtype] = {
                    "wav_scp": open(os.path.join(out_path, "wav.scp"), "w"),
                    "text": open(os.path.join(out_path, "text"), "w"),
                    "utt2spk": open(os.path.join(out_path, "utt2spk"), "w")
                }

        # --- 2. Process Speakers ---
        all_speakers = pathological_speakers + control_speakers
        
        for speaker in all_speakers:
            if speaker in pathological_speakers:
                group = "pathological"
            elif speaker in control_speakers:
                group = "control"
            else:
                continue 

            speaker_dir = os.path.join(torgo_root, speaker)
            if not os.path.isdir(speaker_dir):
                print(f"Speaker directory not found: {speaker_dir}")
                continue

            # --- Extract Scores (Once per Speaker) ---
            # CHANGED: Moved out of the session loop
            if group == "pathological":
                scores = get_speaker_scores(speaker_dir)
                if scores:
                    if scores["word"] != "N/A":
                        speaker_scores[group]["word"][speaker] = scores["word"]
                    if scores["utterances"] != "N/A":
                        speaker_scores[group]["utterances"][speaker] = scores["utterances"]
            else:
                # Default for control speakers
                speaker_scores[group]["word"][speaker] = "Control"
                speaker_scores[group]["utterances"][speaker] = "Control"

            # --- Process Sessions ---
            for session in os.listdir(speaker_dir):
                if not session.startswith("Session"):
                    continue
                
                session_dir = os.path.join(speaker_dir, session)
                prompts_dir = os.path.join(session_dir, "prompts")
                wav_dir = os.path.join(session_dir, "wav_arrayMic")

                if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir):
                    continue

                for prompt_file in os.listdir(prompts_dir):
                    if not prompt_file.endswith(".txt"):
                        continue
                    
                    # Read transcription
                    prompt_path = os.path.join(prompts_dir, prompt_file)
                    try:
                        with open(prompt_path, "r") as f:
                            transcription = f.read().strip()
                    except Exception:
                        continue

                    if not transcription:
                        continue

                    # --- FILTER: Remove bad characters ---
                    if "[" in transcription or "/" in transcription:
                        continue

                    # --- Classify: Word vs Utterance ---
                    if " " in transcription:
                        dtype = "utterances"
                    else:
                        dtype = "word"

                    # Construct IDs and Paths
                    prompt_basename = os.path.splitext(prompt_file)[0]
                    utterance_id = f"{speaker}_{session}_{prompt_basename}"
                    
                    wav_filename = f"{prompt_basename}.wav"
                    wav_path = os.path.join(wav_dir, wav_filename)

                    if not os.path.exists(wav_path):
                        wav_filename_alt = f"{speaker}_{session}_{prompt_basename}.wav"
                        wav_path_alt = os.path.join(wav_dir, wav_filename_alt)
                        if os.path.exists(wav_path_alt):
                            wav_path = wav_path_alt
                        else:
                            continue
                    
                    # Write to files
                    current_handles = handles[group][dtype]
                    current_handles["wav_scp"].write(f"{utterance_id} {wav_path}\n")
                    current_handles["text"].write(f"{utterance_id} {transcription}\n")
                    current_handles["utt2spk"].write(f"{utterance_id} {speaker}\n")

    finally:
        # --- 3. Close Main Files ---
        for group in handles:
            for dtype in handles[group]:
                for f in handles[group][dtype].values():
                    f.close()

    # --- 4. Write Score Files (spk2score) ---
    for group in speaker_groups:
        for dtype in data_types:
            out_path = os.path.join(base_output_dir, "torgo", group, dtype)
            spk2score_path = os.path.join(out_path, "spk2score")
            
            with open(spk2score_path, "w") as f:
                # Iterate through speakers we found in this group
                found_speakers = speaker_scores[group][dtype]
                
                # Sort for tidiness
                for spk in sorted(speaker_groups[group]):
                    score = found_speakers.get(spk, "N/A")
                    f.write(f"{spk} {score}\n")

    print(f"Processing complete. Data saved to: {os.path.join(base_output_dir, 'torgo')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TORGO dataset (Filter bad chars & Extract Scores).")
    parser.add_argument("--torgo_root", default="/data/group1/z40484r/datasets/TORGO", 
                        help="Path to TORGO root directory")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    
    args = parser.parse_args()
    
    extract_torgo_paths(args.torgo_root, args.output_dir)