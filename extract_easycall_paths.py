import os
import argparse
import glob

# ==========================================
# 1. TOM SCORES (User Provided)
# ==========================================
SPEAKER_SCORES = {
    "f01": "1", "f02": "3", "f03": "1", "f05": "1", "f06": "1",
    "f07": "1", "f08": "1", "f09": "1", "f10": "5", "f11": "2",
    "m01": "3", "m02": "1", "m03": "3", "m04": "1", "m05": "4",
    "m06": "4", "m07": "4", "m08": "3", "m09": "1", "m10": "3",
    "m11": "5", "m12": "1", "m13": "1", "m14": "5", "m15": "1",
    "m16": "3", "m17": "1", "m18": "1", "m19": "3", "m20": "1",
    # Controls must be in this list to be processed
    "fc01": "Control", "fc02": "Control", "mc01": "Control", "mc02": "Control"
}

# ==========================================
# 2. OFFICIAL UTTERANCE LIST
# ==========================================
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
# 3. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):
    """
    Removes spaces, punctuation, and casing to create a robust lookup key.
    Ex: "Scorri verso l’alto" -> "scorriversolalto"
    """
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

# ==========================================
# 4. MAIN PROCESSING
# ==========================================

def process_easycall(dataset_root, output_dir):
    
    lookup_table = build_lookup_table(OFFICIAL_TRANSCRIPTS)
    
    groups = ["pathological", "control"]
    handles = {}

    # Setup Output Files
    for g in groups:
        out_path = os.path.join(output_dir, "easycall", g, "commands")
        os.makedirs(out_path, exist_ok=True)
        handles[g] = {
            "wav_scp": open(os.path.join(out_path, "wav.scp"), "w", encoding="utf-8"),
            "text": open(os.path.join(out_path, "text"), "w", encoding="utf-8"),
            "utt2spk": open(os.path.join(out_path, "utt2spk"), "w", encoding="utf-8"),
            "spk2score": open(os.path.join(out_path, "spk2score"), "w", encoding="utf-8"),
            "speakers_found": set()
        }

    print(f"Scanning directory: {dataset_root}...")
    
    all_wavs = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    
    count = 0
    skipped_text_match = 0
    skipped_no_score = 0

    for wav_path in all_wavs:
        filename = os.path.basename(wav_path)
        
        if filename.count("_") < 2:
            continue

        try:
            parts = filename.split("_", 2)
            speaker_id = parts[0]   # m05
            session_id = parts[1]   # 01
            raw_text_part = parts[2] 
            
            # --- NEW CHECK: Exclude speakers without scores ---
            if speaker_id not in SPEAKER_SCORES:
                skipped_no_score += 1
                continue

            raw_text_no_ext = os.path.splitext(raw_text_part)[0]
            
            # 1. Normalize and Lookup
            file_key = normalize_text(raw_text_no_ext)
            official_transcript = lookup_table.get(file_key)
            
            if not official_transcript:
                skipped_text_match += 1
                continue

            # 2. Group Classification
            if speaker_id.lower().startswith(('c', 'fc', 'mc')):
                group = "control"
            else:
                group = "pathological"
            
            # 3. Write Output
            utt_id = f"{speaker_id}_{session_id}_{file_key}"
            
            h = handles[group]
            h["speakers_found"].add(speaker_id)
            
            h["wav_scp"].write(f"{utt_id} {os.path.abspath(wav_path)}\n")
            h["text"].write(f"{utt_id} {official_transcript}\n")
            h["utt2spk"].write(f"{utt_id} {speaker_id}\n")
            
            count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Write Scores and Close Files
    for g in groups:
        h = handles[g]
        for spk in sorted(list(h["speakers_found"])):
            # We already know spk is in SPEAKER_SCORES because we filtered earlier
            score = SPEAKER_SCORES[spk]
            h["spk2score"].write(f"{spk} {score}\n")
            
        h["wav_scp"].close()
        h["text"].close()
        h["utt2spk"].close()
        h["spk2score"].close()

    print(f"Processing complete.")
    print(f"Successfully processed: {count} utterances.")
    if skipped_no_score > 0:
        print(f"Skipped {skipped_no_score} files (Speaker not in score list).")
    if skipped_text_match > 0:
        print(f"Skipped {skipped_text_match} files (Filename did not match official list).")
    print(f"Output saved to: {os.path.join(output_dir, 'easycall')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--easycall_root", default="/data/group1/z40484r/datasets/EasyCall", 
                        help="Path to EasyCall root directory")
    parser.add_argument("--output_dir", default="datasets", 
                        help="Base output directory")
    args = parser.parse_args()
    
    process_easycall(args.easycall_root, args.output_dir)