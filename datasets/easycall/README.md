### EasyCall Dataset Preparation

We processed the EasyCall dataset, which comprises recordings of Italian speakers with dysarthria alongside healthy controls. Our preprocessing pipeline categorized the speech recordings into isolated **word** and continuous **sentence (utterance)** production tasks based on the presence of spaces in the intended transcripts.

#### Task Definition and Transcription

The recorded prompts consist of typical smartphone voice commands and navigation phrases (e.g., "Apri rubrica", "Chiama emergenza"). To establish standardized phonetic targets, we mapped all recordings to a predefined dictionary of canonical, intended transcriptions.

All valid transcripts were subsequently normalized by converting to lowercase, and stripping spaces, punctuation, and apostrophes to maintain strict consistency across the control and pathological cohorts.

#### Intelligibility Scores

Speaker-level intelligibility scores were provided with the dataset which we flipped in directionality to match with the other datasets.