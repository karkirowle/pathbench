### TORGO Dataset Preparation

We processed the TORGO database of dysarthric speech to construct standardized subsets for our analysis. Our processing isolated both **word** and **sentence (utterance)** production tasks. To maintain acoustic consistency, we specifically utilized recordings captured via the array microphone (`wav_arrayMic`).

#### Task Definition and Transcription

For each recording, intended transcriptions were extracted directly from the corresponding textual prompt files. Tasks were categorized into "words" (single tokens) and "utterances" (transcripts containing spaces). 

Recordings with transcripts containing specific non-speech or annotation artifacts (such as `[` or `/`) were systematically excluded. Consequently, this exclusion step filtered out sustained vowels and the majority of prompt-based spontaneous speech tasks.

All valid transcripts were normalized by converting to lowercase and stripping punctuation to maintain strict consistency across control and pathological cohorts.

#### Intelligibility Scores

We mapped the clinical Frenchay Dysarthria Assessment letter grades (e.g., A, A/B, B) provided in the dataset's clinical notes to a continuous numerical intelligibility score (ranging from 5.0 to 1.0) specific to both word and sentence levels. Specifically, a grade of 'A' was mapped to 5.0, 'A/B' to 4.5, and so forth, decreasing linearly in increments of 0.5 down to 'E', which was mapped to 1.0.