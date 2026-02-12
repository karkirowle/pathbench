### Neurovoz Dataset Preparation

We processed the Neurovoz dataset, which comprises recordings of Spanish speakers with Parkinson's Disease (PD) alongside Healthy Controls (HC). Our preprocessing pipeline specifically targeted continuous sentence reading tasks.

To ensure sufficient data density per speaker, the pathological cohort was filtered using a predefined inclusion list, removing PD speakers with excessively sparse recordings. The full control cohort was retained.

#### Task Definition and Transcription

For each selected recording, intended transcriptions were sourced from corresponding canonical text files. This setup ensures that models evaluate deviations against a standardized phonetic target. All valid transcripts were mapped to their respective audio files, alongside speaker-level demographic metadata (age and gender) extracted from the dataset's clinical CSV files.

#### Intelligibility Scores

We integrated perceptual intelligibility evaluations derived from the metadata. Raw assessments, which included both categorical descriptors (e.g., "normal", "mild", "moderate", "severe") and varied numerical formats, were normalized into a uniform 4-point intelligibility scale: 3 (Normal), 2 (Mild deficiency), 1 (Moderate deficiency), and 0 (Severe deficiency). Scores were assigned at the utterance level, and speaker-level intelligibility was subsequently computed as the average of their respective utterance scores.