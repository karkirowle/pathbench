from pathlib import Path
from typing import Dict, Iterator, Any, Optional, List
import numpy as np
from pathbench.string_clean import clean_text


def _load_kaldi_style_file(file_path: Path, num_parts: int) -> Dict[str, list[str]]:
    """Loads a Kaldi-style file (e.g., wav.scp, text, utt2spk) into a dictionary."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=num_parts - 1)
            if len(parts) == num_parts:
                key, *values = parts
                data[key] = values
    return data

PHONEMISER_LANG_MAPPING = {
    "en": "en-us",
    "nl": "nl",
    "it": "it",
    "es": "es",
}


class Dataset:
    """Handles a speech dataset in a Kaldi-style format."""

    def __init__(self, dataset_path: str, use_reference: bool = False,
                 reference_path: str = None,
                 reference_type: str = 'control',
                 reference_mapping: dict = None):
        self.path = Path(dataset_path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")

        # Load language from file or default to 'en'
        lang_file = self.path / "language"
        if lang_file.exists():
            lang = lang_file.read_text().strip()
        else:
            lang = "en"
            print(
                f"Warning: 'language' file not found in {self.path}. Defaulting to 'en'."
            )
        
        self.language = PHONEMISER_LANG_MAPPING.get(lang, lang)

        self.segments = self._load_if_exists("segments", 4)
        self.wav_scp = {key: value[0] for key, value in self._load_if_exists("wav.scp", 2).items()}
        self.text = {key: value[0] for key, value in self._load_if_exists("text", 2).items()}
        self.utt2spk = {key: value[0] for key, value in self._load_if_exists("utt2spk", 2).items()}
        self.spk2score = self._load_scores_if_exists("spk2score")
        self.utt2score = self._load_scores_if_exists("utt2score")
        self.spk2gender = {key: value[0] for key, value in self._load_if_exists("spk2gender", 2).items()}
        
        self.use_reference = use_reference
        self.reference_path = reference_path
        self.reference_type = reference_type
        self.reference_mapping = reference_mapping
        self.reference_dataset = None

        if self.use_reference:
            if self.reference_type == "none":
                pass
            elif self.reference_type in ['control', 'all']:
                if not self.reference_path:
                    raise ValueError('reference_path is required for control/all reference types')
                self.reference_dataset = Dataset(self.reference_path)
            elif self.reference_type == 'custom' and not self.reference_mapping:
                raise ValueError('reference_mapping is required for custom reference type')


    def _load_if_exists(self, filename: str, num_parts: int) -> Dict[str, list[str]]:
        file_path = self.path / filename
        if file_path.exists():
            return _load_kaldi_style_file(file_path, num_parts)
        return {}

    def _load_scores_if_exists(self, filename: str) -> Dict[str, float]:
        file_path = self.path / filename
        scores = {}
        if file_path.exists():
            with open(file_path, 'r') as f:
                for line in f:
                    key, score = line.strip().split()
                    if score == 'N/A':
                        print("Warning: Found 'N/A' score for key:", key)
                        scores[key] = np.nan
                    else:
                        scores[key] = float(score)
        return scores

    def __iter__(self) -> Iterator[tuple[str, str, str, Optional[List[str]], float, float]]:
        """
        Iterates over utterances, yielding utterance ID, audio path, transcription,
        a list of reference audio paths, start time, and end time.
        """
        if self.segments:
            utt_ids = list(self.segments.keys())
        else:
            utt_ids = list(self.wav_scp.keys())

        for utt_id in utt_ids:
            start_time, end_time = 0.0, -1.0
            if self.segments:
                if utt_id in self.segments:
                    rec_id, start_str, end_str = self.segments[utt_id]
                    audio_path = self.wav_scp.get(rec_id)
                    start_time = float(start_str)
                    end_time = float(end_str)
                else:
                    continue
            else:
                audio_path = self.wav_scp.get(utt_id)

            if not audio_path:
                continue

            transcription = self.text.get(utt_id, "")

            reference_audio_paths = None
            if self.use_reference:
                reference_audio_paths = self._get_reference_audios(utt_id, transcription)
            print("gets here")
            yield utt_id, audio_path, transcription, reference_audio_paths, start_time, end_time

    def _get_reference_audios(self, utt_id: str, transcription: str) -> List[tuple[str, float, float]]:
        if self.reference_type == "control":
            return self._load_same_text_references(utt_id, transcription)
        elif self.reference_type == "all":
            current_speaker = self.utt2spk.get(utt_id)
            return self._load_all_same_text_references(transcription, current_speaker)
        elif self.reference_type == "custom":
            return self._load_custom_references(utt_id)
        else:
            raise ValueError(f"Unsupported reference_type: {self.reference_type}")

    def _load_same_text_references(self, utt_id: str, transcription: str) -> List[tuple[str, float, float]]:
        """
        Loads reference audios from control speakers with the same transcription and gender.
        """
        if not self.reference_dataset:
            return []

        current_speaker = self.utt2spk.get(utt_id)
        if not current_speaker:
            return []
        
        current_gender = self.spk2gender.get(current_speaker)
        if not current_gender:
            return []

        ref_paths = []
        cleaned_transcription = clean_text(transcription)
        for ref_utt_id, ref_trans in self.reference_dataset.text.items():
            #print("check sameness:", clean_text(ref_trans), cleaned_transcription)
            if clean_text(ref_trans) == cleaned_transcription:
                #print("same")
                ref_speaker = self.reference_dataset.utt2spk.get(ref_utt_id)
                if not ref_speaker:
                    continue
                
                ref_gender = self.reference_dataset.spk2gender.get(ref_speaker)
                if ref_gender != current_gender:
                    continue

                start_time, end_time = 0.0, -1.0
                if self.reference_dataset.segments:
                    if ref_utt_id in self.reference_dataset.segments:
                        rec_id, start_str, end_str = self.reference_dataset.segments[ref_utt_id]
                        audio_path = self.reference_dataset.wav_scp.get(rec_id)
                        start_time = float(start_str)
                        end_time = float(end_str)
                    else:
                        continue
                else:
                    audio_path = self.reference_dataset.wav_scp.get(ref_utt_id)
                
                if audio_path:
                    ref_paths.append((audio_path, start_time, end_time))
        print(f"Utterance: {utt_id}, References: {ref_paths}")
        return ref_paths

    def _load_all_same_text_references(self, transcription: str, current_speaker: str) -> List[tuple[str, float, float]]:
        """
        Loads all reference audios with the same transcription from different speakers,
        from both the main dataset and the reference dataset.
        """
        ref_paths = []

        # Search in the main dataset (e.g., pathological)
        ref_paths.extend(Dataset._find_matching_references_in_dataset(self, transcription, current_speaker))

        # Search in the reference dataset (e.g., control)
        if self.reference_dataset:
            ref_paths.extend(Dataset._find_matching_references_in_dataset(self.reference_dataset, transcription, current_speaker))
        
        return ref_paths

    @staticmethod
    def _find_matching_references_in_dataset(dataset, transcription: str, current_speaker: str) -> List[tuple[str, float, float]]:
        paths = []
        for utt_id, trans in dataset.text.items():
            if trans == transcription:
                speaker = dataset.utt2spk.get(utt_id)
                # The speaker check should be against the original utterance speaker
                if speaker != current_speaker:
                    start_time, end_time = 0.0, -1.0
                    if dataset.segments:
                        if utt_id in dataset.segments:
                            rec_id, start_str, end_str = dataset.segments[utt_id]
                            path = dataset.wav_scp.get(rec_id)
                            start_time = float(start_str)
                            end_time = float(end_str)
                        else:
                            continue
                    else:
                        path = dataset.wav_scp.get(utt_id)
                    
                    if path:
                        paths.append((path, start_time, end_time))
        return paths

    def _load_custom_references(self, utt_id: str) -> List[tuple[str, float, float]]:
        """
        Loads reference audios based on a custom mapping.
        Assumes the mapping is from utterance ID to a list of audio file paths.
        Segments are not supported for custom references.
        """
        if not self.reference_mapping:
            return []
        
        paths = self.reference_mapping.get(utt_id, [])
        return [(path, 0.0, -1.0) for path in paths]

    def get_utterances(self):
        """Returns a list of utterance IDs."""
        if self.segments:
            return list(self.segments.keys())
        return list(self.wav_scp.keys())
