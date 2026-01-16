from pathlib import Path
from typing import Dict, Iterator, Any

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

class Dataset:
    """Handles a speech dataset in a Kaldi-style format."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")

        # Load language from file or default to 'en'
        lang_file = self.path / "language"
        if lang_file.exists():
            self.language = lang_file.read_text().strip()
        else:
            self.language = "en"
            print(
                f"Warning: 'language' file not found in {self.path}. Defaulting to 'en'."
            )

        self.segments = self._load_if_exists("segments", 4)
        self.wav_scp = {key: value[0] for key, value in self._load_if_exists("wav.scp", 2).items()}
        self.text = {key: value[0] for key, value in self._load_if_exists("text", 2).items()}
        self.utt2spk = {key: value[0] for key, value in self._load_if_exists("utt2spk", 2).items()}
        self.spk2score = self._load_scores_if_exists("spk2score")
        self.utt2score = self._load_scores_if_exists("utt2score")


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
                    scores[key] = float(score)
        return scores

    def __iter__(self) -> Iterator[tuple[str, str, str]]:
        """Iterates over utterances, yielding utterance ID, audio path, and transcription."""
        if self.segments:
            for utt_id, seg_info in self.segments.items():
                rec_id, start, end = seg_info
                audio_path = self.wav_scp.get(rec_id)
                if audio_path:
                    transcription = self.text.get(utt_id, "")
                    yield utt_id, audio_path, transcription
        else:
            for utt_id, audio_path in self.wav_scp.items():
                transcription = self.text.get(utt_id, "")
                yield utt_id, audio_path, transcription

    def get_utterances(self):
        """Returns a list of utterance IDs."""
        if self.segments:
            return list(self.segments.keys())
        return list(self.wav_scp.keys())
