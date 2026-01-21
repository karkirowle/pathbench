from typing import Optional
import librosa
from pathbench.evaluator import Evaluator


class WpmEvaluator(Evaluator):
    """An evaluator that scores based on the speech rate (words per minute)."""

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Returns the speech rate in words per minute (WPM).
        """
        try:
            # Get audio duration
            duration_s = librosa.get_duration(path=audio_path)

            if duration_s == 0:
                return 0.0

            # Count words in transcription
            word_count = len(transcription.split())

            if word_count == 0:
                return 0.0

            # Calculate WPM
            wpm = (word_count / duration_s) * 60
            return wpm
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return None
