import parselmouth
from pathbench.evaluator import SpeakerEvaluator
from typing import List, Optional, Tuple
import numpy as np
import librosa

class F0RangeEvaluator(SpeakerEvaluator):
    """An evaluator that computes the F0 range for a speaker."""

    def score(
        self,
        audio_files: List[Tuple[str, float, float]],
        transcriptions: List[str],
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the F0 range for a speaker based on all their utterances.
        """
        f0_values = []
        for audio_path, start_time, end_time in audio_files:
            try:
                duration = end_time - start_time if end_time != -1 else None
                y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
                sound = parselmouth.Sound(y, sampling_frequency=sr)
                pitch = sound.to_pitch()
                f0 = pitch.selected_array['frequency']
                f0_values.extend(f0[f0 > 0])
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {e}")
        
        if not f0_values:
            print("No valid F0 values found. Returning 0.")
            return 0

        f0_range = np.max(f0_values) - np.min(f0_values)
        return f0_range
