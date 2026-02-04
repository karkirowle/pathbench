import parselmouth
from pathbench.evaluator import SpeakerEvaluator, Evaluator
from typing import List, Optional, Tuple
import numpy as np
import librosa

class F0RangeEvaluator(SpeakerEvaluator):
    """An evaluator that computes the F0 range for a speaker."""

    def score(
        self,
        audio_files: List[Tuple[str, float, float]],
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

class StdPitchEvaluator(Evaluator):
    """An evaluator that computes the standard deviation of the pitch in semitones."""

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the standard deviation of the pitch for an utterance.
        """
        try:
            duration = end_time - start_time if end_time != -1 else None
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            
            # Filter out unvoiced frames
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) < 2:
                print(f"Warning: Not enough voiced frames to calculate std of pitch for {audio_path}. Returning 0.")
                return 0.0

            mean_pitch = np.mean(pitch_values)
            std_pitch = np.std(pitch_values)
            
            if mean_pitch == 0:
                # This case should not be reached if pitch_values has positive values.
                print(f"Warning: Mean pitch is 0 for {audio_path}. Returning 0.")
                return 0.0

            semitone = 39.86 * np.log10((mean_pitch + std_pitch) / mean_pitch)
            return semitone
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return None
