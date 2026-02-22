import parselmouth
from pathbench.evaluator import ReferenceFreeEvaluator, ReferenceFreeSpeakerEvaluator
from typing import List, Optional, Tuple
import numpy as np
import librosa


class StdPitchEvaluator(ReferenceFreeEvaluator):
    """An evaluator that computes the standard deviation of the pitch in semitones."""

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
            return self._score_audio(y, sr)
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return None

    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        try:
            if audio is None or len(audio) == 0:
                return 0.0

            sound = parselmouth.Sound(audio, sampling_frequency=fs)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']

            # Filter out unvoiced frames
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) < 2:
                print("Warning: Not enough voiced frames to calculate std of pitch. Returning 0.")
                return 0.0

            pitch_semitones = 39.86 * np.log10(pitch_values)
            return np.std(pitch_semitones)
        except Exception as e:
            print(f"Error computing StdPitch: {e}")
            return None


class F0RangeEvaluator(ReferenceFreeSpeakerEvaluator):
    """An evaluator that computes the F0 range for a speaker."""

    def score(
        self,
        audio_files: List[Tuple[str, float, float]],
    ) -> Optional[float]:
        audios = []
        for (audio_path, start_time, end_time) in audio_files:
            try:
                duration = end_time - start_time if end_time != -1 else None
                y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
                if y is not None and len(y) > 0:
                    audios.append((y, sr))
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")

        if not audios:
            return None
        return self._score_audio_list(audios)

    def _score_audio_list(
        self, audios: List[Tuple[np.ndarray, int]]
    ) -> Optional[float]:
        f0_values = []
        for y, sr in audios:
            try:
                if y is None or len(y) == 0:
                    continue
                sound = parselmouth.Sound(y, sampling_frequency=sr)
                pitch = sound.to_pitch()
                f0 = pitch.selected_array['frequency']
                f0_values.extend(f0[f0 > 0])
            except Exception as e:
                print(f"Error processing audio: {e}")

        if not f0_values:
            print("No valid F0 values found. Returning 0.")
            return 0

        f0_range = np.max(f0_values) - np.min(f0_values)
        return f0_range
