from typing import List, Optional

import numpy as np
from dtw import dtw
import torch
from transformers import Wav2Vec2Model
import librosa

from pathbench.evaluator import ReferenceAudioEvaluator, ReferenceTxtAndAudioEvaluator
from pathbench.vad import FATrimmer


def load_wav2vec2_featurizer(model_name, layer):
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _featurize(audio_data):
        # Ensure audio_data is float32, not float64
        if audio_data.dtype == np.float64:
            audio_data = audio_data.astype(np.float32)
        input_values = torch.from_numpy(audio_data).unsqueeze(0).to(device)
        with torch.no_grad():
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
        return hidden_states[layer].squeeze(0).cpu().numpy()

    return _featurize


class NADEvaluator(ReferenceAudioEvaluator):
    """
    An evaluator that computes the Normalized Alignment Distance (NAD) using DTW
    on wav2vec2 features.
    """

    def __init__(self, model_id="facebook/wav2vec2-large", layer=10):
        self.featurizer = load_wav2vec2_featurizer(model_id, layer)
        self.min_feature_len = 2 # DTW requires at least 2 feature vectors

    def _get_features(self, audio_path, start_time, end_time):
        """Helper to load and featurize an audio file."""
        # 1. Load audio
        audio = None
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            offset = start_time if start_time != 0.0 else 0
            audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

            if audio is None or len(audio) == 0:
                return None, f"Audio at {audio_path} could not be loaded or is empty."

            # 2. Featurize
            features = self.featurizer(audio)
            if features.shape[0] < self.min_feature_len:
                return None, f"Feature length for {audio_path} is {features.shape[0]}, which is less than minimum {self.min_feature_len}."

            return features, None

        except Exception as e:
            return None, f"Failed to process {audio_path}: {e}"

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        reference_audios: List[tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """Computes the average DTW distance between test and reference audio."""
        if not reference_audios:
            return None

        test_feats, err = self._get_features(audio_path, start_time, end_time)
        if err:
            print(f"Error: Failed to get features for test audio {utterance_id}: {err}")
            return None

        ref_feats = []
        for ref_path, ref_start, ref_end in reference_audios:
            r_feats, err = self._get_features(ref_path, ref_start, ref_end)
            if err:
                print(f"Warning: Failed to get features for ref {ref_path} in group {utterance_id}, skipping ref. Error: {err}")
            else:
                ref_feats.append(r_feats)

        # --- Calculate DTW ---
        if test_feats is None or not ref_feats:
            print(f"Error: Could not obtain valid features for DTW calculation for group {utterance_id}.")
            return None

        distances = []
        for r_feats in ref_feats:
            try:
                distance = dtw(test_feats, r_feats, distance_only=True).normalizedDistance
                distances.append(distance)
            except Exception as e:
                # This can happen if, even after all checks, features are problematic (e.g., all zeros)
                print(f"Error during DTW calculation for {utterance_id}: {e}")
                distances.append(np.nan)

        return np.nanmean(distances) if distances else None


class TrimmedNADEvaluator(ReferenceTxtAndAudioEvaluator):
    """
    An evaluator that computes the Normalized Alignment Distance (NAD) using DTW
    on wav2vec2 features. Falls back to untrimmed audio for the whole group if
    trimming or featurization fails for any member of the group.
    """

    def __init__(self, model_id="facebook/wav2vec2-large", layer=10, trimmer: Optional[FATrimmer] = None):
        self.featurizer = load_wav2vec2_featurizer(model_id, layer)
        self.trimmer = trimmer
        self.min_feature_len = 2 # DTW requires at least 2 feature vectors

    def _get_features(self, audio_path, transcription, language, start_time, end_time, use_trimming):
        """Helper to load, optionally trim, and featurize an audio file."""
        use_segment = start_time != 0.0 or end_time != -1.0

        # 1. Load audio (either trimmed or from segment/file)
        audio = None
        try:
            if use_trimming and self.trimmer and not use_segment:
                trimmed_data = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if trimmed_data and len(trimmed_data[0]) > 0:
                    audio, _ = trimmed_data

            if audio is None: # Fallback for failed trim or if trimming is disabled
                duration = end_time - start_time if end_time != -1.0 else None
                offset = start_time if start_time != 0.0 else 0
                audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

            if audio is None or len(audio) == 0:
                return None, f"Audio at {audio_path} could not be loaded or is empty."

            # 2. Featurize
            features = self.featurizer(audio)
            if features.shape[0] < self.min_feature_len:
                return None, f"Feature length for {audio_path} is {features.shape[0]}, which is less than minimum {self.min_feature_len}."

            return features, None

        except Exception as e:
            return None, f"Failed to process {audio_path}: {e}"

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        reference_audios: List[tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """
        Computes the average DTW distance. If trimming/featurizing fails for any audio
        in a group (test or any reference), it falls back to untrimmed for all.
        """
        if not reference_audios:
            return None

        # --- Pass 1: Attempt to get features with trimming enabled ---
        test_feats = None
        ref_feats = []
        errors = []
        use_trimming = True

        # Check if trimming should be attempted at all
        use_test_segment = start_time != 0.0 or end_time != -1.0
        use_ref_segments = any(ref_start != 0.0 or ref_end != -1.0 for _, ref_start, ref_end in reference_audios)
        if not self.trimmer or use_test_segment or use_ref_segments:
            use_trimming = False

        if use_trimming:
            test_feats, err = self._get_features(audio_path, transcription, language, start_time, end_time, use_trimming=True)
            if err: errors.append(err)

            for ref_path, ref_start, ref_end in reference_audios:
                r_feats, err = self._get_features(ref_path, transcription, language, ref_start, ref_end, use_trimming=True)
                if err: errors.append(err)
                ref_feats.append(r_feats) # Append even if None to keep list aligned

            # If any error occurred, discard all results from this pass
            if errors or test_feats is None or any(f is None for f in ref_feats):
                print(f"Warning: Failed to get trimmed features for group {utterance_id}. Falling back to untrimmed. Errors: {errors}")
                test_feats = None
                ref_feats = []
                use_trimming = False # Force fallback
            else:
                ref_feats = [f for f in ref_feats if f is not None] # Clean up list

        # --- Pass 2: Get features with trimming disabled (if pass 1 failed or was skipped) ---
        if not use_trimming:
            test_feats, err = self._get_features(audio_path, transcription, language, start_time, end_time, use_trimming=False)
            if err:
                print(f"Error: Failed to get untrimmed features for test audio {utterance_id}: {err}")
                return None

            ref_feats = []
            for ref_path, ref_start, ref_end in reference_audios:
                r_feats, err = self._get_features(ref_path, transcription, language, ref_start, ref_end, use_trimming=False)
                if err:
                    print(f"Warning: Failed to get untrimmed features for ref {ref_path} in group {utterance_id}, skipping ref. Error: {err}")
                else:
                    ref_feats.append(r_feats)

        # --- Pass 3: Calculate DTW ---
        if test_feats is None or not ref_feats:
            print(f"Error: Could not obtain valid features for DTW calculation for group {utterance_id}.")
            return None

        distances = []
        for r_feats in ref_feats:
            try:
                distance = dtw(test_feats, r_feats, distance_only=True).normalizedDistance
                distances.append(distance)
            except Exception as e:
                # This can happen if, even after all checks, features are problematic (e.g., all zeros)
                print(f"Error during DTW calculation for {utterance_id}: {e}")
                distances.append(np.nan)

        return np.nanmean(distances) if distances else None