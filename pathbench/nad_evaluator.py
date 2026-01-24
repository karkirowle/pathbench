from typing import List, Optional

import numpy as np
from dtw import dtw
import torch
from transformers import Wav2Vec2Model
import librosa

from pathbench.reference_evaluator import ReferenceEvaluator


def load_wav2vec2_featurizer(model_name, layer):
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _featurize(path, start_time=0.0, end_time=-1.0):
        duration = end_time - start_time if end_time != -1 else None
        input_values, _ = librosa.load(path, sr=16000, mono=True, offset=start_time, duration=duration)
        input_values = torch.from_numpy(input_values).unsqueeze(0).to(device)
        with torch.no_grad():
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
        return hidden_states[layer].squeeze(0).cpu().numpy()

    return _featurize


class NADEvaluator(ReferenceEvaluator):
    """
    An evaluator that computes the Normalized Alignment Distance (NAD) using DTW
    on wav2vec2 features. Based on the script measure_distance_bence.py.
    """

    def __init__(self, model_id="facebook/wav2vec2-base", layer=9, **kwargs):
        super().__init__(**kwargs)
        self.featurizer = load_wav2vec2_featurizer(model_id, layer)

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        reference_audios: List[tuple[str, float, float]],
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the average DTW distance between the test audio and reference audios.
        """
        try:
            test_feats = self.featurizer(audio_path, start_time, end_time)
        except Exception as e:
            print(f"Error featurizing test audio {audio_path}: {e}")
            return None

        if not reference_audios:
            return None

        distances = []
        for ref_path, ref_start, ref_end in reference_audios:
            try:
                ref_feats = self.featurizer(ref_path, ref_start, ref_end)
            except Exception as e:
                print(f"Error featurizing reference audio {ref_path}: {e}")
                distances.append(np.nan)
                continue

            if test_feats.shape[0] < 2 or ref_feats.shape[0] < 2:
                distances.append(np.nan)
                continue

            try:
                distance = dtw(
                    test_feats,
                    ref_feats,
                    distance_only=True,
                ).normalizedDistance
                distances.append(distance)
            except Exception as e:
                print(f"Error calculating DTW between {audio_path} and {ref_path}: {e}")
                distances.append(np.nan)

        return np.nanmean(distances)
