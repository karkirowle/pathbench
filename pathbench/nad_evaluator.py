from typing import List, Optional

import numpy as np
from dtw import dtw
import torch
from transformers import Wav2Vec2Model
import librosa

from pathbench.reference_evaluator import ReferenceEvaluator
from pathbench.vad import FATrimmer


def load_wav2vec2_featurizer(model_name, layer):
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _featurize(audio_data):
        input_values = torch.from_numpy(audio_data).unsqueeze(0).to(device)
        with torch.no_grad():
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
        return hidden_states[layer].squeeze(0).cpu().numpy()

    return _featurize


class NADEvaluator(ReferenceEvaluator):
    """
    An evaluator that computes the Normalized Alignment Distance (NAD) using DTW
    on wav2vec2 features. Based on the script measure_distance_bence.py.
    """

    def __init__(self, model_id="facebook/wav2vec2-base", layer=9, trimmer: Optional[FATrimmer] = None, **kwargs):
        super().__init__(**kwargs)
        self.featurizer = load_wav2vec2_featurizer(model_id, layer)
        self.trimmer = trimmer

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
            use_segment = start_time != 0.0 or end_time != -1.0
            y, sr = None, 16000
            if use_segment:
                duration = end_time - start_time if end_time != -1.0 else None
                y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
            elif self.trimmer:
                trimmed_data = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if trimmed_data:
                    y, sr = trimmed_data
                    if y is not None:
                        y = y.astype(np.float32)
                else:
                    y, sr = librosa.load(audio_path, sr=16000)
            else:
                y, sr = librosa.load(audio_path, sr=16000)
            
            if y is None:
                return None

            test_feats = self.featurizer(y)
        except Exception as e:
            print(f"Error featurizing test audio {audio_path}: {e}")
            return None

        if not reference_audios:
            return None

        distances = []
        for ref_path, ref_start, ref_end in reference_audios:
            try:
                ref_use_segment = ref_start != 0.0 or ref_end != -1.0
                ref_y, ref_sr = None, 16000
                if ref_use_segment:
                    duration = ref_end - ref_start if ref_end != -1.0 else None
                    ref_y, ref_sr = librosa.load(ref_path, sr=16000, offset=ref_start, duration=duration)
                elif self.trimmer:
                    trimmed_data = self.trimmer.trim(ref_path, transcription, language, ref_start, ref_end)
                    if trimmed_data:
                        ref_y, ref_sr = trimmed_data
                        if ref_y is not None:
                            ref_y = ref_y.astype(np.float32)
                    else:
                        ref_y, ref_sr = librosa.load(ref_path, sr=16000)
                else:
                    ref_y, ref_sr = librosa.load(ref_path, sr=16000)

                if ref_y is None:
                    distances.append(np.nan)
                    continue
                
                ref_feats = self.featurizer(ref_y)

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