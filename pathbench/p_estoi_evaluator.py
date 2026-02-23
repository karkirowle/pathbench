from typing import Optional, List

import numpy as np
import librosa

from pathbench.reference_evaluator import ReferenceEvaluator, STOI
from pathbench.string_clean import clean_text
from pathbench.vad import FATrimmer

class ForcedAlignmentPESTOIEvaluator(ReferenceEvaluator):
    """An evaluator that uses P-ESTOI to compute a score after trimming silence using forced alignment."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft", **kwargs):
        super().__init__(**kwargs)
        self.trimmer = FATrimmer(model_id)

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
        Computes the P-ESTOI score after trimming silence.
        """
        use_segments = start_time != 0.0 or end_time != -1.0

        trimmed_audio = None
        if use_segments:
            duration = end_time - start_time if end_time != -1 else None
            try:
                trimmed_audio, _ = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration, dtype=np.float64)
            except Exception as e:
                print(f"Error reading audio file {audio_path}: {e}")
                trimmed_audio = None
        else:
            trimmed_data = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
            if trimmed_data:
                trimmed_audio, _ = trimmed_data


        # Check if test_audio is full silence
        if trimmed_audio is None or np.all(trimmed_audio == 0):
            print(f"Warning: Test audio {audio_path} is silent or could not be trimmed. Returning P-ESTOI score of 0.0.")
            return 0.0
    
        reference_audios_data = []
        if reference_audios:
            for ref_path, ref_start, ref_end in reference_audios:
                ref_use_segments = ref_start != 0.0 or ref_end != -1.0
                ref_audio = None
                if ref_use_segments:
                    duration = ref_end - ref_start if ref_end != -1 else None
                    try:
                        ref_audio, _ = librosa.load(ref_path, sr=16000, offset=ref_start, duration=duration, dtype=np.float64)
                    except Exception as e:
                        print(f"Error reading audio file {ref_path}: {e}")
                        ref_audio = None
                else:
                    trimmed_ref_data = self.trimmer.trim(ref_path, transcription, language, ref_start, ref_end)
                    if trimmed_ref_data:
                        ref_audio, _ = trimmed_ref_data
                
                if ref_audio is not None:
                    reference_audios_data.append(ref_audio)

        if not reference_audios_data:
            print(f"Warning: No valid reference audios found for {utterance_id}. Cannot compute P-ESTOI.")
            return None

        stoi_object = STOI(
            normalization_method='RMS',
            centroid_ind=0,
            frame_deletion=True,
            reference_words=reference_audios_data,
            test_words=[trimmed_audio],
            **self.stoi_kwargs
        )
        return stoi_object.estoi_val[0]
