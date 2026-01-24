from typing import Optional, List

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
import re
import numpy as np
import librosa

from pathbench.reference_evaluator import ReferenceEvaluator, STOI
from pathbench.string_clean import clean_text


class ForcedAlignmentPESTOIEvaluator(ReferenceEvaluator):
    """An evaluator that uses P-ESTOI to compute a score after trimming silence using forced alignment."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft", **kwargs):
        super().__init__(**kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.cache = {}
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

    def _trim_audio(self, audio_path: str, transcription: str, language: str, start_time: float = 0.0, end_time: float = -1.0) -> Optional[np.ndarray]:
        """
        Trims silence from the beginning and end of an audio file using forced alignment.
        """
        cache_key = (audio_path, transcription, language, start_time, end_time)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            duration = end_time - start_time if end_time != -1 else None
            speech, sample_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration, dtype=np.float64)
            print(speech, sample_rate)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Process audio
        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_values).logits

        # 1. Phonemize the ground truth transcription.
        separator = Separator(phone=" ", word="|")
        phonemized_reference = phonemize(
            clean_text(transcription),
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )
        phonemized_reference = re.sub(r"\s+", " ", phonemized_reference.replace("|", " ")).strip()
        if not phonemized_reference:
            print(f"Warning: Could not phonemize reference transcription for {audio_path}.")
            return None

        # 2. Get the mapping from phonemes to model vocab indices.
        vocab = self.processor.tokenizer.get_vocab()
        target_phonemes = phonemized_reference.split()

        # remove j
        target_phonemes = [p.replace("ʲ", "") for p in target_phonemes]
        # remove dz
        target_phonemes = [p.replace("dz", "z") for p in target_phonemes]

        for p in target_phonemes:
            if p not in vocab:
                # Remove
                print(f"Warning: Phoneme {p} not in model vocabulary for {audio_path}.")
                target_phonemes.remove(p)

        try:
            target_ids = [vocab[p] for p in target_phonemes]
        except KeyError as e:
            print(f"Error: Phoneme {e} not in model vocabulary.")
            return None

        # 3. Forced alignment
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = torch.exp(emissions)
        emissions = emissions.cpu()
        targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)

        try:
            aligned_path, scores = torchaudio.functional.forced_align(
                emissions, targets, blank=vocab.get(self.processor.tokenizer.pad_token, 0)
            )
        except Exception as e:
            print(f"Forced alignment failed for {audio_path}: {e}")
            return None

        # 4. Get start and end frames of speech

        
        # The frames are in terms of the model's output, which is every 20ms.
        # The model's output has a lower frame rate than the audio.
        # The ratio is sample_rate / (model_output_rate)
        # The model output rate is roughly 50 Hz (1000ms / 20ms).
        # So the ratio is 16000 / 50 = 320.
        
        # Find the index of first non-zero in aligned path
        for i, x in enumerate(aligned_path[0]):
            if x != 0:
                start_idx = i
                break
        
        # Find the index of last non-zero in aligned path
        for i, x in enumerate(reversed(aligned_path[0])):
            if x != 0:
                end_idx = len(aligned_path[0]) - i
                break

  
        ratio = speech.shape[0] / emissions.shape[1]
        #print("ratio", ratio)
        #print("start_idx", start_idx)
        #print("end_idx", end_idx)   
        start_frame = int(start_idx * ratio)
        end_frame = int((end_idx+1) * ratio)

        trimmed_audio = speech[start_frame:end_frame]
        print("trimmed_audio_length", trimmed_audio.shape)
        self.cache[cache_key] = trimmed_audio
        return trimmed_audio


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
        trimmed_audio = self._trim_audio(audio_path, transcription, language, start_time, end_time)

        # Check if test_audio is full silence
        if trimmed_audio is None or np.all(trimmed_audio == 0):
            print(f"Warning: Test audio {audio_path} is silent or could not be trimmed. Returning P-ESTOI score of 0.0.")
            return 0.0
    
        reference_audios_data = []
        if reference_audios:
            for ref_path, ref_start, ref_end in reference_audios:
                ref_audio = self._trim_audio(ref_path, transcription, language, ref_start, ref_end)
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
