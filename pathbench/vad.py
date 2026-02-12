from typing import Optional, Tuple

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
import re
import numpy as np
import librosa

from pathbench.string_clean import clean_text


class FATrimmer:
    """A class to trim silence from audio using forced alignment."""

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.cache = {}
        print(f"Phonetic model '{model_id}' loaded on {self.device}.")

    def trim(self, audio_path: str, transcription: str, language: str, start_time: float = 0.0, end_time: float = -1.0) -> Optional[Tuple[np.ndarray, int]]:
        """
        Trims silence from the beginning and end of an audio file using forced alignment.
        Returns a tuple of (trimmed_audio_array, sample_rate).
        """
        cache_key = (audio_path, transcription, language, start_time, end_time)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            duration = end_time - start_time if end_time != -1 else None
            speech, sample_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration, dtype=np.float64)
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
            print(f"Unphonemized transcription: '{transcription}'")
            print(f"Updated transcription after cleaning: '{clean_text(transcription)}'")
            return None

        # 2. Get the mapping from phonemes to model vocab indices.
        vocab = self.processor.tokenizer.get_vocab()
        target_phonemes = phonemized_reference.split()

        # remove j
        target_phonemes = [p.replace("ʲ", "") for p in target_phonemes]
        # remove dz
        target_phonemes = [p.replace("dz", "z") for p in target_phonemes]

        original_phonemes = len(target_phonemes)
        target_phonemes = [p for p in target_phonemes if p in vocab]
        if len(target_phonemes) < original_phonemes:
            print(f"Warning: Some phonemes not in model vocabulary for {audio_path}.")

        if not target_phonemes:
            print(f"Warning: No phonemes in model vocabulary for {audio_path}.")
            return None

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
        try:
            start_idx = -1
            for i, x in enumerate(aligned_path[0]):
                if x != 0:
                    start_idx = i
                    break
            
            if start_idx == -1:
                print(f"Warning: Could not find start of speech in {audio_path}. Returning full audio.")
                self.cache[cache_key] = (speech, sample_rate)
                return speech, sample_rate

            end_idx = -1
            for i, x in enumerate(reversed(aligned_path[0])):
                if x != 0:
                    end_idx = len(aligned_path[0]) - i
                    break
            
            if end_idx == -1:
                print(f"Warning: Could not find end of speech in {audio_path}. Returning full audio.")
                self.cache[cache_key] = (speech, sample_rate)
                return speech, sample_rate

            ratio = speech.shape[0] / emissions.shape[1]
            start_frame = int(start_idx * ratio)
            end_frame = int((end_idx + 1) * ratio)

            trimmed_audio = speech[start_frame:end_frame]
            self.cache[cache_key] = (trimmed_audio, sample_rate)
            return trimmed_audio, sample_rate
        except Exception as e:
            print(f"Error during trimming of {audio_path}: {e}")
            return None