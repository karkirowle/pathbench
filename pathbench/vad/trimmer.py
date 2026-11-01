from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torchaudio
import re
import numpy as np
import librosa

from pathbench.core.string_clean import clean_text, cached_phonemize


class FATrimmer:
    """A class to trim silence from audio using forced alignment."""

    MAX_CACHE_SIZE = 10000

    def __init__(self, model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft", use_exp: bool = False):
        from pathbench.core.model_registry import get_ctc_model
        self.processor, self.model, self.device = get_ctc_model(model_id)
        self.use_exp = use_exp
        self.cache = OrderedDict()

    def _cache_put(self, key, value):
        self.cache[key] = value
        if len(self.cache) > self.MAX_CACHE_SIZE:
            self.cache.popitem(last=False)

    def _forced_align(self, audio_path: str, transcription: str, language: str,
                      start_time: float = 0.0, end_time: float = -1.0):
        """Run phoneme forced alignment. Returns
        ``(speech, sample_rate, aligned_path, target_phonemes, blank_id)`` or
        ``None`` on any failure. ``speech`` is the loaded (offset/duration
        applied) 16 kHz waveform; ``aligned_path`` is the per-emission-frame
        token-id path from ``torchaudio.functional.forced_align``."""
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
        phonemized_reference = cached_phonemize(clean_text(transcription), language)
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
        if self.use_exp:
            emissions = torch.exp(emissions)
        emissions = emissions.cpu()
        targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)
        blank_id = vocab.get(self.processor.tokenizer.pad_token, 0)

        try:
            aligned_path, scores = torchaudio.functional.forced_align(
                emissions, targets, blank=blank_id
            )
        except Exception as e:
            print(f"Forced alignment failed for {audio_path}: {e}")
            return None

        return speech, sample_rate, aligned_path, target_phonemes, blank_id

    def align(self, audio_path: str, transcription: str, language: str,
              start_time: float = 0.0, end_time: float = -1.0):
        """Forced-align ``transcription`` to the audio and return
        ``(speech, sample_rate, segments)`` where ``segments`` is a list of
        ``(phone, start_sample, end_sample)`` in waveform-sample units, one per
        target phone (in order). Returns ``None`` on any alignment failure.
        Reuses the same model forward as :meth:`trim`."""
        cache_key = ("align", audio_path, transcription, language, start_time, end_time)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        fa = self._forced_align(audio_path, transcription, language, start_time, end_time)
        if fa is None:
            return None
        speech, sample_rate, aligned_path, target_phonemes, blank_id = fa

        path = aligned_path[0].tolist()
        n_frames = len(path)
        ratio = speech.shape[0] / n_frames

        # Walk the path, advancing a pointer through the target phones. A new
        # phone starts at each non-blank run that is not a continuation of the
        # current target token; this recovers one span per target position even
        # when adjacent phones repeat the same id (blank-separated by the
        # aligner). Span bounds are emission-frame indices -> samples via ratio.
        segments = []
        ptr = -1
        cur_start = None
        prev_tok = blank_id
        for i, tok in enumerate(path):
            if tok == blank_id:
                if cur_start is not None:
                    segments.append((ptr, cur_start, i))
                    cur_start = None
                prev_tok = blank_id
                continue
            if cur_start is None or tok != prev_tok:
                # new phone span begins
                if cur_start is not None:
                    segments.append((ptr, cur_start, i))
                ptr += 1
                cur_start = i
            prev_tok = tok
        if cur_start is not None:
            segments.append((ptr, cur_start, n_frames))

        out = []
        for pos, s, e in segments:
            if 0 <= pos < len(target_phonemes):
                out.append((target_phonemes[pos], int(s * ratio), int(e * ratio)))
        if not out:
            return None

        result = (speech, sample_rate, out)
        self._cache_put(cache_key, result)
        return result

    def trim(self, audio_path: str, transcription: str, language: str, start_time: float = 0.0, end_time: float = -1.0) -> Optional[Tuple[np.ndarray, int]]:
        """
        Trims silence from the beginning and end of an audio file using forced alignment.
        Returns a tuple of (trimmed_audio_array, sample_rate).
        """
        cache_key = (audio_path, transcription, language, start_time, end_time)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        fa = self._forced_align(audio_path, transcription, language, start_time, end_time)
        if fa is None:
            return None
        speech, sample_rate, aligned_path, target_phonemes, blank_id = fa
        emissions_len = aligned_path.shape[1]

        # 4. Get start and end frames of speech
        try:
            start_idx = -1
            for i, x in enumerate(aligned_path[0]):
                if x != 0:
                    start_idx = i
                    break
            
            if start_idx == -1:
                print(f"Warning: Could not find start of speech in {audio_path}. Returning full audio.")
                self._cache_put(cache_key, (speech, sample_rate))
                return speech, sample_rate

            end_idx = -1
            for i, x in enumerate(reversed(aligned_path[0])):
                if x != 0:
                    end_idx = len(aligned_path[0]) - i
                    break
            
            if end_idx == -1:
                print(f"Warning: Could not find end of speech in {audio_path}. Returning full audio.")
                self._cache_put(cache_key, (speech, sample_rate))
                return speech, sample_rate

            ratio = speech.shape[0] / emissions_len
            start_frame = int(start_idx * ratio)
            end_frame = int((end_idx + 1) * ratio)

            trimmed_audio = speech[start_frame:end_frame]
            self._cache_put(cache_key, (trimmed_audio, sample_rate))
            return trimmed_audio, sample_rate
        except Exception as e:
            print(f"Error during trimming of {audio_path}: {e}")
            return None