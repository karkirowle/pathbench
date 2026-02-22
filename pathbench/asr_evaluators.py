from typing import Optional
import re
import os

import jiwer
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
from pyctcdecode import build_ctcdecoder

import numpy as np
from pathbench.evaluator import ReferenceTxtEvaluator, ReferenceFreeEvaluator
from pathbench.string_clean import clean_text


class ASREvaluator(ReferenceTxtEvaluator):
    """Computes WER using an ASR model."""

    def __init__(self, model_id: str):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"ASR model '{model_id}' loaded on {self.device}.")

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(
                audio_path, sr=16000, mono=True, offset=start_time, duration=duration
            )
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping {audio_path} — too short ({len(speech)} samples).")
            return None

        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")
        print(f"Predicted: {predicted_transcription}")

        cleaned_reference = clean_text(transcription)
        cleaned_prediction = clean_text(predicted_transcription)

        print("Cleaned Reference:", cleaned_reference)
        print("Cleaned Predicted:", cleaned_prediction)

        return jiwer.wer(cleaned_reference, cleaned_prediction)


class PEREvaluator(ReferenceTxtEvaluator):
    """Computes PER using a language-specific ASR model."""

    def __init__(self, language: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ids = {
            "en":    "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "en-us": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "es":    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            "nl":    "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            "it":    "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        }
        if language not in model_ids:
            raise ValueError(f"Language '{language}' is not supported for PEREvaluator.")

        model_id = model_ids[language]
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.to(self.device)
        print(f"ASR model '{model_id}' for language '{language}' loaded on {self.device}.")
        self.language = language

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        if language != self.language:
            print(
                f"Warning: PEREvaluator initialized for '{self.language}' "
                f"but received '{language}'. Skipping."
            )
            return None

        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(
                audio_path, sr=16000, mono=True, offset=start_time, duration=duration
            )
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping {audio_path} — too short ({len(speech)} samples).")
            return None

        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt", padding="longest"
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")

        espeak_language_map = {
            "en": "en-us", "en-us": "en-us", "es": "es", "nl": "nl", "it": "it"
        }
        espeak_lang = espeak_language_map.get(language, language)
        separator = Separator(phone=" ", word="|")

        phonemized_reference = phonemize(
            clean_text(transcription), language=espeak_lang, backend="espeak",
            strip=True, preserve_punctuation=False, separator=separator
        )
        phonemized_prediction = phonemize(
            clean_text(predicted_transcription), language=espeak_lang, backend="espeak",
            strip=True, preserve_punctuation=False, separator=separator
        )

        print(f"Phonemized Reference: {phonemized_reference}")
        print(f"Phonemized Predicted: {phonemized_prediction}")

        cleaned_reference = re.sub(r"\s+", " ", phonemized_reference.replace("|", " ")).strip()
        cleaned_prediction = re.sub(r"\s+", " ", phonemized_prediction.replace("|", " ")).strip()

        print("Cleaned Phonemized Reference:", cleaned_reference)
        print("Cleaned Phonemized Predicted:", cleaned_prediction)

        return jiwer.wer(cleaned_reference, cleaned_prediction)


class DirectPEREvaluator(ReferenceTxtEvaluator):
    """Computes PER using the espeak-cv-ft model directly."""

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(
                audio_path, sr=16000, mono=True, offset=start_time, duration=duration
            )
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping {audio_path} — too short ({len(speech)} samples).")
            return None

        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt", padding="longest"
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")

        separator = Separator(phone=" ", word="|")
        phonemized_reference = phonemize(
            clean_text(transcription), language=language, backend="espeak",
            strip=True, preserve_punctuation=False, separator=separator
        )

        print(f"Phonemized Reference: {phonemized_reference}")
        print(f"Phonemized Predicted: {predicted_transcription}")

        cleaned_reference = re.sub(
            r"\s+", " ", phonemized_reference.replace("|", " ")
        ).strip()
        cleaned_prediction = re.sub(
            r"\s+", " ", predicted_transcription.replace("|", " ")
        ).strip()

        print("Cleaned Phonemized Reference:", cleaned_reference)
        print("Cleaned Phonemized Predicted:", cleaned_prediction)

        return jiwer.wer(cleaned_reference, cleaned_prediction)


class DoubleASREvaluator(ReferenceFreeEvaluator):
    """Computes PER between greedy and LM-based CTC decoding."""

    def __init__(self, language: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ids = {
            "en":    "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "en-us": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "es":    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            "nl":    "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            "it":    "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        }
        if language not in model_ids:
            raise ValueError(f"Language '{language}' is not supported for DoubleASREvaluator.")

        model_id = model_ids[language]
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.to(self.device)
        print(f"ASR model '{model_id}' for language '{language}' loaded on {self.device}.")
        self.language = language

        lms_dir = 'lms'
        lm_paths = {
            "en": os.path.join(lms_dir, "wiki_en_token.arpa"),
            "nl": os.path.join(lms_dir, "wiki_nl_token.arpa"),
            "es": os.path.join(lms_dir, "wiki_es_token.arpa.bin"),
            "it": os.path.join(lms_dir, "wiki_it_token.arpa.bin"),
        }

        lm_lang = language.split('-')[0]
        if lm_lang not in lm_paths:
            lm_lang = 'en'

        lm_path = lm_paths.get(lm_lang)
        if lm_path and lm_path.endswith('.arpa'):
            bin_path = lm_path + '.bin'
            if os.path.exists(bin_path):
                lm_path = bin_path

        self.decoder = None
        if lm_path and os.path.exists(lm_path):
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
            labels = list(sorted_vocab_dict.keys())
            self.decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path)
            print(f"CTC decoder for '{language}' with LM '{lm_path}' built.")
        else:
            print(f"Warning: Language model for '{language}' not found at '{lm_path}'.")

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        if not self.decoder:
            print(f"Error: No decoder available for language '{self.language}'.")
            return None

        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(
                audio_path, sr=16000, mono=True, offset=start_time, duration=duration
            )
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping {audio_path} — too short ({len(speech)} samples).")
            return None

        return self._score_audio(speech, sample_rate)

    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        input_values = self.processor(
            audio, sampling_rate=fs, return_tensors="pt"
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        greedy_transcription = self.processor.batch_decode(predicted_ids)[0]
        lm_transcription = self.decoder.decode(logits.cpu().numpy()[0])

        print(f"Greedy: {greedy_transcription}")
        print(f"With LM: {lm_transcription}")

        espeak_language_map = {
            "en": "en-us", "en-us": "en-us", "es": "es", "nl": "nl", "it": "it"
        }
        espeak_lang = espeak_language_map.get(self.language, self.language)
        separator = Separator(phone=" ", word="|")

        phonemized_greedy = phonemize(
            clean_text(greedy_transcription), language=espeak_lang, backend="espeak",
            strip=True, preserve_punctuation=False, separator=separator
        )
        phonemized_lm = phonemize(
            clean_text(lm_transcription), language=espeak_lang, backend="espeak",
            strip=True, preserve_punctuation=False, separator=separator
        )

        print(f"Phonemized Greedy: {phonemized_greedy}")
        print(f"Phonemized With LM: {phonemized_lm}")

        cleaned_greedy = re.sub(r"\s+", " ", phonemized_greedy.replace("|", " ")).strip()
        cleaned_lm = re.sub(r"\s+", " ", phonemized_lm.replace("|", " ")).strip()

        print("Cleaned Phonemized Greedy:", cleaned_greedy)
        print("Cleaned Phonemized With LM:", cleaned_lm)

        return jiwer.wer(cleaned_greedy, cleaned_lm)
