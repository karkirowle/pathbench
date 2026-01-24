from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import re

import jiwer
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator

from pathbench.string_clean import clean_text

class Evaluator(ABC):
    """Abstract base class for evaluators."""

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
        Scores a given utterance.

        Args:
            utterance_id: The ID of the utterance.
            audio_path: The path to the audio file.
            transcription: The transcription of the utterance.
            language: The language of the utterance.
            **kwargs: Additional information that might be needed by the evaluator.

        Returns:
            A score, or None if a score cannot be computed.
        """
        pass


class SpeakerEvaluator(ABC):
    """Abstract base class for speaker-level evaluators."""

    @abstractmethod
    def score(
        self,
        audio_files: List[tuple[str, float, float]],
        transcriptions: List[str],
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Scores a given speaker based on all their utterances.

        Args:
            audio_paths: A list of paths to the audio files for the speaker.
            transcriptions: A list of transcriptions for the speaker.
            language: The language of the utterances.
            **kwargs: Additional information that might be needed by the evaluator.

        Returns:
            A score, or None if a score cannot be computed.
        """
        pass


class Utt2ScoreEvaluator(Evaluator):
    """An evaluator that uses a pre-computed utt2score mapping."""

    def __init__(self, scores: Dict[str, float]):
        self.scores = scores

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
        return self.scores.get(utterance_id)


class Spk2ScoreEvaluator(Evaluator):
    """An evaluator that uses a pre-computed spk2score mapping."""

    def __init__(self, spk2score: Dict[str, float], utt2spk: Dict[str, str]):
        self.spk2score = spk2score
        self.utt2spk = utt2spk

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
        speaker_id = self.utt2spk.get(utterance_id)
        if speaker_id:
            return self.spk2score.get(speaker_id)
        return None


class ASREvaluator(Evaluator):
    """An evaluator that uses an ASR model to compute a score based on WER."""

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
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Performs ASR on the audio file and returns 1 - WER as the score.
        """
        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(audio_path, sr=16000, mono=True, offset=start_time, duration=duration)
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

        # Get ASR prediction
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")
        print(f"Predicted: {predicted_transcription}")

        # Clean transcriptions
        cleaned_reference = clean_text(transcription)
        cleaned_prediction = clean_text(predicted_transcription)

        print("Cleaned Reference:", cleaned_reference)
        print("Cleaned Predicted:", cleaned_prediction)
        # Calculate WER
        wer = jiwer.wer(cleaned_reference, cleaned_prediction)

        return wer


class PEREvaluator(Evaluator):
    """An evaluator that uses an ASR model to compute a score based on PER."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ids = {
            "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "en-us": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        }
        self.processors = {}
        self.models = {}
        for lang, model_id in model_ids.items():
            self.processors[lang] = Wav2Vec2Processor.from_pretrained(model_id)
            self.models[lang] = Wav2Vec2ForCTC.from_pretrained(model_id)
            self.models[lang].to(self.device)
            print(f"ASR model '{model_id}' for language '{lang}' loaded on {self.device}.")


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
        Performs ASR on the audio file and returns 1 - PER as the score.
        """
        if language not in self.models:
            print(f"Error: Language '{language}' is not supported for PEREvaluator. Supported languages are: {list(self.models.keys())}")
            return None

        processor = self.processors[language]
        model = self.models[language]

        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(audio_path, sr=16000, mono=True, offset=start_time, duration=duration)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Process audio
        input_values = processor(
            speech, sampling_rate=sample_rate, return_tensors="pt", padding="longest"
        ).input_values
        input_values = input_values.to(self.device)

        # Get ASR prediction
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")
        print(f"Predicted: {predicted_transcription}")

        separator = Separator(phone = " ", word = "|")
        
        espeak_language_map = {
            "en": "en-us",
            "en-us": "en-us",
            "es": "es",
            "nl": "nl"
        }
        espeak_lang = espeak_language_map.get(language, language)

        # Phonemize transcriptions
        phonemized_reference = phonemize(
            clean_text(transcription),
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )
        phonemized_prediction = phonemize(
            clean_text(predicted_transcription),
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )
        
        print(f"Phonemized Reference: {phonemized_reference}")
        print(f"Phonemized Predicted: {phonemized_prediction}")

        cleaned_reference = phonemized_reference.replace("|", " ")
        cleaned_prediction = phonemized_prediction.replace("|", " ")

        cleaned_reference = re.sub(r"\s+", " ", cleaned_reference).strip()
        cleaned_prediction = re.sub(r"\s+", " ", cleaned_prediction).strip()
        print("Cleaned Phonemized Reference:", cleaned_reference)
        print("Cleaned Phonemized Predicted:", cleaned_prediction)


        # Calculate PER
        per = jiwer.wer(cleaned_reference, cleaned_prediction)

        # Return 1 - PER so that a higher score is better
        return per

class DirectPEREvaluator(Evaluator):
    """An evaluator that uses an ASR model to compute a score based on PER."""

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


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
        Performs ASR on the audio file and returns 1 - PER as the score.
        """
        try:
            duration = end_time - start_time if end_time >= 0 else None
            speech, sample_rate = librosa.load(audio_path, sr=16000, mono=True, offset=start_time, duration=duration)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if len(speech) < 400:
            print(f"Warning: Skipping audio file {audio_path} because it is too short ({len(speech)} samples).")
            return None

        # Process audio
        input_values = self.processor(
            speech, sampling_rate=sample_rate, return_tensors="pt", padding="longest"
        ).input_values
        input_values = input_values.to(self.device)

        # Get ASR prediction
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"Reference: {transcription}")

        separator = Separator(phone = " ", word = "|")
        # Phonemize transcriptions
        phonemized_reference = phonemize(
            clean_text(transcription),
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=separator
        )

        phonemized_prediction = predicted_transcription
        print(f"Phonemized Reference: {phonemized_reference}")
        print(f"Phonemized Predicted: {phonemized_prediction}")

        cleaned_reference = phonemized_reference.replace("|", " ")
        cleaned_prediction = phonemized_prediction.replace("|", " ")

        cleaned_reference = re.sub(r"\s+", " ", cleaned_reference).strip()
        cleaned_prediction = re.sub(r"\s+", " ", cleaned_prediction).strip()
        print("Cleaned Phonemized Reference:", cleaned_reference)
        print("Cleaned Phonemized Predicted:", cleaned_prediction)


        # Calculate PER
        per = jiwer.wer(cleaned_reference, cleaned_prediction)

        # Return 1 - PER so that a higher score is better
        return per





