from abc import ABC, abstractmethod
from typing import Dict, Optional

import jiwer
import soundfile as sf
import torch
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
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
        **kwargs,
    ) -> Optional[float]:
        """
        Returns the pre-computed score for a given utterance ID.
        The other arguments are ignored.
        """
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
        **kwargs,
    ) -> Optional[float]:
        """
        Returns the pre-computed score for a given utterance ID based on its speaker.
        The other arguments are ignored.
        """
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
        **kwargs,
    ) -> Optional[float]:
        """
        Performs ASR on the audio file and returns 1 - WER as the score.
        """
        try:
            speech, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if sample_rate != 16000:
            # TODO: Add resampling for non-16kHz audio.
            print(
                f"Warning: audio for {utterance_id} has sample rate {sample_rate}Hz. "
                "ASR model expects 16kHz. Skipping."
            )
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

        # Calculate WER
        # Normalize transcriptions for a fairer comparison
        transformation = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.SentencesToListOfWords(),
            ]
        )

        wer = jiwer.wer(
            transcription,
            predicted_transcription,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )

        # Return 1 - WER so that a higher score is better
        return 1 - wer


class DurationEvaluator(Evaluator):
    """An evaluator that scores based on the duration of the audio."""

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Returns the duration of the audio file in seconds.
        """
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            print(f"Error reading audio file info {audio_path}: {e}")
            return None
