from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Abstract base classes — utterance-level
# ---------------------------------------------------------------------------

class LookupEvaluator(ABC):
    """Evaluator that maps utterance/speaker IDs to pre-computed scores.
    Needs only the utterance ID — no audio, transcription, or reference."""

    @abstractmethod
    def score(self, utterance_id: str) -> Optional[float]:
        pass


class ReferenceFreeEvaluator(ABC):
    """Utterance-level evaluator that needs only audio + segment bounds.
    No transcription, no reference audio, no language."""

    @abstractmethod
    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        pass

    @abstractmethod
    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        """Compute score from a pre-loaded audio array.
        Used by TrimmedReferenceFreeEvaluator to inject trimmed audio."""
        pass


class ReferenceTxtEvaluator(ABC):
    """Utterance-level evaluator that needs transcription + language.
    Used for ASR-based metrics and FA-trimming wrappers."""

    @abstractmethod
    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        pass


class ReferenceAudioEvaluator(ABC):
    """Utterance-level evaluator that needs reference audio files.
    No transcription or language required."""

    @abstractmethod
    def score(
        self,
        utterance_id: str,
        audio_path: str,
        reference_audios: List[Tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        pass


class ReferenceTxtAndAudioEvaluator(ABC):
    """Utterance-level evaluator that needs both transcription (for FA trimming)
    AND reference audio files (for distance computation).
    Used for TrimmedNADEvaluator."""

    @abstractmethod
    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        reference_audios: List[Tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        pass


# ---------------------------------------------------------------------------
# Abstract base classes — speaker-level
# ---------------------------------------------------------------------------

def load_audios(
    audio_files: List[Tuple[str, float, float]],
) -> List[Tuple[np.ndarray, int]]:
    """Load a list of (path, start, end) tuples into (ndarray, fs) pairs with librosa.

    Used by script-level dispatch before calling _score_audio_list() on plain
    (non-trimmed) speaker evaluators.
    """
    audios = []
    for audio_path, start_time, end_time in audio_files:
        duration = end_time - start_time if end_time != -1.0 else None
        try:
            audio, fs = librosa.load(
                audio_path, sr=16000, offset=start_time, duration=duration
            )
            if audio is not None and len(audio) > 0:
                audios.append((audio, fs))
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
    return audios

class ReferenceFreeSpeakerEvaluator(ABC):
    """Speaker-level evaluator that needs only audio files + segment bounds.
    No transcription, no language.

    Callers load audio with load_audios() and pass the result to _score_audio_list().
    The trimmed wrapper (TrimmedReferenceFreeSpeakerEvaluator) does the same after
    FA-trimming each utterance.
    """

    @abstractmethod
    def _score_audio_list(
        self, audios: List[Tuple[np.ndarray, int]]
    ) -> Optional[float]:
        """Compute score from a list of pre-loaded (audio, fs) tuples."""
        pass


class LanguageAwareSpeakerEvaluator(ABC):
    """Speaker-level evaluator that needs audio + language.
    Language is required for acoustic model parameters (e.g. vowel formant tables),
    not only for FA trimming.

    Callers load audio with load_audios() and pass the result to _score_audio_list().
    The trimmed wrapper (TrimmedLanguageAwareSpeakerEvaluator) does the same after
    FA-trimming each utterance.
    """

    @abstractmethod
    def _score_audio_list(
        self, audios: List[Tuple[np.ndarray, int]], language: str
    ) -> Optional[float]:
        """Compute score from pre-loaded audio list."""
        pass


# ---------------------------------------------------------------------------
# Trimmer wrappers (decorator pattern)
#
# The FATrimmer requires transcription + language for forced alignment.
# Rather than leaking this requirement into reference-free evaluators,
# these wrappers encapsulate the trimming concern:
#
#   score(…, transcription, language, …)
#     → trimmer.trim(…) → ndarray          [or librosa fallback]
#     → inner._score_audio(audio, fs)      [inner sees only audio]
# ---------------------------------------------------------------------------

class TrimmedReferenceFreeEvaluator(ReferenceTxtEvaluator):
    """Wraps a ReferenceFreeEvaluator with FA trimming.

    The inner evaluator stays reference-free — it never sees transcription or
    language. This wrapper is a ReferenceTxtEvaluator because the trimmer
    needs transcription + language to perform forced alignment.

    Delegation flow:
      1. Receive (audio_path, transcription, language, start_time, end_time)
      2. If no explicit segment: call trimmer.trim() → trimmed ndarray
      3. Fallback to librosa.load() if trim fails or segment is specified
      4. Call inner._score_audio(audio, fs)  ← inner knows nothing about text
    """

    def __init__(self, inner: ReferenceFreeEvaluator, trimmer):
        self.inner = inner
        self.trimmer = trimmer

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        use_segment = start_time != 0.0 or end_time != -1.0
        audio, fs = None, None

        if not use_segment:
            result = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
            if result is not None:
                audio, fs = result

        if audio is None:
            duration = end_time - start_time if end_time != -1.0 else None
            try:
                audio, fs = librosa.load(
                    audio_path, sr=16000, mono=True, offset=start_time, duration=duration
                )
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                return None

        if audio is None or len(audio) == 0:
            return None

        return self.inner._score_audio(audio, fs)


class TrimmedReferenceFreeSpeakerEvaluator:
    """Wraps a ReferenceFreeSpeakerEvaluator with FA trimming.

    Trims each utterance in the speaker's audio list, then delegates
    to inner._score_audio_list() with the trimmed audio arrays.
    """

    def __init__(self, inner: ReferenceFreeSpeakerEvaluator, trimmer):
        self.inner = inner
        self.trimmer = trimmer

    def score(
        self,
        audio_files: List[Tuple[str, float, float]],
        transcriptions: List[str],
        language: str,
    ) -> Optional[float]:
        audios = []
        for (audio_path, start_time, end_time), transcription in zip(audio_files, transcriptions):
            use_segment = start_time != 0.0 or end_time != -1.0
            audio, fs = None, None

            if not use_segment:
                result = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if result is not None:
                    audio, fs = result

            if audio is None:
                duration = end_time - start_time if end_time != -1.0 else None
                try:
                    audio, fs = librosa.load(
                        audio_path, sr=16000, offset=start_time, duration=duration
                    )
                except Exception as e:
                    print(f"Error loading audio {audio_path}: {e}")
                    continue

            if audio is not None and len(audio) > 0:
                audios.append((audio, fs))

        if not audios:
            return None
        return self.inner._score_audio_list(audios)


class TrimmedLanguageAwareSpeakerEvaluator:
    """Wraps a LanguageAwareSpeakerEvaluator with FA trimming.

    Same delegation as TrimmedReferenceFreeSpeakerEvaluator but passes
    language through to inner._score_audio_list() since the inner evaluator
    uses language for its own computation (e.g. VSA vowel formant tables).
    """

    def __init__(self, inner: LanguageAwareSpeakerEvaluator, trimmer):
        self.inner = inner
        self.trimmer = trimmer

    def score(
        self,
        audio_files: List[Tuple[str, float, float]],
        transcriptions: List[str],
        language: str,
    ) -> Optional[float]:
        audios = []
        for (audio_path, start_time, end_time), transcription in zip(audio_files, transcriptions):
            use_segment = start_time != 0.0 or end_time != -1.0
            audio, fs = None, None

            if not use_segment:
                result = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if result is not None:
                    audio, fs = result

            if audio is None:
                duration = end_time - start_time if end_time != -1.0 else None
                try:
                    audio, fs = librosa.load(
                        audio_path, sr=16000, offset=start_time, duration=duration
                    )
                except Exception as e:
                    print(f"Error loading audio {audio_path}: {e}")
                    continue

            if audio is not None and len(audio) > 0:
                audios.append((audio, fs))

        if not audios:
            return None
        return self.inner._score_audio_list(audios, language)


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# These keep non-refactored evaluators (ArticulatoryPrecision, WadaSNR, etc.)
# importable while they are awaiting their own refactor.
# ---------------------------------------------------------------------------

class Evaluator:
    """Deprecated. Kept for backward compatibility. Use the typed ABCs instead."""
    pass


class SpeakerEvaluator:
    """Deprecated. Kept for backward compatibility. Use the typed ABCs instead."""
    pass


# ---------------------------------------------------------------------------
# Lookup evaluators
# ---------------------------------------------------------------------------

class Utt2ScoreEvaluator(LookupEvaluator):
    """Maps utterance IDs to pre-computed scores."""

    def __init__(self, scores: Dict[str, float]):
        self.scores = scores

    def score(self, utterance_id: str) -> Optional[float]:
        return self.scores.get(utterance_id)


class Spk2ScoreEvaluator(LookupEvaluator):
    """Maps utterance IDs → speaker IDs → pre-computed speaker scores."""

    def __init__(self, spk2score: Dict[str, float], utt2spk: Dict[str, str]):
        self.spk2score = spk2score
        self.utt2spk = utt2spk

    def score(self, utterance_id: str) -> Optional[float]:
        speaker_id = self.utt2spk.get(utterance_id)
        if speaker_id:
            return self.spk2score.get(speaker_id)
        return None

