"""Reference-audio intelligibility evaluators built on speech-to-EMA features
from :mod:`pathbench.articulatory.runner`.

Metrics that share feature extraction (one BiGRU forward pass per audio file,
cached) and differ in the articulatory trajectory (EMA or tract variables),
optional prosodic augmentation, and whether forced-alignment trimming is
applied:

* :class:`ARTNADEvaluator` -- joint 12-D DTW over EMA, no VAD.
* :class:`TrimmedARTNADEvaluator` -- joint 12-D DTW after forced-alignment
  trimming (analog of :class:`pathbench.evaluators.nad.TrimmedNADEvaluator`,
  which is what the master results table publishes as "NAD").
* ``*TV*`` variants -- the same over the tract-variable trajectory.

They apply per-utterance, per-channel z-score normalization before DTW: the
released ``hprc_*`` inversion checkpoint is "speaker-independent" but its
learned coordinate frame still carries per-speaker biases that would
otherwise dominate the DTW cost over the trajectory shape we actually care
about.
"""
from typing import List, Optional

import librosa
import numpy as np
from dtw import dtw

from pathbench.articulatory.runner import ArticulatoryRunner
from pathbench.core.evaluator import (
    ReferenceAudioEvaluator,
    ReferenceTxtAndAudioEvaluator,
)
from pathbench.vad import FATrimmer


_MIN_FEATURE_LEN = 2  # DTW needs at least 2 frames per sequence.

# Reference pitch for the semitone scale. Its exact value only adds a constant
# offset, which mean-centering removes, so it is irrelevant whenever the
# semitone channel is mean-normalized; pinned to the pyin fmin for clarity.
_F0_REF_HZ = 50.0


def _extract_prosodic(audio: np.ndarray, sr: int = 16000, hop: int = 160,
                      f0_transform: str = "log") -> "tuple[np.ndarray, np.ndarray]":
    """F0 (pyin) and log-RMS at 100 Hz so they align with the BiGRU's
    EMA framerate. NaN gaps in F0 (unvoiced regions) are linearly
    interpolated so DTW sees a continuous trajectory; the voicing decision
    is implicitly captured by the energy contour. ``f0_transform`` selects the
    F0 channel scale: ``"log"`` (natural log Hz) or ``"semitone"``
    (12*log2(f0 / ref), a perceptual unit that keeps pitch range as signal).
    Returns ``(f0_feat, log_rms)`` both shape ``(T,)``."""
    f0, _, _ = librosa.pyin(audio, sr=sr, hop_length=hop, fmin=50.0, fmax=500.0)
    clipped = np.clip(f0, 1e-3, None)
    if f0_transform == "semitone":
        f0_feat = 12.0 * np.log2(clipped / _F0_REF_HZ)
    else:
        f0_feat = np.log(clipped)
    nan_mask = np.isnan(f0_feat)
    if nan_mask.all():
        f0_feat = np.zeros_like(f0_feat)
    elif nan_mask.any():
        valid = np.where(~nan_mask)[0]
        f0_feat = np.interp(np.arange(len(f0_feat)), valid, f0_feat[valid])
    rms = librosa.feature.rms(y=audio, hop_length=hop, frame_length=400)[0]
    log_rms = np.log(np.clip(rms, 1e-9, None))
    n = min(len(f0_feat), len(log_rms))
    return f0_feat[:n].astype(np.float32), log_rms[:n].astype(np.float32)


def _augment_ema_with_prosody(ema: np.ndarray, audio: np.ndarray,
                              f0_transform: str = "log") -> np.ndarray:
    """Concatenate an F0 channel (``f0_transform``: log or semitone) and a
    log-RMS channel onto an existing 12-D EMA array, truncating all three to
    the common length. Returns ``(n, 14)``."""
    f0_feat, log_rms = _extract_prosodic(audio, f0_transform=f0_transform)
    n = min(ema.shape[0], len(f0_feat), len(log_rms))
    return np.concatenate([
        ema[:n],
        f0_feat[:n, None],
        log_rms[:n, None],
    ], axis=1)


def _zscore_per_channel(feats: np.ndarray) -> np.ndarray:
    """Per-channel mean=0, std=1 normalization. Std-floor of 1e-6 avoids
    division by zero on degenerate (flat) channels."""
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (feats - mu) / sd


def _identity_norm(feats: np.ndarray) -> np.ndarray:
    """No normalization. Keeps absolute channel magnitudes, which carry the
    intelligibility signal for tract variables (constriction degree/location)."""
    return feats


def _raw_with_centered_prosody(feats: np.ndarray, n_prosody: int = 2) -> np.ndarray:
    """Articulatory channels left raw; the trailing ``n_prosody`` channels
    (log-F0, log-RMS) are mean-centered per channel. Removes the speaker pitch
    offset (an octave between speakers) while keeping prosody's small natural
    excursion commensurate with the raw tract-variable scale."""
    out = feats.copy()
    out[:, -n_prosody:] = feats[:, -n_prosody:] - feats[:, -n_prosody:].mean(0, keepdims=True)
    return out


def _raw_with_zscored_prosody(feats: np.ndarray, n_prosody: int = 2) -> np.ndarray:
    """Articulatory channels left raw; the trailing ``n_prosody`` channels
    (log-F0, log-RMS) are z-scored per channel."""
    out = feats.copy()
    p = feats[:, -n_prosody:]
    sd = p.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    out[:, -n_prosody:] = (p - p.mean(0, keepdims=True)) / sd
    return out


def _dtw_joint(test_feats: np.ndarray, ref_feats: np.ndarray) -> float:
    """Single multivariate DTW over all 12 EMA channels."""
    return dtw(test_feats, ref_feats, distance_only=True).normalizedDistance


# ----------------------------------------------------------------------------
# Untrimmed variants (no VAD).
# ----------------------------------------------------------------------------


class _ARTBaseEvaluator(ReferenceAudioEvaluator):
    """Shared feature extraction + caching for the no-VAD ART-NAD variants.
    Concrete subclasses pick the trajectory via :meth:`_feats` and a feature
    normalization via :attr:`_norm_fn`.
    """

    _dtw_fn = staticmethod(_dtw_joint)
    _norm_fn = staticmethod(_zscore_per_channel)

    def __init__(self, runner: ArticulatoryRunner):
        self.runner = runner
        # Cache stores already-z-scored features so the normalization runs once
        # per audio file even when an utterance appears as a reference for many
        # other test utterances. Keyed by (path, start, end) — same scheme as
        # NADEvaluator.
        self._feature_cache: dict = {}

    def _feats(self, audio):
        """Tract-variable trajectory for one waveform, from the TV inversion
        checkpoint (requires a runner built with a custom TV checkpoint)."""
        return self.runner.extract_tv(audio)

    def _get_features(self, audio_path, start_time, end_time):
        cache_key = (audio_path, start_time, end_time)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            offset = start_time if start_time != 0.0 else 0
            audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)
            if audio is None or len(audio) == 0:
                result = (None, f"Audio at {audio_path} could not be loaded or is empty.")
                self._feature_cache[cache_key] = result
                return result
            ema = self._feats(audio)  # (T, C)
            if ema.shape[0] < _MIN_FEATURE_LEN:
                result = (None, f"EMA length {ema.shape[0]} < {_MIN_FEATURE_LEN} for {audio_path}.")
                self._feature_cache[cache_key] = result
                return result
            ema = self._norm_fn(ema)
            result = (ema, None)
            self._feature_cache[cache_key] = result
            return result
        except Exception as e:
            result = (None, f"Failed to process {audio_path}: {e}")
            self._feature_cache[cache_key] = result
            return result

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        reference_audios: List[tuple],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        if not reference_audios:
            return None

        test_feats, err = self._get_features(audio_path, start_time, end_time)
        if err:
            print(f"Error: Failed to get EMA for test {utterance_id}: {err}")
            return None

        ref_feats = []
        for ref_path, ref_start, ref_end in reference_audios:
            r_feats, err = self._get_features(ref_path, ref_start, ref_end)
            if err:
                print(f"Warning: Failed to get EMA for ref {ref_path} in group {utterance_id}, skipping ref. {err}")
            else:
                ref_feats.append(r_feats)

        if not ref_feats:
            print(f"Error: No usable reference EMA for {utterance_id}.")
            return None

        distances = []
        for r in ref_feats:
            try:
                distances.append(self._dtw_fn(test_feats, r))
            except Exception as e:
                print(f"Error during DTW for {utterance_id}: {e}")
                distances.append(np.nan)

        return np.nanmean(distances) if distances else None


class ARTNADTVEvaluator(_ARTBaseEvaluator):
    """ART-NAD on tract variables: joint DTW over the per-utterance z-scored
    quasi-TV trajectory from the TV inversion model."""

    _dtw_fn = staticmethod(_dtw_joint)


class ARTNADTVRawEvaluator(ARTNADTVEvaluator):
    """ART-NAD on raw (un-normalized) tract variables: joint DTW with no
    per-channel normalization, keeping the absolute constriction scale."""

    _norm_fn = staticmethod(_identity_norm)


class ARTNADTVAugEvaluator(_ARTBaseEvaluator):
    """ART-NAD on tract variables augmented with prosodic channels: joint DTW
    over (TV + log-F0 + log-RMS), z-scored per channel."""

    _dtw_fn = staticmethod(_dtw_joint)
    _norm_fn = staticmethod(_zscore_per_channel)
    _f0_transform = "log"

    def _augment(self, ema, audio):
        """Hook: produce the augmented feature matrix from EMA + raw audio.
        Subclasses override to swap the appended channels."""
        return _augment_ema_with_prosody(ema, audio, self._f0_transform)

    def _get_features(self, audio_path, start_time, end_time):
        cache_key = (audio_path, start_time, end_time)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            offset = start_time if start_time != 0.0 else 0
            audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)
            if audio is None or len(audio) == 0:
                result = (None, f"Audio at {audio_path} could not be loaded or is empty.")
                self._feature_cache[cache_key] = result
                return result
            ema = self._feats(audio)
            feats = self._augment(ema, audio)
            if feats.shape[0] < _MIN_FEATURE_LEN:
                result = (None, f"Aug-feature length {feats.shape[0]} < {_MIN_FEATURE_LEN} for {audio_path}.")
                self._feature_cache[cache_key] = result
                return result
            feats = self._norm_fn(feats)
            result = (feats, None)
            self._feature_cache[cache_key] = result
            return result
        except Exception as e:
            result = (None, f"Failed to process {audio_path}: {e}")
            self._feature_cache[cache_key] = result
            return result


# ----------------------------------------------------------------------------
# Trimmed variants (forced-alignment VAD), mirroring
# :class:`pathbench.evaluators.nad.TrimmedNADEvaluator` two-pass fallback.
# ----------------------------------------------------------------------------


class _TrimmedARTBaseEvaluator(ReferenceTxtAndAudioEvaluator):
    """Trimmed ART-NAD base class. Two-pass scoring: try with FA trimming
    enabled; if the trimmer fails on ANY audio in the group (test or any
    reference), fall back to untrimmed for the whole group so the DTW
    distances within a group remain comparable. Mirrors the structure of
    :class:`pathbench.evaluators.nad.TrimmedNADEvaluator`.
    """

    _dtw_fn = staticmethod(_dtw_joint)
    _norm_fn = staticmethod(_zscore_per_channel)

    def __init__(self, runner: ArticulatoryRunner,
                 trimmer: Optional[FATrimmer] = None):
        self.runner = runner
        self.trimmer = trimmer
        # Cache keyed by (path, start, end, use_trimming) so the trimmed and
        # untrimmed versions of the same file are stored separately.
        self._feature_cache: dict = {}

    def _feats(self, audio):
        """Tract-variable trajectory for one waveform, from the TV inversion
        checkpoint (requires a runner built with a custom TV checkpoint)."""
        return self.runner.extract_tv(audio)

    def _get_features(self, audio_path, transcription, language,
                      start_time, end_time, use_trimming):
        cache_key = (audio_path, start_time, end_time, use_trimming)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        use_segment = start_time != 0.0 or end_time != -1.0
        audio = None
        try:
            if use_trimming and self.trimmer and not use_segment:
                trimmed = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if trimmed and len(trimmed[0]) > 0:
                    audio, _ = trimmed

            if audio is None:  # fallback for failed trim or trimming disabled
                duration = end_time - start_time if end_time != -1.0 else None
                offset = start_time if start_time != 0.0 else 0
                audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

            if audio is None or len(audio) == 0:
                result = (None, f"Audio at {audio_path} could not be loaded or is empty.")
                self._feature_cache[cache_key] = result
                return result

            ema = self._feats(audio)
            if ema.shape[0] < _MIN_FEATURE_LEN:
                result = (None, f"EMA length {ema.shape[0]} < {_MIN_FEATURE_LEN} for {audio_path}.")
                self._feature_cache[cache_key] = result
                return result

            ema = self._norm_fn(ema)
            result = (ema, None)
            self._feature_cache[cache_key] = result
            return result
        except Exception as e:
            result = (None, f"Failed to process {audio_path}: {e}")
            self._feature_cache[cache_key] = result
            return result

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        reference_audios: List[tuple],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        if not reference_audios:
            return None

        # --- Decide whether trimming is even attempted. Trimming requires no
        # external segment bounds (the FA trimmer trims the whole file). ---
        use_test_segment = start_time != 0.0 or end_time != -1.0
        use_ref_segments = any(rs != 0.0 or re != -1.0 for _, rs, re in reference_audios)
        attempt_trim = self.trimmer is not None and not use_test_segment and not use_ref_segments

        test_feats = None
        ref_feats: list = []
        use_trimming = attempt_trim

        # --- Pass 1: trimming enabled. Discard whole group on any failure. ---
        if use_trimming:
            errors = []
            test_feats, err = self._get_features(audio_path, transcription, language, start_time, end_time, True)
            if err:
                errors.append(err)
            ref_feats = []
            for ref_path, rs, re in reference_audios:
                r_feats, err = self._get_features(ref_path, transcription, language, rs, re, True)
                if err:
                    errors.append(err)
                ref_feats.append(r_feats)

            if errors or test_feats is None or any(f is None for f in ref_feats):
                print(f"Warning: trimmed EMA failed for group {utterance_id}; falling back to untrimmed. Errors: {errors}")
                test_feats = None
                ref_feats = []
                use_trimming = False
            else:
                ref_feats = [f for f in ref_feats if f is not None]

        # --- Pass 2: untrimmed fallback (or first pass if trimming skipped). ---
        if not use_trimming:
            test_feats, err = self._get_features(audio_path, transcription, language, start_time, end_time, False)
            if err:
                print(f"Error: untrimmed EMA failed for test {utterance_id}: {err}")
                return None
            ref_feats = []
            for ref_path, rs, re in reference_audios:
                r_feats, err = self._get_features(ref_path, transcription, language, rs, re, False)
                if err:
                    print(f"Warning: untrimmed EMA failed for ref {ref_path} in group {utterance_id}, skipping. {err}")
                else:
                    ref_feats.append(r_feats)

        if test_feats is None or not ref_feats:
            print(f"Error: no usable EMA for group {utterance_id}.")
            return None

        distances = []
        for r in ref_feats:
            try:
                distances.append(self._dtw_fn(test_feats, r))
            except Exception as e:
                print(f"Error during DTW for {utterance_id}: {e}")
                distances.append(np.nan)

        return np.nanmean(distances) if distances else None


class TrimmedARTNADTVEvaluator(_TrimmedARTBaseEvaluator):
    """ART-NAD-FA on tract variables: joint DTW over per-utterance z-scored
    quasi-TV trajectories, with forced-alignment silence trimming."""

    _dtw_fn = staticmethod(_dtw_joint)


class TrimmedARTNADTVRawEvaluator(TrimmedARTNADTVEvaluator):
    """ART-NAD-FA on raw (un-normalized) tract variables: joint DTW with no
    per-channel normalization, forced-alignment trimmed."""

    _norm_fn = staticmethod(_identity_norm)


class TrimmedARTNADTVAugEvaluator(_TrimmedARTBaseEvaluator):
    """ART-NAD-Aug-FA on tract variables: joint DTW over (TV + log-F0 +
    log-RMS), z-scored, with forced-alignment silence trimming."""

    _dtw_fn = staticmethod(_dtw_joint)
    _norm_fn = staticmethod(_zscore_per_channel)
    _f0_transform = "log"

    def _augment(self, ema, audio):
        """Hook: produce the augmented feature matrix from EMA + raw audio."""
        return _augment_ema_with_prosody(ema, audio, self._f0_transform)

    def _get_features(self, audio_path, transcription, language,
                      start_time, end_time, use_trimming):
        cache_key = (audio_path, start_time, end_time, use_trimming)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        use_segment = start_time != 0.0 or end_time != -1.0
        audio = None
        try:
            if use_trimming and self.trimmer and not use_segment:
                trimmed = self.trimmer.trim(audio_path, transcription, language, start_time, end_time)
                if trimmed and len(trimmed[0]) > 0:
                    audio, _ = trimmed
            if audio is None:
                duration = end_time - start_time if end_time != -1.0 else None
                offset = start_time if start_time != 0.0 else 0
                audio, _ = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)
            if audio is None or len(audio) == 0:
                result = (None, f"Audio at {audio_path} could not be loaded or is empty.")
                self._feature_cache[cache_key] = result
                return result
            ema = self._feats(audio)
            feats = self._augment(ema, audio)
            if feats.shape[0] < _MIN_FEATURE_LEN:
                result = (None, f"Aug-feature length {feats.shape[0]} < {_MIN_FEATURE_LEN} for {audio_path}.")
                self._feature_cache[cache_key] = result
                return result
            feats = self._norm_fn(feats)
            result = (feats, None)
            self._feature_cache[cache_key] = result
            return result
        except Exception as e:
            result = (None, f"Failed to process {audio_path}: {e}")
            self._feature_cache[cache_key] = result
            return result


class TrimmedARTNADTVRawAugCEvaluator(TrimmedARTNADTVAugEvaluator):
    """ART-NAD-Aug-FA on raw tract variables + centered prosody: TV channels
    left un-normalized, log-F0/log-RMS mean-centered, FA-trimmed."""

    _norm_fn = staticmethod(_raw_with_centered_prosody)


class TrimmedARTNADTVRawAugZEvaluator(TrimmedARTNADTVAugEvaluator):
    """ART-NAD-Aug-FA on raw tract variables + z-scored prosody: TV channels
    left un-normalized, log-F0/log-RMS z-scored, FA-trimmed."""

    _norm_fn = staticmethod(_raw_with_zscored_prosody)
