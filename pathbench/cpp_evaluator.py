import math
from typing import Optional

import numpy as np
import scipy.signal
import librosa
import parselmouth
from parselmouth.praat import call

from pathbench.evaluator import ReferenceFreeEvaluator

eps = np.finfo(float).eps

def cpp_func(x, fs, normOpt, double_log=False):
    """
    Computes cepstral peak prominence for a given signal

    Parameters
    -----------
    x: ndarray
        The audio signal
    fs: integer
        The sampling frequency
    normOpt: string
        'line', 'mean' or 'nonorm' for selecting normalisation type
    double_log: bool
        If True, uses the legacy double-log formulation (incorrect but kept for comparison).
        If False (default), uses the standard CPP formulation.

    Returns
    -----------
    cpp: ndarray
        The CPP with time values
    """
    # Settings
    frame_length = int(np.round(0.04*fs))
    frame_shift = int(np.round(0.01*fs))
    half_len = int(np.round(frame_length/2))
    x_len = len(x)
    frame_len = half_len*2 + 1
    NFFT = 2**(math.ceil(np.log(frame_len)/np.log(2)))

    # Allowed quefrency range (pitch 60-333.3 Hz)
    pitch_range = [60, 333.3]
    quef_lim = [int(np.round(fs/pitch_range[1])),
                int(np.round(fs/pitch_range[0]))]
    quef_seq = range(quef_lim[0]-1, quef_lim[1])

    # Time samples
    time_samples = np.array(
        range(frame_length+1, x_len-frame_length+1, frame_shift))
    N = len(time_samples)
    if N == 0:
        return np.array([]), np.array([])
    frame_start = time_samples-half_len
    frame_stop = time_samples+half_len

    # High-pass filtering (pre-emphasis)
    HPfilt_b = [1, -0.97]
    x = scipy.signal.lfilter(HPfilt_b, 1, x)

    # Frame matrix
    frameMat = np.zeros([NFFT, N])
    for n in range(0, N):
        frameMat[0: frame_len, n] = x[frame_start[n]-1:frame_stop[n]]

    # Hanning window
    def hanning(N):
        x = np.array([i/(N+1) for i in range(1, int(np.ceil(N/2))+1)])
        w = 0.5-0.5*np.cos(2*np.pi*x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int((np.ceil(N % 2))):]))
    win = hanning(frame_len)
    winmat = np.tile(win, (N, 1)).transpose()
    frameMat = frameMat[0:frame_len, :]*winmat

    # Cepstrum computation
    SpecMat = np.abs(np.fft.fft(frameMat, axis=0))
    with np.errstate(divide='ignore'):
        SpecdB = 20*np.log10(SpecMat + eps)

    if double_log:
        # Legacy (incorrect) formulation: extra log of cepstrum
        ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB, axis=0)) + eps)
    else:
        # Standard CPP: cepstrum = FFT(log(spectrum))
        ceps = np.abs(np.fft.fft(SpecdB, axis=0))

    # Finding the peak in quefrency range
    ceps_lim = ceps[quef_seq, :]
    ceps_max = ceps_lim.max(axis=0)
    max_index = ceps_lim.argmax(axis=0)

    # Normalisation (regression line or mean)
    ceps_norm = np.zeros([N])
    if normOpt == 'line':
        for n in range(0, N):
            p = np.polyfit(quef_seq, ceps_lim[:, n], 1)
            ceps_norm[n] = np.polyval(p, quef_seq[max_index[n]])
    elif normOpt == 'mean':
        ceps_norm = np.mean(ceps_lim)

    cpp = ceps_max - ceps_norm

    return cpp, time_samples


class CPPEvaluator(ReferenceFreeEvaluator):
    """Cepstral Peak Prominence (standard formulation). Reference-free."""

    def __init__(self, normOpt: str = 'line'):
        self.normOpt = normOpt
        self.double_log = False

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            audio, fs = librosa.load(audio_path, sr=16000, mono=True,
                                     offset=start_time, duration=duration)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        if audio is None or len(audio) == 0:
            print(f"Warning: Audio for {audio_path} is empty.")
            return None

        return self._score_audio(audio, fs)

    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        cpp, _ = cpp_func(audio, fs, self.normOpt, double_log=self.double_log)
        if len(cpp) == 0:
            return None
        return float(np.mean(cpp))


class CPPDoubleLogEvaluator(CPPEvaluator):
    """Legacy CPP evaluator using the double-log formulation (incorrect but kept for comparison).

    The standard CPP is: peak(FFT(log(spectrum))) - regression_line
    This version uses: peak(log(FFT(log(spectrum)))) - regression_line
    """

    def __init__(self, normOpt: str = 'line'):
        super().__init__(normOpt=normOpt)
        self.double_log = True


class PraatCPPEvaluator(ReferenceFreeEvaluator):
    """CPP evaluator using Praat's built-in PowerCepstrogram implementation via parselmouth.

    This is the reference implementation used in clinical voice research.
    Uses Praat's "Get CPPS" command which computes smoothed Cepstral Peak Prominence
    following the methodology of Hillenbrand et al. (1994).

    Reference: https://www.fon.hum.uva.nl/praat/manual/PowerCepstrogram__Get_CPPS___.html
    """

    def __init__(
        self,
        pitch_floor: float = 60.0,
        pitch_ceiling: float = 330.0,
        time_averaging_window: float = 0.02,
        quefrency_averaging_window: float = 0.0005,
    ):
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_averaging_window = time_averaging_window
        self.quefrency_averaging_window = quefrency_averaging_window

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        use_segment = start_time != 0.0 or end_time != -1.0
        try:
            if use_segment:
                duration = end_time - start_time if end_time != -1.0 else None
                audio, fs = librosa.load(audio_path, sr=16000, mono=True,
                                         offset=start_time, duration=duration)
                sound = parselmouth.Sound(audio, sampling_frequency=fs)
            else:
                sound = parselmouth.Sound(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None

        if sound.n_samples < 400:
            print(f"Warning: Audio too short for {audio_path}.")
            return None

        return self._score_audio_from_sound(sound)

    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        sound = parselmouth.Sound(audio, sampling_frequency=fs)
        if sound.n_samples < 400:
            return None
        return self._score_audio_from_sound(sound)

    def _score_audio_from_sound(self, sound) -> Optional[float]:
        try:
            power_cepstrogram = call(
                sound,
                "To PowerCepstrogram",
                self.pitch_floor,
                0.002,
                8000.0,
                50.0
            )
            cpps = call(
                power_cepstrogram,
                "Get CPPS",
                "yes",
                self.time_averaging_window,
                self.quefrency_averaging_window,
                self.pitch_floor,
                self.pitch_ceiling,
                0.05,
                "Parabolic",
                0.001,
                0.05,
                "Straight",
                "Robust slow"
            )
            return cpps
        except Exception as e:
            print(f"Error computing Praat CPP: {e}")
            return None
