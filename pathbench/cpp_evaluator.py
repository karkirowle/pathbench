import math
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import librosa

from pathbench.evaluator import Evaluator

eps = np.finfo(float).eps

def cpp_func(x, fs, normOpt, dBScaleOpt):
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
    dBScaleOpt: binary
        True or False for using decibel scale

    Returns
    -----------
    cpp: ndarray
        The CPP with time values 
    """
    # Settings
    frame_length = int(np.round(0.04*fs)) # from _
    frame_shift = int(np.round(0.01*fs)) # from _
    half_len = int(np.round(frame_length/2)) # from _
    x_len = len(x)
    frame_len = half_len*2 + 1
    NFFT = 2**(math.ceil(np.log(frame_len)/np.log(2)))
    quef = np.linspace(0, frame_len/1000, NFFT)

    # Allowed quefrency range
    pitch_range = [60, 333.3]
    quef_lim = [int(np.round(fs/pitch_range[1])), # round_
                int(np.round(fs/pitch_range[0]))] # round_
    quef_seq = range(quef_lim[0]-1, quef_lim[1])

    # Time samples
    time_samples = np.array(
        range(frame_length+1, x_len-frame_length+1, frame_shift))
    N = len(time_samples)
    if N == 0:
        return np.array([]), np.array([])
    frame_start = time_samples-half_len
    frame_stop = time_samples+half_len

    # High-pass filtering
    HPfilt_b = [1, -0.97]
    x = scipy.signal.lfilter(HPfilt_b, 1, x)

    # Frame matrix
    frameMat = np.zeros([NFFT, N])
    for n in range(0, N):
        frameMat[0: frame_len, n] = x[frame_start[n]-1:frame_stop[n]]

    # Hanning
    def hanning(N):
        x = np.array([i/(N+1) for i in range(1, int(np.ceil(N/2))+1)])
        w = 0.5-0.5*np.cos(2*np.pi*x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int((np.ceil(N % 2))):]))
    win = hanning(frame_len)
    #winmat = np.matlib.repmat(win, N, 1).transpose()
    winmat = np.tile(win, (N, 1)).transpose()
    frameMat = frameMat[0:frame_len, :]*winmat

    # Cepstrum
    SpecMat = np.abs(np.fft.fft(frameMat, axis=0))
    with np.errstate(divide='ignore'):
        SpecdB = 20*np.log10(SpecMat + eps)
    if dBScaleOpt:
        ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB, axis=0)) + eps)
    else:
        ceps = 2*np.log(np.abs(np.fft.fft(SpecdB, axis=0)) + eps)

    # Finding the peak
    ceps_lim = ceps[quef_seq, :]
    ceps_max = ceps_lim.max(axis=0)
    max_index = ceps_lim.argmax(axis=0)

    # Normalisation
    ceps_norm = np.zeros([N])
    if normOpt == 'line':
        for n in range(0, N):
            p = np.polyfit(quef_seq, ceps_lim[:, n], 1)
            ceps_norm[n] = np.polyval(p, quef_seq[max_index[n]])
    elif normOpt == 'mean':
        ceps_norm = np.mean(ceps_lim)

    cpp = ceps_max-ceps_norm

    return cpp, time_samples


class CPPEvaluator(Evaluator):
    """An evaluator that computes the Cepstral Peak Prominence (CPP)."""

    def __init__(self, normOpt: str = 'line', dBScaleOpt: bool = True):
        self.normOpt = normOpt
        self.dBScaleOpt = dBScaleOpt

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        **kwargs,
    ) -> Optional[float]:
        """
        Computes the CPP for the given audio file.
        """
        try:
            audio, fs = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None

        cpp, _ = cpp_func(audio, fs, self.normOpt, self.dBScaleOpt)

        if len(cpp) == 0:
            return None

        return np.mean(cpp)
