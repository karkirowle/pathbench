#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from scipy.stats import pearsonr
from scipy.signal import spectrogram, windows, convolve
from scipy.ndimage import uniform_filter1d

import numpy as np
import soundfile as sf
import os.path
from ltfatpy import dgtreal # TF feature repesentation extraction
from numpy.linalg import norm # distance calc in dtw
from dtw import dtw #
from pathbench.utils import normalise_signal, moving_average_filtering
from typing import List, Optional

import librosa
from pathbench.evaluator import ReferenceAudioEvaluator

eps = np.finfo(float).eps


class STOI():


    def __init__(self, reference_words: np.ndarray,
                 test_words: np.ndarray,
                 normalization_method: str,
                 centroid_ind: int,
                 frame_deletion:bool = True,
                 fs: int = 16000):
        '''
        Short Term Objective Intelligibility (STOI) measure

        :params
        reference_words: list of reference words
        test_words: list of test words
        normalization_method: 'RMS' or 'zero_mean'
        frame_deletion: True or False

        '''
        self.reference_words = [w.copy() for w in reference_words]




        self.train_target = np.ones(len(reference_words))
        self.test_words = [w.copy() for w in test_words]
        print("self.test_words", self.test_words)
        self.normalization_method = normalization_method
        self.Tw = 32  # 32
        self.Ts = 16  # 16
        self.J = 15 # Number of 1/3 octave bands
        self.mn = 150 # Center frequency of first 1/3 octave band in Hz.
        self.centroid_ind = centroid_ind
        self.consecN = 15

        Beta = -15 # lower SDR-bound
        self.c = 10**(-Beta/20) # constant for clipping procedure

        self.frame_deletion = frame_deletion
        self.considered_first_bin = 0
        self.fs = fs

        self.Nw = int(round(1E-3 * self.Tw * self.fs))
        self.Ns = int(round(1E-3 * self.Ts * self.fs))

        self.nfft = int(2**np.ceil(np.log2(self.Nw)))

        self.stoi_val = 0
        self.estoi_val = 0

        self.ref_create()
        self.STOI_value()
        #return self.stoi_val, self.estoi_val
    @staticmethod
    def thirdoct(fs, N_fft, number_of_bands, mn):
        """
        Extracts a one-thirdthird octave band representation

        :param fs: sampling frequency (Hz)
        :param N_fft: number of bins for the FFT
        :param number_of_bands: number of one-third octave bands, marked as J in the paper
        :param mn:
        :return:
        """

        f = np.linspace(0, fs, N_fft + 1)
        f = f[0:int(N_fft/2) + 1]
        k = np.arange(0, (number_of_bands))
        cf = 2**(k/3)*mn
        fl = np.sqrt((2**(k/3)*mn)*2**((k - 1)/3)*mn)
        fr = np.sqrt((2**(k/3)*mn)*2**((k + 1)/3)*mn)
        A = np.zeros((number_of_bands, len(f)))
        for i in np.arange(0, (len(cf))):
            b = np.argmin((f - fl[i])**2)
            fl[i] = f[b]
            fl_ii = b
            b = np.argmin((f - fr[i])**2)
            fr[i] = f[b]
            fr_ii = b
            A[i, fl_ii:(fr_ii)] = 1
        A = A[0:number_of_bands, :]
        cf = cf[0:number_of_bands]
        return A, cf


    @staticmethod
    def difference_oct(X, Y):
        return np.abs(np.log10((X)) - np.log10((Y)))

    @staticmethod
    def dgt_real_substitute(signal, wi, ns, nfft, nw):

        # window = windows.hamming(window_length, sym=True)
        window = windows.hamming(512, sym=True)
        nfft = 512
        n_overlap = nw - ns
        _, _, D = spectrogram(signal, 16000, window=window, noverlap=n_overlap, nperseg=nw, nfft=nfft, mode='complex')
        #D_real = np.abs(D)

        return D

    def log_octave_transform_extractor(self, word_set):

        log_octave_transforms = [None] * len(word_set) # Storage?
        mean_sum = np.zeros((1, self.J))

        for num, word_signal in enumerate(word_set):

            # Perform normalisation on the audio signal
            #print("word_signal before normalise", word_signal)
            word_signal = normalise_signal(word_signal, self.normalization_method)
            #print("word_signal after normalise", word_signal)
            self.H, cf = self.thirdoct(self.fs, self.nfft, self.J, self.mn) # also H in the paper ---> this is a filter bank most liekly

            wi = {'name': ('tight', 'hamming'), 'M': self.Nw} # these are most likely some windowing properties

            # Performs a discrete Gabor transform with a Hamming-window, and time shift of Ns, with Nfft modulations
            # Ls is length of input signal, G is gabor length

            gabor_word_signal, Ls, g = dgtreal(word_signal, wi, self.Ns, self.nfft)
            #gabor_word_signal_2 = self.dgt_real_substitute(word_signal, wi, self.Ns, self.nfft, self.Nw)

            #print("gabor_word_signal shape", gabor_word_signal.shape)
            # Delete 0 columns
            gabor_word_signal = np.delete(gabor_word_signal, (np.where(gabor_word_signal == 0)[1]), 1)

            # Modify the Gabor transformed signal with the H filter bank
            X = np.sqrt(np.dot(self.H, np.abs(gabor_word_signal)**2))

            # Perform moving average filtering to smooth the values due to the discontinunities caused by the removal ?? TODO
            X = moving_average_filtering(X)


            log_octave_transforms[num] = (np.log10(np.transpose(X))) - np.mean(np.log10(np.transpose(X)), axis=0)
            mean_sum += np.sum((np.log10(np.transpose(X))), axis=0)/X.shape[1]

        return log_octave_transforms

    def align_dtw(self, control, test, frame_deletion: bool, test_time: bool):
        """

        Aligns two TF representations together using dynamic time warping (DTW)

        :param control: control signal to align with (np.ndarray)
        :param test: test (pathological) signal to align with (np.ndarray)
        :param frame_deletion: whether to delete repeated frames. My intuition is it is useful because you align two identical
        length samples, and you don't need to decide which to align to? (TODO: check)
        :param test_time: i have no idea (TODO: check)
        :return: dtw frame paths
        """

        # Calculate the path using two-norm distance based DTW
        #alignment = dtw(control, test, dist_method='euclidean', keep_internals=True)
        #dist = alignment.distance
        #path = np.array([alignment.index1, alignment.index2])

        # NOTE: Seems to have varied a lot by switching DTW implementations
        _, path = librosa.sequence.dtw(X=control.T, Y=test.T, metric='euclidean')
        path = np.transpose(path)
        #librosa

        if not test_time:
            if frame_deletion:
                new_path_control = np.delete(np.array(path)[0, :], 1 + np.where(np.diff(np.array(path)[0, :]) == 0)[0])
                new_path_test = np.delete(np.array(path)[1, :], 1 + np.where(np.diff(np.array(path)[0, :]) == 0)[0])
            else:
                new_path_control = np.array(path)[0, :]
                new_path_test = np.array(path)[1, :]
        else:
            if self.frame_deletion:
                # Paper: Repeated frames affects intelligibility measures
                new_path_control = np.delete(np.array(path)[1, :], 1 + np.where(np.diff(np.array(path)[1, :]) == 0)[0])
                new_path_test = np.delete(np.array(path)[0, :], 1 + np.where(np.diff(np.array(path)[1, :]) == 0)[0])
                new_path_control = np.delete(new_path_control, 1 + np.where(np.diff(new_path_test) == 0)[0])
                new_path_test = np.delete(new_path_test, 1 + np.where(np.diff(new_path_test) == 0)[0])
            else:
                new_path_control = np.array(path)[1, :]
                new_path_test = np.array(path)[0, :]

        return new_path_control, new_path_test


    def ref_create(self):
        """
        Creates the global reference signal for the comparison based on the reference signal which should contain common word/utterance
        NOTE: global reference is not exactly the same as centroid. Centroid is the one that's used for creating the global reference.

        :return:
        """

        # Creates the reference I guess ?

        self.reference_log_octave_transforms = [None] * len(self.reference_words) # Storage?
        self.test_log_octave_transforms = [None] * len(self.test_words) # Storage?

        # TODO: Is there any purpose to mean sum calculations? (removed for now)
        self.reference_log_octave_transforms = self.log_octave_transform_extractor(self.reference_words)
        self.test_log_octave_transforms = self.log_octave_transform_extractor(self.test_words)

        # =============================================================================
        # #XXXXX     The next part create reference from many octave band representations
        # =============================================================================

        subjects = [self.reference_words[i] for i in np.where(self.train_target == 1)[0]]
        number_of_subjects = len(subjects)
        centroid = self.reference_log_octave_transforms[np.where(self.train_target == 1)[0][self.centroid_ind]]  # initial representation [centroid]
        sum_f = np.zeros_like(centroid) # ?
        sum_f_num = np.zeros((np.size(centroid, 0), 1)) # ?

        # subject_range is all other representations except centroid
        subject_range = [i for i in range(number_of_subjects) if i != self.centroid_ind]

        # Here all the other representations are aligned to the centroid representation. Then the energies of the
        # octave-band rperesentations are summed

        for num in subject_range:
            aln2 = self.reference_log_octave_transforms[np.where(self.train_target == 1)[0][num]]
            new_path_cont, new_path_test = self.align_dtw(centroid,aln2, frame_deletion=True,test_time=False)

            # My understanding that this sums the energies in the frames like Eq1, but not entirely sure
            for frame_ind in range(np.size(centroid, 0)):
                sum_f[frame_ind, :] += np.sum(10**aln2[new_path_test[new_path_cont == frame_ind], :], axis=0)

                # This array holds is filled with frame counts
                sum_f_num[frame_ind, 0] += len(new_path_test[new_path_cont == frame_ind])
        # Final reference representations

        ref_for_tr = np.log10(sum_f/sum_f_num)

        # Repeat the reference for all test words
        tr = [ref_for_tr for _ in range(len(self.test_words))]

        self.ref_test = tr

    @staticmethod
    def _safe_pearsonr(x, y):
        """
        Helper to calculate Pearson correlation safely.
        Returns (0.0, 1.0) if input variance is effectively zero (constant input),
        otherwise calls scipy.stats.pearsonr.
        """
        # Check for constant inputs (near-zero standard deviation)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0, 1.0  # Correlation 0, p-value 1 (or return NaN based on preference)
            
        return pearsonr(x, y)


    def stoi_calculation(self, N, X, Y, frame_shift, subject_id):

        d_interm = np.zeros((np.size(X, axis=0), len(np.arange(N, np.size(X, axis=1) + 1, frame_shift))))

        for i, m in enumerate(range(N, X.shape[1] + 1, frame_shift)):
            x_segment = X[:, (m - N):m]  # region with length N of clean TF-Units for all j
            y_segment = Y[:, (m - N):m]  # region with length N of processed TF-units for all j
            alpha = np.sqrt(np.sum(x_segment ** 2, axis=1) / np.sum(y_segment ** 2, axis=1))
            aY_seg = y_segment * alpha[:, np.newaxis]
            for j in range(self.J):
                d1 = (self.c+1) * x_segment[j, :]
                d2 = aY_seg[j, :]
                y_prime = np.min(np.array([d1, d2]), axis=0)
                d_interm[j, i], _ = self._safe_pearsonr(x_segment[j, :], y_prime)  # Eq 2 from Parvaneh's paper

        # NaN columns are removed from the calculation
        tmp = np.isnan(d_interm)
        tmp = np.sum(tmp, axis=0)
        self.stoi_val[subject_id] = np.mean(d_interm[self.considered_first_bin:, tmp == 0])

    def estoi_calculation(self, N, X, Y, frame_shift, subject_id):

        d_interm_e = np.zeros((N, len(np.arange(N, np.size(X, axis=1) + 1, frame_shift))))

        for ind, m in enumerate(range(N, X.shape[1] + 1, frame_shift)):
            y_segment = (Y[:, (m - N):m] - np.mean(Y[:, (m - N):m], axis=1, keepdims=True)) / \
                        (np.std(Y[:, (m - N):m], axis=1, keepdims=True) + eps)
            x_segment = (X[:, (m - N):m] - np.mean(X[:, (m - N):m], axis=1, keepdims=True)) / \
                        (np.std(X[:, (m - N):m], axis=1, keepdims=True) + eps)
            for j in range(N):
                d_interm_e[j, ind], _ = self._safe_pearsonr(x_segment[:, j], y_segment[:, j])  # Eq 4 from Parvaneh's paper

        tmp = (np.isnan(d_interm_e))
        tmp = np.sum(tmp, axis=0)
        estoi_val = np.mean((d_interm_e[self.considered_first_bin:, (tmp == 0)]))
        self.estoi_val[subject_id] = estoi_val

    def STOI_value(self):

        self.stoi_val = np.zeros(len(self.test_words))
        self.estoi_val = np.zeros(len(self.test_words))

        number_of_subjects = len(self.test_words)

        self.aligned_ref = [None]
        self.aligned_test = [None] * number_of_subjects

        self.aligned_ref = None
        self.aligned_test = None

        for subject_id in range(number_of_subjects):

            aln1 = self.test_log_octave_transforms[subject_id]

            new_path_cont, new_path_test = self.align_dtw(aln1, self.ref_test[subject_id],
                                                          frame_deletion=self.frame_deletion,
                                                          test_time=True)

            aln1 = 10 ** aln1[new_path_test, :]
            cont = 10 ** self.ref_test[subject_id][new_path_cont, :]

            self.aligned_ref = cont
            self.aligned_test = aln1

            X = np.transpose(cont)
            Y = np.transpose(aln1)

            frame_shift = 1
            N  =  np.min([self.consecN, np.size(X, axis= 1)])

            try:

                # STOI
                self.stoi_calculation(N, X, Y, frame_shift, subject_id)

                # ESTOI
                self.estoi_calculation(N, X, Y, frame_shift, subject_id)

            except ValueError as err:
                self.stoi_val = [np.nan]
                self.estoi_val = [np.nan]
                #print(err)
                #print('error in:', self.test_words[subject_id])
                #pass

class ReferenceEvaluator:
    """Deprecated. Kept for backward compatibility. Use ReferenceAudioEvaluator instead."""

    def __init__(self, **kwargs):
        self.stoi_kwargs = kwargs


class PSTOIEvaluator(ReferenceAudioEvaluator):
    """An evaluator that uses PSTOI to compute a score."""

    def __init__(self, **kwargs):
        self.stoi_kwargs = kwargs

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        reference_audios: List[tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """
        Computes the PSTOI score.
        """
        duration = end_time - start_time if end_time != -1 else None
        test_audio, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration, dtype=np.float64)

        reference_audios_data = []
        for ref_path, ref_start, ref_end in reference_audios:
            ref_duration = ref_end - ref_start if ref_end != -1 else None
            ref_audio, _ = librosa.load(ref_path, sr=16000, offset=ref_start, duration=ref_duration, dtype=np.float64)
            reference_audios_data.append(ref_audio)

        stoi_object = STOI(
            reference_words=reference_audios_data,
            test_words=[test_audio],
            **self.stoi_kwargs
        )
        return stoi_object.stoi_val[0]

class ESTOIEvaluator(ReferenceAudioEvaluator):
    """An evaluator that uses P-ESTOI to compute a score."""

    def __init__(self, **kwargs):
        self.stoi_kwargs = kwargs

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        reference_audios: List[tuple[str, float, float]],
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """
        Computes the P-ESTOI score.
        """
        duration = end_time - start_time if end_time != -1 else None
        test_audio, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration, dtype=np.float64)

        # Check if test_audio is full silence
        if np.all(test_audio == 0):
            print(f"Warning: Test audio {audio_path} is silent. Returning P-ESTOI score of 0.0.")
            return 0.0

        reference_audios_data = []
        for ref_path, ref_start, ref_end in reference_audios:
            ref_duration = ref_end - ref_start if ref_end != -1 else None
            ref_audio, _ = librosa.load(ref_path, sr=16000, offset=ref_start, duration=ref_duration, dtype=np.float64)
            reference_audios_data.append(ref_audio)

        stoi_object = STOI(
            reference_words=reference_audios_data,
            test_words=[test_audio],
            **self.stoi_kwargs
        )
        return stoi_object.estoi_val[0]
