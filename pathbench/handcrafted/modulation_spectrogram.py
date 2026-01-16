
import librosa
import numpy as np
import scipy.signal

def compute_modulation_spectrogram(y, sr, n_fft=2048, hop_length=512, mod_n_fft=2048):
    """
    Computes a modulation spectrogram.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate of y.
    - n_fft: FFT window size for the initial STFT.
    - hop_length: Hop length for the initial STFT.
    - mod_n_fft: FFT window size for the modulation FFT.

    Returns:
    - mod_spectrogram_db: The modulation spectrogram in dB.
    - freqs: Frequencies of the original spectrogram.
    - mod_freqs: Modulation frequencies.
    """

    # 1. Compute the Short-Time Fourier Transform (STFT)
    # This gives us a complex-valued matrix where rows are frequency bins
    # and columns are time frames.
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # 2. Get the magnitude spectrogram
    # We take the absolute value to get the magnitude, then convert to dB.
    magnitude_spectrogram = np.abs(stft_matrix)
    
    # It's common to work with log-magnitude spectrogram for modulation analysis
    log_magnitude_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)

    # 3. Apply FFT along the time axis for each frequency band
    # This is the core step for the modulation spectrogram.
    # We perform an FFT on each row (frequency band) of the log-magnitude spectrogram.
    # The result will have modulation frequencies along the new axis.
    
    # Pad or truncate to mod_n_fft if necessary for consistent FFT length
    # For simplicity, we'll assume mod_n_fft is chosen appropriately.
    # If the number of time frames is less than mod_n_fft, padding will occur.
    # If it's more, we'll take the first mod_n_fft frames or handle it differently.
    
    # For a more robust implementation, one might window the time frames before FFT
    # and handle overlapping windows for modulation analysis, similar to STFT.
    # For this example, we'll apply a direct FFT to each frequency band's time series.
    
    # Ensure the number of time frames is at least mod_n_fft for a meaningful FFT
    if log_magnitude_spectrogram.shape[1] < mod_n_fft:
        # Pad with zeros if not enough time frames
        padded_spec = np.pad(log_magnitude_spectrogram, ((0, 0), (0, mod_n_fft - log_magnitude_spectrogram.shape[1])), mode='constant')
        modulation_spectrum_complex = np.fft.fft(padded_spec, axis=1)
    else:
        # Take the first mod_n_fft frames if too many, or implement sliding window
        modulation_spectrum_complex = np.fft.fft(log_magnitude_spectrogram[:, :mod_n_fft], axis=1)


    # Take the magnitude of the modulation spectrum
    modulation_spectrogram = np.abs(modulation_spectrum_complex)

    # Convert to dB for visualization
    mod_spectrogram_db = librosa.amplitude_to_db(modulation_spectrogram, ref=np.max)

    # Get the frequencies for the original spectrogram
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Get the modulation frequencies
    # The FFT output has frequencies from 0 to sr_mod / 2 (Nyquist)
    # where sr_mod is the effective sampling rate of the spectrogram's time axis.
    # The sampling rate of the spectrogram's time axis is sr / hop_length.
    mod_sr = sr / hop_length
    mod_freqs = np.fft.fftfreq(mod_n_fft, d=1/mod_sr)
    
    # We are usually interested in positive modulation frequencies
    mod_freqs = mod_freqs[:mod_n_fft // 2 + 1]
    mod_spectrogram_db = mod_spectrogram_db[:, :mod_n_fft // 2 + 1]


    return mod_spectrogram_db, freqs, mod_freqs

def extract_score(audio_path):
    """
    Extracts a score from an audio file based on the modulation spectrogram.

    Parameters:
    - audio_path: Path to the audio file.

    Returns:
    - score: A single score representing the modulation spectrogram.
    """
    y, sr = librosa.load(audio_path)
    mod_spec_db, _, _ = compute_modulation_spectrogram(y, sr)
    
    # Calculate the mean of the modulation spectrogram as the score
    score = np.mean(mod_spec_db)
    
    return score
