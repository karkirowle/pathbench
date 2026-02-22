from typing import Optional
import librosa
from pathbench.evaluator import ReferenceFreeEvaluator, ReferenceTxtEvaluator
import math
import parselmouth
from parselmouth.praat import call
import numpy as np


class WpmEvaluator(ReferenceTxtEvaluator):
    """An evaluator that scores based on the speech rate (words per minute)."""

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        transcription: str,
        language: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """
        Returns the speech rate in words per minute (WPM).
        """
        try:
            duration_s = 0
            duration = end_time - start_time if end_time != -1.0 else None
            audio, fs = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)

            if audio is None or fs is None or len(audio) == 0:
                duration_s = 0
            else:
                duration_s = len(audio) / fs

            if duration_s <= 0:
                return 0.0

            # Count words in transcription
            word_count = len(transcription.split())

            if word_count == 0:
                return 0.0

            # Calculate WPM
            wpm = (word_count / duration_s) * 60
            return wpm
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return None

###########################################################################
#                                                                         #
#  Praat Script Syllable Nuclei                                           #
#  Copyright (C) 2008  Nivja de Jong and Ton Wempe                        #
#                                                                         #
#    This program is free software: you can redistribute it and/or modify #
#    it under the terms of the GNU General Public License as published by #
#    the Free Software Foundation, either version 3 of the License, or    #
#    (at your option) any later version.                                  #
#                                                                         #
#    This program is distributed in the hope that it will be useful,      #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
#    GNU General Public License for more details.                         #
#                                                                         #
#    You should have received a copy of the GNU General Public License    #
#    along with this program.  If not, see http://www.gnu.org/licenses/   #
#                                                                         #
###########################################################################
#
# modified 2010.09.17 by Hugo Quené, Ingrid Persoon, & Nivja de Jong
# Overview of changes:
# + change threshold-calculator: rather than using median, use the almost maximum
#     minus 25dB. (25 dB is in line with the standard setting to detect silence
#     in the "To TextGrid (silences)" function.
#     Almost maximum (.99 quantile) is used rather than maximum to avoid using
#     irrelevant non-speech sound-bursts.
# + add silence-information to calculate articulation rate and ASD (average syllable
#     duration.
#     NB: speech rate = number of syllables / total time
#         articulation rate = number of syllables / phonation time
# + remove max number of syllable nuclei
# + refer to objects by unique identifier, not by name
# + keep track of all created intermediate objects, select these explicitly,
#     then Remove
# + provide summary output in Info window
# + do not save TextGrid-file but leave it in Object-window for inspection
#     (if requested in startup-form)
# + allow Sound to have starting time different from zero
#      for Sound objects created with Extract (preserve times)
# + programming of checking loop for mindip adjusted
#      in the orig version, precedingtime was not modified if the peak was rejected !!
#      var precedingtime and precedingint renamed to currenttime and currentint
#
# + bug fixed concerning summing total pause, feb 28th 2011
###########################################################################


# counts syllables of all sound utterances in a directory
# NB unstressed syllables are sometimes overlooked
# NB filter sounds that are quite noisy beforehand
# NB use Silence threshold (dB) = -25 (or -20?)
# NB use Minimum dip between peaks (dB) = between 2-4 (you can first try;
#                                                      For clean and filtered: 4)
#
#
# Translated to Python in 2019 by David Feinberg
# I changed all the variable names so they are human readable

class PraatSpeechRateEvaluator(ReferenceFreeEvaluator):
    """
    An evaluator that scores based on the speech rate (syllables per second)
    using a Python translation of a Praat script by de Jong and Wempe.
    """

    def score(
        self,
        utterance_id: str,
        audio_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Optional[float]:
        """
        Returns the speech rate in syllables per second.
        """
        try:
            duration = end_time - start_time if end_time != -1.0 else None
            audio, fs = librosa.load(audio_path, sr=None, mono=True, offset=start_time, duration=duration)
        except Exception as e:
            print(f"Error loading audio {audio_path} with PraatSpeechRateEvaluator: {e}")
            return None
        return self._score_audio(audio, fs)

    def _score_audio(self, audio: np.ndarray, fs: int) -> Optional[float]:
        try:
            silencedb = -25
            mindip = 2
            minpause = 0.3

            sound = parselmouth.Sound(audio, sampling_frequency=fs)

            originaldur = sound.get_total_duration()
            if originaldur == 0:
                return 0.0

            intensity = sound.to_intensity(50)
            min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
            max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

            max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

            threshold = max_99_intensity + silencedb
            threshold2 = max_intensity - max_99_intensity
            threshold3 = silencedb - threshold2
            if threshold < min_intensity:
                threshold = min_intensity

            textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")

            intensity_matrix = call(intensity, "Down to Matrix")
            sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)

            point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
            numpeaks = call(point_process, "Get number of points")
            t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

            timepeaks = []
            peakcount = 0
            intensities = []
            for i in range(numpeaks):
                value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
                if value > threshold:
                    peakcount += 1
                    intensities.append(value)
                    timepeaks.append(t[i])

            if peakcount < 2:
                return 0.0

            validpeakcount = 0
            currenttime = timepeaks[0]
            currentint = intensities[0]
            validtime = []

            for p in range(peakcount - 1):
                following = p + 1
                followingtime = timepeaks[p + 1]
                dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
                diffint = abs(currentint - dip)
                if diffint > mindip:
                    validpeakcount += 1
                    validtime.append(timepeaks[p])
                currenttime = timepeaks[following]
                currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

            pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
            voicedcount = 0
            for time_index in range(validpeakcount):
                querytime = validtime[time_index]
                whichinterval = call(textgrid, "Get interval at time", 1, querytime)
                whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
                value = pitch.get_value_at_time(querytime)
                if not math.isnan(value):
                    if whichlabel == "sounding":
                        voicedcount += 1

            speakingrate = voicedcount / originaldur
            return speakingrate
        except Exception as e:
            # Parselmouth can raise a generic "PraatError"
            print(f"Error processing audio with PraatSpeechRateEvaluator: {e}")
            return None
