from typing import List, Optional, Tuple
import os
import matplotlib.pyplot as plt

import numpy as np
import parselmouth
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from pathbench.evaluator import LanguageAwareSpeakerEvaluator


class VSAEvaluator(LanguageAwareSpeakerEvaluator):
    """
    An evaluator that computes the Vowel Space Area (VSA) for a speaker.

    The algorithm is based on:
    -

    The vowel formant data for initialization is from:
    - English: Hillenbrand, J., Getty, L. A., Clark, M. J., & Wheeler, K. (1995).
      Acoustic characteristics of American English vowels. JASA, 97(5), 3099-3111.
    - Dutch: Adank et al. We use souther Dutch formant values due to Belgian context.

    - Italian: Bertinetto, P. M. (?) "The sound pattern of Standard Italian, as compared
    with the varieties spoken in Florence, Milan and Rome."
    - Spanish: Bradlow: A comparative acoustic study of English and Spanish vowels

    """

    def __init__(self, gender: Optional[str] = None):
        """
        Initializes the VSA evaluator.
        Args:
            gender: The gender of the speaker ('m' or 'f'). If not provided, it will be
                    estimated from the pitch at score time.
        """
        self.gender = gender

        # English (Hillenbrand 1995) - 12 vowels (Bence: I haven't checked this data myself)
        hillenbrand_female = np.array([
            [437, 2779], [533, 2479], [543, 2320], [691, 2056],
            [855, 1845], [912, 1448], [707, 1219], [537, 1069],
            [524, 1331], [463, 1180], [759, 1403], [548, 1650]
        ])
        hillenbrand_male = np.array([
            [370, 2299], [468, 2049], [479, 2019], [610, 1794],
            [738, 1661], [793, 1299], [624, 1089], [471, 939],
            [462, 1179], [395, 1029], [663, 1239], [490, 1469]
        ])

        # Italian - 7 vowels
        italian_male = np.array([
            [290,2310], [350, 2050], [490, 1950], [780, 1430], [550, 970],
            [390, 870], [320, 800]])
        italian_female = italian_male

        # Spanish (Bradlow) - 5 vowels
        spanish_female = np.array([
            [286, 2147], [458, 1814], [638, 1353], [460, 1019], [322, 992]
        ])
        spanish_male = np.array([
            [290, 2250], [450, 1850], [700, 1300], [450, 950], [300, 850]
        ])

        # Dutch (Adank et al.) - 15 vowels
        dutch_female = np.array([
            [725, 1262], [868, 1640], [581, 1932], [436, 2420], [439, 1804],
            [455, 2115], [317, 2647], [475, 987], [321, 1019], [418, 968],
            [457, 1785], [337, 2077], [696, 1282], [670, 2159], [696, 1762]
        ])
        dutch_male = np.array([
            [555, 1066], [717, 1429], [475, 1616], [384, 1993], [374, 1539],
            [364, 1745], [278, 2179], [398, 850], [266, 978], [369, 862],
            [353, 1492], [265, 1825], [549, 1127], [545, 1779], [583, 1484]
        ])

        self._vowel_tables = {
            'en': {'n_clusters': 12, 'female': hillenbrand_female, 'male': hillenbrand_male},
            'it': {'n_clusters': 7,  'female': italian_female,     'male': italian_male},
            'es': {'n_clusters': 5,  'female': spanish_female,     'male': spanish_male},
            'nl': {'n_clusters': 15, 'female': dutch_female,       'male': dutch_male},
        }

    def _score_audio_list(
        self,
        audios: List[Tuple[np.ndarray, int]],
        language: str,
        speaker_id: str = "unknown_speaker",
    ) -> Optional[float]:
        # Dataset.language uses phonemiser locale codes (e.g. 'en-us'); strip to base code.
        language = language.split('-')[0]
        tables = self._vowel_tables[language]
        n_clusters = tables['n_clusters']

        all_formants = []

        gender = self.gender
        if gender is None:
            pitches = []
            for speech, sample_rate in audios:
                try:
                    sound = parselmouth.Sound(speech, sampling_frequency=sample_rate)
                    pitch = sound.to_pitch()
                    pitch_values = pitch.selected_array['frequency']
                    voiced_pitches = pitch_values[pitch_values > 0]
                    if len(voiced_pitches) > 0:
                        pitches.append(np.mean(voiced_pitches))
                except Exception as e:
                    print(f"Could not process audio for pitch estimation: {e}")

            if pitches:
                avg_pitch = np.mean(pitches)
                gender = 'f' if avg_pitch > 165 else 'm'
                print(f"Estimated gender as '{gender}' with average pitch {avg_pitch:.2f} Hz.")
            else:
                gender = 'f'  # Default to female if pitch cannot be determined
                print("Could not determine pitch, defaulting to female.")

        max_formant = 5500 if gender == 'f' else 5000
        init_clusters = tables['female'] if gender == 'f' else tables['male']

        for speech, sample_rate in audios:
            try:
                sound = parselmouth.Sound(speech, sampling_frequency=sample_rate)

                pitch = sound.to_pitch()
                formants = sound.to_formant_burg(
                    time_step=0.01, max_number_of_formants=5,
                    maximum_formant=max_formant, window_length=0.05,
                    pre_emphasis_from=50
                )

                num_frames = pitch.get_number_of_frames()
                for i in range(num_frames):
                    frame_time = pitch.get_time_from_frame_number(i + 1)
                    if pitch.get_value_in_frame(i + 1) > 0:  # Voiced frame
                        f1 = formants.get_value_at_time(1, frame_time)
                        f2 = formants.get_value_at_time(2, frame_time)
                        if not np.isnan(f1) and not np.isnan(f2):
                            all_formants.append([f1, f2])
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue

        if not all_formants:
            print("Warning: No formants extracted. Cannot calculate VSA.")
            return 0.0

        Fp = np.array(all_formants)

        if Fp.shape[0] < 4:  # n_components for GMM
            print("Warning: Not enough formant points to build GMM. Cannot calculate VSA.")
            return 0.0

        # B. Filtering
        try:
            gmm = GaussianMixture(n_components=min(4, Fp.shape[0]), random_state=0).fit(Fp)
            log_likelihood = gmm.score_samples(Fp)
            mean_ll = np.mean(log_likelihood)
            std_ll = np.std(log_likelihood)
            threshold = mean_ll - 2 * std_ll

            Fp_filtered = Fp[log_likelihood >= threshold]
        except Exception as e:
            print(f"Error during GMM filtering: {e}")
            return 0.0

        if len(Fp_filtered) < n_clusters:
            print(
                f"Warning: Not enough data points ({len(Fp_filtered)}) after filtering "
                f"for clustering. Cannot calculate VSA."
            )
            return 0.0

        # C. Clustering
        try:
            kmeans = KMeans(
                n_clusters=n_clusters, init=init_clusters, n_init=1, random_state=0
            ).fit(Fp_filtered)
            Kp = kmeans.cluster_centers_
        except Exception as e:
            print(f"Error during KMeans clustering: {e}")
            return 0.0

        # D. Convex hull/area calculation
        try:
            hull = ConvexHull(Kp)
            # Known naming trap in scipy: in 2D, .volume is the enclosed area and
            # .area is the perimeter. hull.volume is intentionally correct here.
            vsa = hull.volume
        except Exception as e:
            print(f"Error calculating convex hull: {e}")
            return 0.0

        # Visualization
        fig_dir = "figure"
        os.makedirs(fig_dir, exist_ok=True)

        plt.figure(figsize=(8, 8))

        # Plot filtered formants
        plt.scatter(Fp_filtered[:, 1], Fp_filtered[:, 0], alpha=0.2, label="Formants")

        # Plot cluster centers
        plt.scatter(Kp[:, 1], Kp[:, 0], marker='x', s=100, c='r', label="Cluster Centers")

        # Plot convex hull
        for simplex in hull.simplices:
            plt.plot(Kp[simplex, 1], Kp[simplex, 0], 'g-')

        plt.xlabel("F2 (Hz)")
        plt.ylabel("F1 (Hz)")
        plt.title(f"Vowel Space Area for Speaker {speaker_id} (VSA: {vsa:.2f})")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlim(3000, 700)
        plt.ylim(1200, 200)
        plt.grid(True)

        fig_path = os.path.join(fig_dir, f"{speaker_id}_vsa.png")
        plt.savefig(fig_path)
        plt.close()

        return vsa
