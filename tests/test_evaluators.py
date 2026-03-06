"""
Unit tests for PathBench evaluators.

# Running the tests
    source tools/venv/bin/activate
    python -m pytest tests/test_evaluators.py -v

# Bootstrapping expected values
If you change the reference audio file or update an evaluator implementation,
set the corresponding EXPECTED_* constant to None. The test will print the
actual computed value and skip the assertion so you can paste it back in.

# Verifying dataset integrity
To check that your dataset files match a known reference, fill in the
EXPECTED_DATASET_HASHES dict and run:
    python -m pytest tests/test_evaluators.py::TestDatasetIntegrity -v

To compute hashes for a dataset directory:
    python tests/test_evaluators.py --hash /path/to/dataset
"""

import argparse
import hashlib
import os
import sys
import unittest

# ---------------------------------------------------------------------------
# Synthetic reference audio — committed to the repo, used only for the
# audio-integrity check. Evaluator score tests use the real BLUE recordings
# below instead.
# ---------------------------------------------------------------------------
TEST_AUDIO = os.path.join(os.path.dirname(__file__), "data", "test_audio.wav")
EXPECTED_AUDIO_SHA256 = "c2f6cdbe8a590fa3ea6b767c94eca05e14382b96ceb9dba5b651a26d96771fa4"

# ---------------------------------------------------------------------------
# Real audio files — single-word ("BLUE") recordings from the
# neural-acoustic-distance corpus (public domain, CC0).
#   Accented speaker  : japanese10 — used as the test/pathological utterance.
#   Typical speakers  : english32/33/34 — used as control references.
# ---------------------------------------------------------------------------
_DATA = os.path.join(os.path.dirname(__file__), "data")
BLUE_ACCENTED  = os.path.join(_DATA, "BLUE_japanese10.wav")   # 0.210 s, 16 kHz mono
BLUE_CONTROL_1 = os.path.join(_DATA, "BLUE_english32.wav")    # 0.270 s
BLUE_CONTROL_2 = os.path.join(_DATA, "BLUE_english33.wav")    # 0.240 s
BLUE_CONTROL_3 = os.path.join(_DATA, "BLUE_english34.wav")    # 0.180 s
BLUE_CONTROLS  = [(BLUE_CONTROL_1, 0.0, -1.0),
                  (BLUE_CONTROL_2, 0.0, -1.0),
                  (BLUE_CONTROL_3, 0.0, -1.0)]

EXPECTED_BLUE_ACCENTED_SHA256  = "2f67bec80deee5757978763f603430eec6d24aa461b10026d7c0a4ddc9cafa6f"
EXPECTED_BLUE_CONTROL_1_SHA256 = "3c23c4a3947b2d0546103355dbb4d8fc34306ae6f4d868aa1c5a22dfe9ac7e3b"
EXPECTED_BLUE_CONTROL_2_SHA256 = "a15ad6aa18a4d0a609726ad80e8cb9602eb6d608635ab24fd93acf9d57ca6b85"
EXPECTED_BLUE_CONTROL_3_SHA256 = "6aefd9e4d80ffc3647e9f0ed5904e40d6961dafbd7560365b3a7d90a1992407e"

# ---------------------------------------------------------------------------
# Expected evaluator scores on the real BLUE recordings.
# Set any value to None to enter bootstrap mode: the test prints the actual
# value and skips the assertion so you can paste the number in.
# ---------------------------------------------------------------------------
EXPECTED_CPP            = 732.418
EXPECTED_CPP_DOUBLE_LOG = 18.399
EXPECTED_CPP_PRAAT      = 9.836
EXPECTED_STD_PITCH      = 0.1828   # places=4
EXPECTED_F0_RANGE       = 152.34   # places=2
EXPECTED_WADA_SNR       = 19.59    # places=2
EXPECTED_PRAAT_SPEECH_RATE = 0.0   # word too short for syllable-nuclei detection
EXPECTED_WPM            = 285.7    # places=1; deterministic: 1 word / 0.210 s * 60
EXPECTED_PSTOI          = 0.715    # places=3
EXPECTED_ESTOI          = 0.0016   # places=4
EXPECTED_NAD                        = 13.7288  # places=4; facebook/wav2vec2-large
EXPECTED_NAD_TRIMMED                = 13.7288  # places=4; TrimmedNADEvaluator (no trimmer → untrimmed fallback)
EXPECTED_ARTICULATORY_PRECISION_OLD = 0.5374   # places=4; facebook/wav2vec2-xlsr-53-espeak-cv-ft, confidence measure
EXPECTED_ARTICULATORY_PRECISION     = 0.0695   # places=4; forced-alignment AP
EXPECTED_ARTP_DOUBLE_ASR            = 0.4405   # places=4; double-pass ASR + phonetic model + LM
EXPECTED_FA_PESTOI                  = -0.0761  # places=4; forced-alignment P-ESTOI


def file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Evaluator method tests
# ---------------------------------------------------------------------------

class TestEvaluatorMethods(unittest.TestCase):
    """
    Tests each evaluator on real BLUE word recordings from the
    neural-acoustic-distance corpus (CC0).
      - Accented (test)   : japanese10/BLUE_japanese10.wav
      - Typical (controls): english32/33/34 BLUE_english{32,33,34}.wav

    If all score tests fail, run test_audio_integrity first — a hash mismatch
    means the committed files changed and all EXPECTED_* constants need
    updating (set to None to re-bootstrap).

    Heavy-model evaluators (NAD, articulatory precision, ForcedAlignment P-ESTOI)
    were previously skipped. All are now active.

    Phoneme-model tests require the espeak-ng shared library, set via:
        PHONEMIZER_ESPEAK_LIBRARY=.../libespeak-ng.so pytest tests/test_evaluators.py
    """

    @classmethod
    def setUpClass(cls):
        missing = [p for p in [TEST_AUDIO, BLUE_ACCENTED,
                                BLUE_CONTROL_1, BLUE_CONTROL_2, BLUE_CONTROL_3]
                   if not os.path.exists(p)]
        if missing:
            raise unittest.SkipTest(
                "Audio files not found:\n" + "\n".join(f"  {p}" for p in missing)
            )

    def test_audio_integrity(self):
        """Verify all committed audio files match their SHA256 hashes."""
        checks = [
            (TEST_AUDIO,      EXPECTED_AUDIO_SHA256,        "synthetic test_audio"),
            (BLUE_ACCENTED,   EXPECTED_BLUE_ACCENTED_SHA256, "BLUE_japanese10"),
            (BLUE_CONTROL_1,  EXPECTED_BLUE_CONTROL_1_SHA256, "BLUE_english32"),
            (BLUE_CONTROL_2,  EXPECTED_BLUE_CONTROL_2_SHA256, "BLUE_english33"),
            (BLUE_CONTROL_3,  EXPECTED_BLUE_CONTROL_3_SHA256, "BLUE_english34"),
        ]
        for path, expected_hash, label in checks:
            with self.subTest(file=label):
                actual = file_sha256(path)
                self.assertEqual(
                    actual, expected_hash,
                    f"{label} hash mismatch.\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual}",
                )

    def _assert_score(self, name, score, expected, places=3):
        self.assertIsNotNone(score, f"{name} returned None")
        if expected is None:
            print(f"\n[bootstrap] {name} = {score!r}  ← paste into EXPECTED_{name.upper().replace(' ', '_')}")
            return
        self.assertAlmostEqual(
            score, expected, places=places,
            msg=f"{name} score {score} differs from expected {expected}",
        )

    # ------------------------------------------------------------------
    # Reference-free evaluators  (accented speaker as test utterance)
    # ------------------------------------------------------------------

    def test_cpp(self):
        """CPP (standard formulation) on accented BLUE recording."""
        from pathbench.cpp_evaluator import CPPEvaluator
        score = CPPEvaluator().score("test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0)
        self._assert_score("CPP", score, EXPECTED_CPP)

    def test_cpp_double_log(self):
        """CPP legacy double-log formulation on accented BLUE recording."""
        from pathbench.cpp_evaluator import CPPDoubleLogEvaluator
        score = CPPDoubleLogEvaluator().score("test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0)
        self._assert_score("CPP_DoubleLog", score, EXPECTED_CPP_DOUBLE_LOG)

    def test_cpp_praat(self):
        """CPP Praat reference implementation on accented BLUE recording."""
        from pathbench.cpp_evaluator import PraatCPPEvaluator
        score = PraatCPPEvaluator().score("test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0)
        self._assert_score("CPP_Praat", score, EXPECTED_CPP_PRAAT)

    def test_std_pitch(self):
        """Standard deviation of pitch (semitones) on accented BLUE recording."""
        from pathbench.f0_range_evaluator import StdPitchEvaluator
        score = StdPitchEvaluator().score("test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0)
        self._assert_score("StdPitch", score, EXPECTED_STD_PITCH, places=4)

    def test_wada_snr(self):
        """WADA SNR (blind SNR estimate) on accented BLUE recording."""
        from pathbench.wada_snr import WadaSnrEvaluator
        score = WadaSnrEvaluator().score("test", BLUE_ACCENTED)
        self._assert_score("WadaSNR", score, EXPECTED_WADA_SNR, places=2)

    def test_praat_speech_rate(self):
        """Praat syllable-nuclei speech rate (syll/s) on accented BLUE recording."""
        from pathbench.speech_rate import PraatSpeechRateEvaluator
        score = PraatSpeechRateEvaluator().score("test", BLUE_ACCENTED)
        self._assert_score("PraatSpeechRate", score, EXPECTED_PRAAT_SPEECH_RATE, places=3)

    def test_wpm(self):
        """Words-per-minute on accented BLUE recording with transcription 'blue'."""
        from pathbench.speech_rate import WpmEvaluator
        score = WpmEvaluator().score("test", BLUE_ACCENTED, "blue", "en")
        self._assert_score("WPM", score, EXPECTED_WPM, places=1)

    # ------------------------------------------------------------------
    # Reference-free speaker-level evaluators  (all 4 BLUE recordings)
    # ------------------------------------------------------------------

    def test_f0_range(self):
        """F0 range (Hz) across all four BLUE speakers."""
        from pathbench.f0_range_evaluator import F0RangeEvaluator
        all_files = [(BLUE_ACCENTED, 0.0, -1.0)] + list(BLUE_CONTROLS)
        score = F0RangeEvaluator().score(all_files)
        self._assert_score("F0Range", score, EXPECTED_F0_RANGE, places=1)

    def test_vsa(self):
        """VSA (Vowel Space Area) across all four BLUE speakers."""
        import librosa
        from pathbench.vsa_evaluator import VSAEvaluator
        all_files = [(BLUE_ACCENTED, 0.0, -1.0)] + list(BLUE_CONTROLS)
        audios = []
        for path, start, end in all_files:
            duration = end - start if end != -1.0 else None
            audio, fs = librosa.load(path, sr=16000, offset=start, duration=duration)
            audios.append((audio, fs))
        score = VSAEvaluator()._score_audio_list(audios, "en")
        # Just verify it runs without error and returns a numeric value or None.
        self.assertTrue(
            score is None or isinstance(score, (int, float)),
            f"VSA returned unexpected type: {type(score)}",
        )
        if score is not None:
            print(f"\n[info] VSA = {score!r} on BLUE recordings")

    # ------------------------------------------------------------------
    # Reference-audio evaluators  (accented as test, controls as reference)
    # ------------------------------------------------------------------

    def test_pstoi(self):
        """PSTOI: accented BLUE vs three typical-English BLUE controls."""
        from pathbench.reference_evaluator import PSTOIEvaluator
        score = PSTOIEvaluator(
            normalization_method='RMS', centroid_ind=0, frame_deletion=True
        ).score("test", BLUE_ACCENTED, BLUE_CONTROLS)
        self._assert_score("PSTOI", score, EXPECTED_PSTOI, places=3)

    def test_estoi(self):
        """ESTOI: accented BLUE vs three typical-English BLUE controls."""
        from pathbench.reference_evaluator import ESTOIEvaluator
        score = ESTOIEvaluator(
            normalization_method='RMS', centroid_ind=0, frame_deletion=True
        ).score("test", BLUE_ACCENTED, BLUE_CONTROLS)
        self._assert_score("ESTOI", score, EXPECTED_ESTOI, places=4)

    # ------------------------------------------------------------------
    # Lookup evaluator
    # ------------------------------------------------------------------

    def test_spk2age(self):
        """Spk2AgeEvaluator looks up age from a pre-computed dict (no audio processing)."""
        from pathbench.age_evaluator import Spk2AgeEvaluator
        evaluator = Spk2AgeEvaluator(spk2age={"spk1": 45.0}, utt2spk={"test": "spk1"})
        score = evaluator.score("test")
        self.assertEqual(score, 45.0)

    # ------------------------------------------------------------------
    # Skipped: require model downloads
    # ------------------------------------------------------------------

    def test_nad(self):
        """NAD: accented BLUE vs three typical-English BLUE controls."""
        from pathbench.nad_evaluator import NADEvaluator
        score = NADEvaluator().score(
            "test", BLUE_ACCENTED, BLUE_CONTROLS, start_time=0.0, end_time=-1.0
        )
        self._assert_score("NAD", score, EXPECTED_NAD, places=4)

    def test_nad_trimmed(self):
        """TrimmedNAD: accented BLUE vs three typical-English BLUE controls (no trimmer → untrimmed fallback)."""
        from pathbench.nad_evaluator import TrimmedNADEvaluator
        score = TrimmedNADEvaluator().score(
            "test", BLUE_ACCENTED, "blue", "en", BLUE_CONTROLS
        )
        self._assert_score("TrimmedNAD", score, EXPECTED_NAD_TRIMMED, places=4)

    def test_articulatory_precision_old(self):
        """ArticulatoryPrecisionEvaluatorOld — confidence measure (avg phoneme probability
        from argmax decoding, without forced alignment) on accented BLUE recording.
        """
        from pathbench.articulatory_precision_evaluator import PhoneticConfidenceEvaluator
        score = PhoneticConfidenceEvaluator().score("test", BLUE_ACCENTED)
        self._assert_score("ArticulatoryPrecisionOld", score, EXPECTED_ARTICULATORY_PRECISION_OLD, places=4)

    def test_articulatory_precision(self):
        """ArticulatoryPrecisionEvaluator — forced-alignment articulatory precision
        (avg alignment score per phoneme) on accented BLUE recording.
        """
        from pathbench.articulatory_precision_evaluator import ArticulatoryPrecisionEvaluator
        score = ArticulatoryPrecisionEvaluator().score("test", BLUE_ACCENTED, "blue", "en-us")
        self._assert_score("ArticulatoryPrecision", score, EXPECTED_ARTICULATORY_PRECISION, places=4)

    @unittest.skipUnless(
        os.path.exists("lms/wiki_en_token.arpa.bin") or os.path.exists("lms/wiki_en_token.arpa"),
        "Language model not available"
    )
    def test_artp_double_asr(self):
        """ArtPDoubleASREvaluator (articulatory precision via double-pass ASR) on accented BLUE recording."""
        from pathbench.artp_double_asr_evaluator import ArtPDoubleASREvaluator
        score = ArtPDoubleASREvaluator(language="en-us").score(
            "test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0
        )
        self._assert_score("ArtPDoubleASR", score, EXPECTED_ARTP_DOUBLE_ASR, places=4)

    def test_fa_pestoi(self):
        """ForcedAlignmentPESTOI: accented BLUE vs three typical-English BLUE controls."""
        from pathbench.p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
        score = ForcedAlignmentPESTOIEvaluator().score(
            utterance_id="test",
            audio_path=BLUE_ACCENTED,
            transcription="blue",
            language="en-us",
            reference_audios=BLUE_CONTROLS,
            start_time=0.0,
            end_time=-1.0,
        )
        self._assert_score("ForcedAlignmentPESTOI", score, EXPECTED_FA_PESTOI, places=4)


# ---------------------------------------------------------------------------
# Dataset integrity tests
# ---------------------------------------------------------------------------

# Kaldi-style files to hash when verifying a dataset.
DATASET_FILES = ["wav.scp", "text", "utt2spk", "spk2score", "utt2score"]

# Fill in expected hashes for a specific dataset to enable integrity checks.
# Run `python tests/test_evaluators.py --hash /path/to/dataset` to compute them.
#
# Example:
#   EXPECTED_DATASET_HASHES = {
#       "wav.scp":    "abc123...",
#       "spk2score":  "def456...",
#   }
EXPECTED_DATASET_DIR    = None   # e.g. "datasets/copas/pathological/word/balanced"
EXPECTED_DATASET_HASHES = {}     # fill in after running --hash


def compute_dataset_hashes(dataset_dir: str) -> dict:
    """Return a dict of filename → SHA256 for key Kaldi dataset files."""
    hashes = {}
    for fname in DATASET_FILES:
        path = os.path.join(dataset_dir, fname)
        if os.path.exists(path):
            hashes[fname] = file_sha256(path)
    return hashes


class TestDatasetIntegrity(unittest.TestCase):
    """
    Verifies that dataset files match known SHA256 hashes.

    To use:
      1. Run: python tests/test_evaluators.py --hash /path/to/dataset
      2. Paste the printed hashes into EXPECTED_DATASET_HASHES above
      3. Set EXPECTED_DATASET_DIR to the same path
      4. Run: python -m pytest tests/test_evaluators.py::TestDatasetIntegrity -v
    """

    @unittest.skipUnless(
        EXPECTED_DATASET_DIR and EXPECTED_DATASET_HASHES,
        "Set EXPECTED_DATASET_DIR and EXPECTED_DATASET_HASHES to enable dataset integrity checks",
    )
    def test_dataset_files(self):
        actual = compute_dataset_hashes(EXPECTED_DATASET_DIR)
        for fname, expected_hash in EXPECTED_DATASET_HASHES.items():
            with self.subTest(file=fname):
                self.assertIn(
                    fname, actual,
                    f"{fname} not found in {EXPECTED_DATASET_DIR}",
                )
                self.assertEqual(
                    actual[fname], expected_hash,
                    f"{fname} hash mismatch — dataset may differ from reference.\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual[fname]}",
                )


def _discover_dataset_dirs():
    """Find all dataset directories that contain wav_hash.scp under datasets/."""
    datasets_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    result = []
    if not os.path.isdir(datasets_root):
        return result
    for dirpath, _, filenames in os.walk(datasets_root):
        if "wav_hash.scp" in filenames:
            rel = os.path.relpath(dirpath, datasets_root)
            result.append((rel, dirpath))
    result.sort()
    return result


def _check_audio_hashes_for_dataset(test_case, dataset_dir, rel):
    """Verify audio files in a single dataset match wav_hash.scp."""
    # Load wav.scp: utt_id -> audio_path
    wav_scp = {}
    with open(os.path.join(dataset_dir, "wav.scp")) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                wav_scp[parts[0]] = parts[1]

    # Load wav_hash.scp: utt_id -> expected_hash
    expected = {}
    with open(os.path.join(dataset_dir, "wav_hash.scp")) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[parts[0]] = parts[1]

    for utt_id, expected_hash in expected.items():
        with test_case.subTest(utt_id=utt_id):
            test_case.assertIn(utt_id, wav_scp,
                               f"{utt_id} in wav_hash.scp but not in wav.scp")
            audio_path = wav_scp[utt_id]
            test_case.assertTrue(os.path.isfile(audio_path),
                                 f"Audio file missing: {audio_path}")
            actual_hash = file_sha256(audio_path)
            test_case.assertEqual(
                actual_hash, expected_hash,
                f"Audio hash mismatch for {utt_id} in {rel}\n"
                f"  File:     {audio_path}\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}",
            )


class TestAudioFileHashes(unittest.TestCase):
    """Per-dataset audio file integrity tests.

    Each dataset directory with a wav_hash.scp gets its own test method,
    making it easy to spot which dataset has bad files:

        # Run all dataset hash checks
        python -m pytest tests/test_evaluators.py::TestAudioFileHashes -v

        # Run only a specific dataset (use -k with any part of the path)
        python -m pytest tests/test_evaluators.py::TestAudioFileHashes -k copas_pathological_word_balanced
        python -m pytest tests/test_evaluators.py::TestAudioFileHashes -k torgo
        python -m pytest tests/test_evaluators.py::TestAudioFileHashes -k easycall

    Generate wav_hash.scp files with:
        python scripts/generate_wav_hashes.py
    """
    pass


# Dynamically generate one test method per dataset directory.
for _rel, _abs_dir in _discover_dataset_dirs():
    # Convert path separators to underscores for a valid method name.
    _test_name = "test_" + _rel.replace(os.sep, "_").replace("-", "_").replace(".", "_")

    def _make_test(dataset_dir, rel):
        def test_method(self):
            _check_audio_hashes_for_dataset(self, dataset_dir, rel)
        test_method.__doc__ = f"Verify audio hashes for {rel}"
        return test_method

    setattr(TestAudioFileHashes, _test_name, _make_test(_abs_dir, _rel))


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _generate_audio():
    """Generate the reference synthetic audio file."""
    import numpy as np
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Run: pip install soundfile")
        sys.exit(1)

    os.makedirs(os.path.dirname(TEST_AUDIO), exist_ok=True)
    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    audio = sum(np.sin(2 * np.pi * k * 110 * t) / k for k in range(1, 8))
    audio = (audio / np.max(np.abs(audio)) * 0.5).astype(np.float32)
    sf.write(TEST_AUDIO, audio, sr)
    sha = file_sha256(TEST_AUDIO)
    print(f"Written: {TEST_AUDIO}")
    print(f"SHA256:  {sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathBench evaluator unit tests")
    parser.add_argument("--hash", metavar="DATASET_DIR",
                        help="Compute and print SHA256 hashes of dataset files")
    parser.add_argument("--generate", action="store_true",
                        help="Generate the reference synthetic audio file")
    args, remaining = parser.parse_known_args()

    if args.generate:
        _generate_audio()
        sys.exit(0)

    if args.hash:
        hashes = compute_dataset_hashes(args.hash)
        if not hashes:
            print(f"No dataset files found in: {args.hash}")
            sys.exit(1)
        print(f"Dataset: {args.hash}")
        for fname, h in hashes.items():
            print(f'    "{fname}": "{h}",')
        sys.exit(0)

    # Run tests normally
    unittest.main(argv=[sys.argv[0]] + remaining)
