__version__ = "0.1.0"

# Abstract base classes and trimming wrappers
from .evaluator import (
    LookupEvaluator,
    ReferenceFreeEvaluator,
    ReferenceTxtEvaluator,
    ReferenceAudioEvaluator,
    ReferenceTxtAndAudioEvaluator,
    ReferenceFreeSpeakerEvaluator,
    LanguageAwareSpeakerEvaluator,
    TrimmedReferenceFreeEvaluator,
    TrimmedReferenceFreeSpeakerEvaluator,
    TrimmedLanguageAwareSpeakerEvaluator,
    Utt2ScoreEvaluator,
    Spk2ScoreEvaluator,
)

# Dataset loader
from .dataset import Dataset

# Forced-alignment trimmer
from .vad import FATrimmer

# Concrete evaluators
from .cpp_evaluator import CPPEvaluator, CPPDoubleLogEvaluator, PraatCPPEvaluator
from .f0_range_evaluator import StdPitchEvaluator, F0RangeEvaluator
from .wada_snr import WadaSnrEvaluator
from .speech_rate import WpmEvaluator, PraatSpeechRateEvaluator
from .vsa_evaluator import VSAEvaluator
from .nad_evaluator import NADEvaluator, TrimmedNADEvaluator
from .reference_evaluator import PSTOIEvaluator, ESTOIEvaluator
from .p_estoi_evaluator import ForcedAlignmentPESTOIEvaluator
from .asr_evaluators import ASREvaluator, PEREvaluator, DirectPEREvaluator, DoubleASREvaluator
from .articulatory_precision_evaluator import PhoneticConfidenceEvaluator, ArticulatoryPrecisionEvaluator
from .artp_double_asr_evaluator import ArtPDoubleASREvaluator
from .age_evaluator import Spk2AgeEvaluator
