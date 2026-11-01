__version__ = "0.1.0"

# Abstract base classes and trimming wrappers
from .core.evaluator import (
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
from .data import Dataset

# Forced-alignment trimmer
from .vad import FATrimmer

# Concrete evaluators
from .evaluators.cpp import CPPEvaluator, CPPDoubleLogEvaluator, PraatCPPEvaluator
from .evaluators.f0_range import StdPitchEvaluator, F0RangeEvaluator
from .evaluators.wada_snr import WadaSnrEvaluator
from .evaluators.speech_rate import WpmEvaluator, PraatSpeechRateEvaluator
from .evaluators.vsa import VSAEvaluator
from .evaluators.nad import NADEvaluator, TrimmedNADEvaluator
from .evaluators.stoi import PSTOIEvaluator, ESTOIEvaluator
from .evaluators.p_estoi import ForcedAlignmentPESTOIEvaluator
from .asr.asr import ASREvaluator, PEREvaluator, DirectPEREvaluator, DoubleASREvaluator
from .asr.articulatory_precision import PhoneticConfidenceEvaluator, ArticulatoryPrecisionEvaluator
from .asr.artp_double_asr import ArtPDoubleASREvaluator
from .evaluators.age import Spk2AgeEvaluator
