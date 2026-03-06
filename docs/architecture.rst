Architecture
============

Evaluator Hierarchy
-------------------

All evaluators live in :mod:`pathbench.evaluator` as abstract base classes. The
ABC a class inherits from defines exactly what inputs ``score()`` receives.
When adding a new evaluator, pick the right ABC before writing any logic.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - ABC
     - ``score()`` signature
     - Use for
   * - :class:`~pathbench.evaluator.LookupEvaluator`
     - ``(utt_id)``
     - Pre-computed scores
   * - :class:`~pathbench.evaluator.ReferenceFreeEvaluator`
     - ``(utt_id, audio_path, start, end)``
     - Audio-only metrics (CPP, SNR, ...)
   * - :class:`~pathbench.evaluator.ReferenceTxtEvaluator`
     - ``+ transcription, language``
     - ASR/FA-based metrics
   * - :class:`~pathbench.evaluator.ReferenceAudioEvaluator`
     - ``+ reference_audios``
     - Reference comparison (NAD, ESTOI, ...)
   * - :class:`~pathbench.evaluator.ReferenceTxtAndAudioEvaluator`
     - ``+ transcription, language, reference_audios``
     - FA-trimmed reference metrics
   * - :class:`~pathbench.evaluator.ReferenceFreeSpeakerEvaluator`
     - ``_score_audio_list(audios)``
     - Speaker-level aggregation
   * - :class:`~pathbench.evaluator.LanguageAwareSpeakerEvaluator`
     - ``_score_audio_list(audios, language)``
     - Speaker-level + language (VSA)

FA-Trimming: Decorator Pattern
-------------------------------

Forced-alignment silence trimming is never baked into evaluators. Instead,
wrappers in ``evaluator.py`` handle it:

- :class:`~pathbench.evaluator.TrimmedReferenceFreeEvaluator` -- wraps any
  ``ReferenceFreeEvaluator``, presents a ``ReferenceTxtEvaluator`` interface
- :class:`~pathbench.evaluator.TrimmedReferenceFreeSpeakerEvaluator` --
  speaker-level equivalent
- :class:`~pathbench.evaluator.TrimmedLanguageAwareSpeakerEvaluator` --
  language-aware speaker-level equivalent

The trimmer is :class:`~pathbench.vad.FATrimmer` in ``pathbench/vad.py``. If
trimming fails or a segment offset is specified, it falls back to plain
``librosa.load()``.

:class:`~pathbench.nad_evaluator.TrimmedNADEvaluator` is an exception -- it
implements its own two-pass trimming logic directly, because the fallback must
be group-consistent (all references fall back together).

Dataset Format
--------------

Each dataset directory uses Kaldi-style plain text files:

- ``wav.scp`` -- ``utt_id -> audio_file_path``
- ``text`` -- ``utt_id -> transcription``
- ``utt2spk`` -- ``utt_id -> speaker_id``
- ``segments`` -- ``utt_id -> recording_id start_time end_time`` (optional)
- ``spk2score`` -- ``speaker_id -> float`` (ground truth; ``N/A`` for unavailable)
- ``spk2gender`` -- ``speaker_id -> m|f``
- ``language`` -- single line, two-letter code (``en``, ``nl``, ``it``, ``es``)

:class:`~pathbench.dataset.Dataset` loads these and iterates as
``(utt_id, audio_path, transcription, ref_audio_list, start_time, end_time)``.
Reference audio is matched by shared transcription text and, optionally, gender.
