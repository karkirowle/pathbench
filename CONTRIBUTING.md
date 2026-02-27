# Contributing to PathBench

There are two main ways to contribute: **fixing a bug** and **developing a new evaluator or feature**.

## Setup

```bash
cd tools && make          # Creates tools/venv/, installs PyTorch + dependencies
source tools/venv/bin/activate
```

External system dependencies required: `espeak-ng`, `Praat` (parselmouth).

## Running tests

```bash
pytest tests/test_evaluators.py -v
```

If you cannot install `espeak-ng` system-wide (e.g. no sudo access), point phonemizer at a local build:

```bash
PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so pytest tests/test_evaluators.py -v
```

---

## Fixing a bug

1. Identify and fix the root cause in the relevant module.
2. **If the fix changes any evaluator's numerical output**, update the corresponding `EXPECTED_*` constant in [tests/test_evaluators.py](tests/test_evaluators.py):
   - Set the constant to `None` to enter bootstrap mode.
   - Run the test suite — it will print the actual value instead of asserting.
   - Set the constant to the printed value.
3. Run the full test suite and confirm there are no regressions.

---

## Developing a new evaluator

### Step 0 — Check the taxonomy

Use the flow-chart below to decide where your evaluator fits. If it does not fit, open an issue to discuss.

![Taxonomy](taxonomy.png)

### Step 1 — Study the reference

If there is an accompanying paper or reference implementation, read it before writing any code and cite it in the module-level docstring of your new file.

### Step 2 — Pick the right abstract base class

All ABCs live in [pathbench/evaluator.py](pathbench/evaluator.py). The ABC you inherit from determines exactly what inputs `score()` receives.

| ABC | `score()` signature | Use for |
|-----|---------------------|---------|
| `LookupEvaluator` | `(utt_id)` | Pre-computed scores |
| `ReferenceFreeEvaluator` | `(utt_id, audio_path, start, end)` | Audio-only metrics (CPP, SNR…) |
| `ReferenceTxtEvaluator` | `+ transcription, language` | ASR/FA-based metrics |
| `ReferenceAudioEvaluator` | `+ reference_audios` | Reference comparison (NAD, ESTOI…) |
| `ReferenceTxtAndAudioEvaluator` | `+ transcription, language, reference_audios` | FA-trimmed reference metrics |
| `ReferenceFreeSpeakerEvaluator` | `_score_audio_list(audios)` | Speaker-level aggregation |
| `LanguageAwareSpeakerEvaluator` | `_score_audio_list(audios, language)` | Speaker-level + language (VSA) |

### Step 3 — Implement the evaluator

Create `pathbench/<name>_evaluator.py` and implement the method(s) required by your chosen ABC.

`ReferenceFreeEvaluator` additionally requires `_score_audio(audio: np.ndarray, fs: int)` — this private method is what FA-trimming wrappers call (see Step 4).

### Step 4 (optional) — Add FA-trimming support

If your evaluator is a `ReferenceFreeEvaluator` and you want automatic forced-alignment silence trimming, implement `_score_audio(audio, fs)`. This lets the benchmark wrap your evaluator with `TrimmedReferenceFreeEvaluator` at registration time with no changes to your class:

```python
TrimmedReferenceFreeEvaluator(inner=MyEvaluator(), trimmer=trimmer)
```

The `FATrimmer` is defined in [pathbench/vad.py](pathbench/vad.py). Fallback behaviour (when trimming fails) is handled by the wrapper — you do not need to implement it yourself.

> **Exception:** If your evaluator compares against a group of reference audios and all references must fall back together when trimming fails on any one of them, implement the fallback logic directly inside the class (see `TrimmedNADEvaluator` in [pathbench/nad_evaluator.py](pathbench/nad_evaluator.py) for the pattern).

### Step 5 — Register the evaluator

Add your evaluator to `build_evaluators()` in [scripts/evaluate_spk2score.py](scripts/evaluate_spk2score.py). Reference-based evaluators (those that need a reference audio set) should be added to `utt_evaluators` and their name added to `REF_EVALUATOR_NAMES` so the benchmark runs them against both `control` and `all` reference types.

### Step 6 — Export from the package

Add your class to the import list in [pathbench/__init__.py](pathbench/__init__.py).

### Step 7 — Add a unit test

Add a test to [tests/test_evaluators.py](tests/test_evaluators.py). See the **Unit test conventions** section below.

---

## Unit test conventions

**Audio fixtures** — use the existing CC0 BLUE-word recordings:

```python
BLUE_ACCENTED  = "tests/data/BLUE_japanese10.wav"   # test speaker (accented/pathological)
BLUE_CONTROLS  = [                                   # reference speakers
    ("tests/data/BLUE_english32.wav", 0.0, -1.0),
    ("tests/data/BLUE_english33.wav", 0.0, -1.0),
    ("tests/data/BLUE_english34.wav", 0.0, -1.0),
]
```

**Expected-value constant** — add a module-level constant for the expected score:

```python
EXPECTED_MY_METRIC = None   # set to None to bootstrap
```

**Test method** — call `_assert_score()` with the result:

```python
def test_my_metric(self):
    score = MyEvaluator().score("test", BLUE_ACCENTED, start_time=0.0, end_time=-1.0)
    self._assert_score("my_metric", score, EXPECTED_MY_METRIC, places=3)
```

**Bootstrap workflow:**

1. Leave `EXPECTED_MY_METRIC = None`.
2. Run `pytest tests/test_evaluators.py::TestEvaluatorMethods::test_my_metric -v` — the test prints the actual value and passes.
3. Copy the printed value into the constant and commit.

**Choosing `places`:** match the precision of the metric (fewer decimal places for metrics with high variance).

**Skipping gracefully:** if your test requires an optional large model or file, check for its existence and call `self.skipTest(...)` rather than letting the test fail.

---

## Coding conventions

- **Do not remove or rewrite existing comments**, even if they look like debug notes or TODOs. Only remove a comment when explicitly asked to.
- **Keep FA trimming as a decorator.** Never bake silence-trimming logic directly into a new evaluator unless you need the group-consistent fallback pattern (see Step 4 above).
- **Speaker-level evaluators must return `0.0` on failure, not `None`.** Returning `None` would exclude the speaker from the common speaker set in `compute_correlations()`, reducing the effective N for all other metrics.
