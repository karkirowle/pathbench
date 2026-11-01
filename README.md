# PathBench

<p align="center"><img src="assets/leonberger_transparent.png" width="150" /></p>

[![Unit Tests](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml/badge.svg)](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/path-benc/badge/?version=latest)](https://path-benc.readthedocs.io/en/latest/)
![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)

PathBench is a benchmark designed to evaluate pathological speech assessment systems.

## Results

Speaker-level Pearson Correlation Coefficients (PCC) results can be found below.
Currently ArtP is the best reference-based, and DArtP is the best reference-free method.

* **CSV** (canonical, in this repo): [`results/results_table.csv`](results/results_table.csv) (RTX 3090) and [`results/results_table_2080ti.csv`](results/results_table_2080ti.csv) (RTX 2080 Ti). The two are the same benchmark on different GPUs; small differences (≤0.01) are due to GPU floating-point non-determinism.
* **Google docs** [Google Docs](https://docs.google.com/spreadsheets/d/1ri-y_bHPgED3jJLuonChuwpaSP_3ddrx/edit?usp=sharing&ouid=112094007551573667400&rtpof=true&sd=true)


## Usage guide

There are several use cases for PathBench:

* [I want to evaluate my newly developed predictor](#i-want-to-evaluate-my-newly-developed-predictor)

* [I want to use the predictors developed by you](#i-want-to-use-the-predictors-developed-by-you)

* [I want to contribute a new predictor to this repository, how do I do that?](#i-want-to-contribute-a-new-predictor-to-this-repository-how-do-i-do-that)

* [I want to reproduce your research](#i-want-to-reproduce-your-research)

### I want to evaluate my newly developed predictor

No install needed beyond numpy. Provide a CSV of predicted speaker scores and compare against the PathBench ground truth.


**Single dataset:**
```bash
python scripts/evaluate_from_csv.py \
    --predictions results/datasets/copas/pathological/word/balanced/dummy_scores.csv \
    --ground-truth datasets/copas/pathological/word/balanced/spk2score
```

**Full benchmark** — evaluate one evaluator across all datasets. Place your CSVs in a results directory that mirrors the dataset structure, with each CSV named `<evaluator>.csv`. Dummy score files are provided as a worked example:

```
results/datasets/
  copas/pathological/word/balanced/dummy_scores.csv
  torgo/pathological/utterances/balanced/dummy_scores.csv
  youtube/dummy_scores.csv
```

Then run:
```bash
python scripts/evaluate_from_csv.py \
    --results-dir results/datasets/ \
    --datasets-root datasets/ \
    --evaluator dummy_scores
```

This prints a table with the Pearson correlation for each dataset and the mean across all datasets.

Expected CSV format (speaker IDs must match the ground truth exactly):
```
speaker_id,score
C16,15.61
C17,16.74
```

All speaker IDs in the CSV must match the ground truth exactly — the script exits with an error if any are missing from either side.

If it works well, you might want to consider opening a pull request for your evaluator and incorporating into a codebase along the contribution
guidelines, as this is the only way we can ensure that you are not "cheating", i.e., using resources that are not allowed for your evaluator.

### I want to use the predictors developed by you

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karkirowle/pathbench/blob/main/examples/dartp_colab.ipynb) The [DArtP notebook](examples/dartp_colab.ipynb) installs PathBench, downloads the English language model and scores example and uploaded recordings with DArtP, the best reference-free metric. It runs on the free Colab tier.

Follow the steps in the [Installation](#installation) section. Then you can run inference on individual audio files. For example, to score a single utterance with ArtP:

```python
from pathbench import ArticulatoryPrecisionEvaluator

evaluator = ArticulatoryPrecisionEvaluator()
score = evaluator.score("utt1", "/path/to/audio.wav", transcription="the cat sat on the mat", language="en-us")
print(f"ArtP score: {score}")
```

### I want to contribute a new predictor to this repository, how do I do that?

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide.


### I want to reproduce your research

1. Follow the steps in the [Installation](#installation) section.
2. Follow the steps in the [Downloads](#downloads) section.
3. Follow the steps in the [Testing](#testing) section.
4. Run the evaluation script on the dataset(s) you want to evaluate. For example, to evaluate on the YouTube dataset:

```bash
python scripts/evaluate_spk2score.py datasets/youtube
```

To also write per-evaluator score CSVs, pass `--results-dir`:

```bash
python scripts/evaluate_spk2score.py datasets/youtube --results-dir results/datasets/
```

You can evaluate multiple datasets in a single run:

```bash
python scripts/evaluate_spk2score.py \
    datasets/copas/pathological/word/balanced \
    datasets/torgo/pathological/utterances/balanced \
    datasets/youtube
```

Results are written to the `results_11/` directory as timestamped text files containing per-evaluator Pearson correlations and a summary table.


## Installation

There are two ways to install PathBench, depending on what you need:

| | [Use PathBench in your project](#use-pathbench-in-your-project) | [Reproduce the paper](#reproduce-the-paper) |
|---|---|---|
| Install | `pip install` | `make` (or the container) |
| sudo needed | only for your OS's `espeak-ng` package | no |
| Python libraries | any version within the tested ranges | exact versions (`constraints.txt`) |
| espeak-ng | whatever your system has | built from the pinned commit |
| Scores vs. the paper | phoneme-based metrics may differ slightly | identical up to GPU rounding |

**What can still make scores differ.** Phoneme-based metrics (PER, dPER, ArtP, DArtP, Double ASR) turn text into phonemes with [espeak-ng](https://github.com/espeak-ng/espeak-ng), a system C library that pip cannot install. Different espeak-ng versions phonemize some languages differently (e.g. Italian `r` vs `ɾ`), so the paper uses commit [`2ea41210`](https://github.com/espeak-ng/espeak-ng/commit/2ea41210). Model weights are pinned to fixed Hugging Face revisions on both routes. Different GPUs round floating point differently, which moves the published correlations by up to about 0.01–0.03. No installation route can remove that.

### Use PathBench in your project

```bash
sudo apt-get install espeak-ng            # or your OS's package manager
pip install "pathbench @ git+https://github.com/karkirowle/pathbench.git"
```

Install PyTorch for your CUDA version first if you need a specific build ([pytorch.org](https://pytorch.org/get-started/locally/)). Optional extras:

- `pathbench[lm]` installs `kenlm` for the language-model evaluators (DArtP, Double ASR). It compiles from source, so it needs `cmake` and a C++ compiler.
- `pathbench[scripts]` installs what `scripts/` uses.

The dependency ranges in `pyproject.toml` are tested at both ends by CI, so any combination in between should work. To use the paper's espeak-ng without root, build it as in the `espeak.done` target of [`tools/Makefile`](tools/Makefile) and point phonemizer at it with `export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so`.

### Reproduce the paper

```bash
git clone https://github.com/karkirowle/pathbench.git
cd pathbench
make -C tools                  # CUDA 12.4 PyTorch; use `make -C tools CUDA_VERSION=` for CPU
source tools/venv/bin/activate
```

This builds espeak-ng `2ea41210` into `tools/espeak-ng/` (no sudo), creates `tools/venv` with the exact versions from [`constraints.txt`](constraints.txt), and points the venv at the local espeak-ng. It needs Python 3.10–3.12, `git`, `cmake` and a C/C++ compiler.

**No build tools, or an HPC cluster:** run the same `make` inside the [container](https://github.com/karkirowle/pathbench/releases/download/v0.1.0/pathbench.sif), which provides them. Its definition is [`container/pathbench.def`](container/pathbench.def).

```bash
apptainer exec --nv pathbench.sif make -C tools
apptainer exec --nv pathbench.sif tools/venv/bin/python scripts/evaluate_spk2score.py ...
```

## Downloads

### Datasets

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

* [EasyCall](http://neurolab.unife.it/easycallcorpus/)

* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

* [NeuroVoz](https://zenodo.org/records/10777657)

* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

* [Oral Cancer - YouTube](https://zenodo.org/records/18738598)

* [MDSC (Mandarin Dysarthric Speech Corpus)](https://www.aishelltech.com/AISHELL_6B)

After downloading the datasets, repoint the `wav.scp` files to your local dataset root. We do not provide a script for this, but you can use a regex replacement such as:

```bash
find datasets/ -name "wav.scp" -exec sed -i 's|/data/group1/z40484r/datasets|/path/to/your/datasets|g' {} +
```

**EasyCall fix:** Some EasyCall audio files have a stray space in their filename (`m13 _` instead of `m13_`). Rename them before running the benchmark:

```bash
find /path/to/your/datasets/easycall/EasyCall/m13 -name "m13 _*" -exec bash -c 'mv "$1" "${1//m13 _/m13_}"' _ {} \;
```

### N-gram models

The n-gram models required for DArtP and ArtP are included in the [Oral Cancer - YouTube](https://zenodo.org/records/18738598) download.

## Testing

### Installation integrity

It is recommended that after this setup you run the unit tests below. If these pass you can be reasonably sure about installation integrity.

```bash
source tools/venv/bin/activate
python -m pytest tests/test_evaluators.py::TestEvaluatorMethods -v
```

All tests should pass. If all evaluator tests fail simultaneously, the reference audio file in `tests/data/test_audio.wav` may be corrupted — the `test_audio_integrity` test will confirm this.

> **Note:** During the NAD evaluator tests you will see a `Wav2Vec2Model LOAD REPORT` table listing several keys (e.g. `project_q`, `quantizer`) as **UNEXPECTED**. These warnings are harmless — the keys belong to pre-training heads that are not needed for feature extraction and can be safely ignored.

### Dataset integrity

```bash
python -m pytest tests/test_evaluators.py::TestDatasetIntegrity::test_audio_file_hashes -v
```

Share these hashes alongside your results so others can verify they are using the same data.

> **Note:** Different versions of UASpeech exist. A denoising step was applied to UASpeech in December 2020. PathBench uses the **denoised** version canonically — `datasets/uaspeech/` points at the `noisereduce` audio set. If you have the pre-denoising audio your hashes will not match.



# Citation

If you use PathBench in your research, please cite:

```bibtex
@misc{halpern2026pathbenchspeechintelligibilitybenchmark,
      title={PathBench: Speech Intelligibility Benchmark for Automatic Pathological Speech Assessment},
      author={Bence Mark Halpern and Thomas Tienkamp and Defne Abur and Tomoki Toda},
      year={2026},
      eprint={2603.08097},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08097},
}
```

# License

The PathBench repository code is released under the MIT License. However, some individual evaluators include or derive from code released under more restrictive licenses. In particular, the `PraatSpeechRateEvaluator` in `pathbench/evaluators/speech_rate.py` is based on the Praat Script Syllable Nuclei by Nivja de Jong and Ton Wempe, which is licensed under the GNU General Public License v3 (GPL-3.0).

We are not able to provide legal advice. If you believe there is a licensing concern with any component in this codebase, please [open an issue](https://github.com/karkirowle/pathbench/issues).

# Acknowledgements

Many parts were shamelessly copied from others libraries or reproduced after consultation with those people.
I would like to especially say thanks to  Martijn Bartelds and Parvaneh Janbakhshi.
-  WADA-SNR: https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75
-  NAD: https://github.com/Bartelds/neural-acoustic-distance
-  CPP: https://github.com/satvik-dixit/CPP
-  Unit test audio from the [Speech Accent Archive](https://accent.gmu.edu/)

# Funding

This work is partly financed by the Dutch Research Council (NWO) under project number 019.232SG.011, and partly supported by JST CREST JPMJCR19A3, Japan.

## Author

Bence Mark Halpern, Nagoya University

