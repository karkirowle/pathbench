# PathBench

<p align="center"><img src="assets/leonberger_transparent.png" width="150" /></p>

[![Unit Tests](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml/badge.svg)](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml)

PathBench is a benchmark designed to evaluate pathological speech assessment systems.

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

### I want to use the predictors developed by you

Follow the steps in the [Installation](#installation) section.

### I want to contribute a new predictor to this repository, how do I do that?

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide covering bug fixes, new evaluator development, unit test conventions, and coding conventions.

### I want to reproduce your research

1. Follow the steps in the [Installation](#installation) section.
2. Follow the steps in the [Downloads](#downloads) section.
3. Follow the steps in the [Testing](#testing) section.


## Installation

### Package installation

PathBench cannot be published to PyPI because it depends on Git-hosted forks of `phonemizer` and `pyctcdecode`.

### Make installation

The `make` installation route assumes the default setup of a standard Ubuntu 22.04 image (`ubuntu-2204-jammy`).

```bash
sudo apt-get update -qq
sudo apt install python3 python3-pip python3-venv build-essential cmake espeak-ng libfftw3-dev liblapack-dev -y
git clone git@github.com:karkirowle/pathbench.git
cd pathbench/tools && make
cd ..
source tools/venv/bin/activate
```

**Without sudo access:** A containerised environment such as Docker is recommended.

## Downloads

### Datasets

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

* [EasyCall](http://neurolab.unife.it/easycallcorpus/)

* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

* [NeuroVoz](https://zenodo.org/records/10777657)

* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

* [Oral Cancer - YouTube](https://zenodo.org/records/18738598)

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
python -m unittest tests.test_evaluators -v
```

All tests should pass. If all evaluator tests fail simultaneously, the reference audio file in `tests/data/test_audio.wav` may be corrupted — the `test_audio_integrity` test will confirm this.

> **Note:** During the NAD evaluator tests you will see a `Wav2Vec2Model LOAD REPORT` table listing several keys (e.g. `project_q`, `quantizer`) as **UNEXPECTED**. These warnings are harmless — the keys belong to pre-training heads that are not needed for feature extraction and can be safely ignored.

### Dataset integrity

```bash
python -m pytest tests/test_evaluators.py::TestDatasetIntegrity::test_audio_file_hashes -v
```

Share these hashes alongside your results so others can verify they are using the same data.

> **Note:** Different versions of UASpeech exist. A denoising step was applied to UASpeech in December 2020, so hashes will differ depending on whether you have the original or denoised version.



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

