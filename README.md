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
    --predictions my_scores.csv \
    --ground-truth datasets/copas/pathological/word/balanced/spk2score
```

**Full benchmark** — evaluate one evaluator across all datasets. Place your CSVs in a results directory that mirrors the dataset structure, with each CSV named `<evaluator>.csv`:

```
results/
  copas/pathological/word/balanced/my_metric.csv
  torgo/pathological/utterances/balanced/my_metric.csv
  youtube/my_metric.csv
```

Then run:
```bash
python scripts/evaluate_from_csv.py \
    --results-dir results/ \
    --datasets-root datasets/ \
    --evaluator my_metric
```

This prints a table with the Pearson correlation for each dataset and the mean across all datasets.

Expected CSV format:
```
speaker_id,score
F01,2.5
M03,4.0
```

All speaker IDs in the CSV must match the ground truth exactly — the script exits with an error if any are missing from either side.

### I want to use the predictors developed by you

You have to install the framework but not download the stuff (datasets).

### I want to contribute a new predictor to this repository, how do I do that?

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide covering bug fixes, new evaluator development, unit test conventions, and coding conventions.

### I want to reproduce your research

Follow the steps in the [Installation](#installation) section.

Follow the steps in the [Downloads](#how-do-i-download-the-required-datasets) section.

After running the benchmark, verify that your evaluator implementations and datasets match the reference before comparing results.


All tests should pass. If all evaluator tests fail simultaneously, the reference audio file in `tests/data/test_audio.wav` may be corrupted — the `test_audio_integrity` test will confirm this.

**Check dataset integrity:**
```bash
# Print SHA256 hashes of your dataset files
python tests/test_evaluators.py --hash datasets/copas/pathological/word/balanced
```

Share these hashes alongside your results so others can verify they are using the same data.

## How do I download the required datasets?

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

* [EasyCall](http://neurolab.unife.it/easycallcorpus/)

* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

* [NeuroVoz](https://zenodo.org/records/10777657)

* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

* [Oral Cancer - YouTube](https://zenodo.org/records/18738598)
** This download also includes the n-grams for DArtP and ArtP

## Setup

After downloading the datasets, repoint the `wav.scp` files to your local dataset root:

```bash
python scripts/setup_paths.py /path/to/your/datasets
```

This automatically detects the current path prefix in all `wav.scp` files under `datasets/` and replaces it with the path you provide.

## Quick start

The following code shows how to evaluate a dataset and get the correlation of each metric with the ground truth scores.

## Installation

### Python dependencies

PathBench cannot be published to PyPI because it depends on Git-hosted forks of `phonemizer` and `pyctcdecode`.

The `make` installation route assumes the default setup of a standard Ubuntu 22.04 image (`ubuntu-2204-jammy`).

```bash
sudo apt install python3 python3-pip python3-venv build-essential -y
cd tools && make
cd ..
source tools/venv/bin/activate
```

### System dependencies

PathBench requires [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) for forced alignment via phonemizer.

**With sudo access:**
```bash
sudo apt-get install espeak-ng
```

**Without sudo access:** A containerised environment such as Docker is recommended.

### Testing installation

It is recommended that after this setup you run the unit tests below. If these pass you can be reasonably sure about installation integrity.

**Check evaluator implementations:**
```bash
source tools/venv/bin/activate
python -m unittest tests.test_evaluators -v
```


# Funding

This project was sponsored by the NWO Rubicon grant.

# Acknowledgements

Many parts were shamelessly copied from others libraries or reproduced after consultation with those people.
I would like to especially say thanks to  Martijn Bartelds and Parvaneh Janbakhshi.
-  WADA-SNR: https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75
-  NAD: https://github.com/Bartelds/neural-acoustic-distance
-  CPP: https://github.com/satvik-dixit/CPP

## Author

Bence Mark Halpern, Nagoya University

