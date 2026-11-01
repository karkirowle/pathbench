# PathBench

<p align="center"><img src="assets/leonberger_transparent.png" width="150" /></p>

[![Unit Tests](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml/badge.svg)](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/path-benc/badge/?version=latest)](https://path-benc.readthedocs.io/en/latest/)
![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karkirowle/pathbench/blob/main/examples/dartp_colab.ipynb)

PathBench is a benchmark designed to evaluate pathological speech assessment systems.

## Results

Speaker-level Pearson correlations (PCC). Currently ArtP is the best reference-based, and DArtP is the best reference-free method.

| Table | GPU | Link |
|---|---|---|
| Canonical | RTX 3090 | [`results/results_table.csv`](results/results_table.csv) |
| Same benchmark | RTX 2080 Ti | [`results/results_table_2080ti.csv`](results/results_table_2080ti.csv) |
| Spreadsheet | | [Google Sheets](https://docs.google.com/spreadsheets/d/1ri-y_bHPgED3jJLuonChuwpaSP_3ddrx/edit?usp=sharing&ouid=112094007551573667400&rtpof=true&sd=true) |

Some cells differ by 0.01 between the two GPUs.

## Quick start

Try DArtP in the [Colab notebook](examples/dartp_colab.ipynb), or in your own code:

```python
from pathbench import ArtPDoubleASREvaluator, ArticulatoryPrecisionEvaluator

dartp = ArtPDoubleASREvaluator(language="en-us", lm_path="lms/wiki_en_token.arpa.bin")  # audio only
print(dartp.score("utt1", "audio.wav"))

artp = ArticulatoryPrecisionEvaluator()  # needs the transcript
print(artp.score("utt1", "audio.wav", transcription="the cat sat on the mat", language="en-us"))
```

Languages use espeak-ng codes: `en-us` (not `en`), `es`, `nl`, `it`, `cmn`.

## Usage guide

* [I want to evaluate my newly developed predictor](#i-want-to-evaluate-my-newly-developed-predictor)
* [I want to use the predictors developed by you](#i-want-to-use-the-predictors-developed-by-you)
* [I want to reproduce your research](#i-want-to-reproduce-your-research)
* [I want to contribute a new predictor](#i-want-to-contribute-a-new-predictor)

### I want to evaluate my newly developed predictor

You only need numpy and a clone of this repository. Compare a CSV of speaker scores against the ground truth:

```bash
python scripts/evaluate_from_csv.py \
    --predictions results/datasets/copas/pathological/word/balanced/dummy_scores.csv \
    --ground-truth datasets/copas/pathological/word/balanced/spk2score
```

The CSV has one row per speaker; the IDs must match the ground truth exactly:

```
speaker_id,score
C16,15.61
C17,16.74
```

For the full benchmark, put one `<evaluator>.csv` per dataset in a directory that mirrors `datasets/` (see the dummy files in `results/datasets/`) and run:

```bash
python scripts/evaluate_from_csv.py --results-dir results/datasets/ --datasets-root datasets/ --evaluator dummy_scores
```

If it works well, consider opening a pull request to add your evaluator along the [contribution guidelines](CONTRIBUTING.md), as this is the only way we can ensure that you are not "cheating", i.e., using resources that are not allowed for your evaluator.

### I want to use the predictors developed by you

```bash
sudo apt-get install espeak-ng
pip install "pathbench[lm] @ git+https://github.com/karkirowle/pathbench.git"
```

`[lm]` adds `kenlm` (compiled from source; needs `cmake`) for DArtP and Double ASR. These two also need a KenLM language model from `lms.zip` in the [Zenodo record](https://zenodo.org/records/18738598); the notebook's `download_zip_member` fetches a single language without the whole 35 GB.

This route allows a range of library versions and your system's espeak-ng, so phoneme-based scores can differ slightly from the paper (see [Reproducibility](#reproducibility)).

### I want to reproduce your research

1. **Install** the exact environment (no sudo; needs `git`, `cmake` and a C compiler):

   ```bash
   git clone https://github.com/karkirowle/pathbench.git && cd pathbench
   make -C tools                  # CPU only: make -C tools CUDA_VERSION=
   source tools/venv/bin/activate
   ```

   On an HPC cluster or without build tools, run `make` inside the [container](https://github.com/karkirowle/pathbench/releases/download/v0.1.0/pathbench.sif) ([`tools/pathbench.def`](tools/pathbench.def)): `apptainer exec --nv pathbench.sif make -C tools`.

2. **Download the datasets** below and repoint the `wav.scp` files, e.g. `find datasets/ -name "wav.scp" -exec sed -i 's|/data/group1/z40484r/datasets|/path/to/your/datasets|g' {} +`.

3. **Download the language models**: unzip `lms.zip` from [Zenodo](https://zenodo.org/records/18738598) in the repository root. The Mandarin model is not in it yet.

4. **Test** the installation and your data (see [Testing](#testing)).

5. **Run** the benchmark:

   ```bash
   python scripts/evaluate_spk2score.py datasets/youtube datasets/torgo/pathological/utterances/balanced
   ```

   Correlations go to `results_13/`. `--results-dir results/datasets/` also writes per-speaker scores; `--evaluators artp,artp_double_asr` runs only some metrics.

### I want to contribute a new predictor

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide.

## Datasets

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)
* [EasyCall](http://neurolab.unife.it/easycallcorpus/)
* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
* [NeuroVoz](https://zenodo.org/records/10777657)
* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)
* [Oral Cancer - YouTube](https://zenodo.org/records/18738598)
* [MDSC (Mandarin Dysarthric Speech Corpus)](https://www.aishelltech.com/AISHELL_6B)

**EasyCall fix:** some files have a stray space in their name (`m13 _` instead of `m13_`):

```bash
find /path/to/your/datasets/easycall/EasyCall/m13 -name "m13 _*" -exec bash -c 'mv "$1" "${1//m13 _/m13_}"' _ {} \;
```

**UASpeech:** PathBench uses the denoised version (December 2020); older copies will not match the hashes.

## Testing

```bash
python -m pytest tests/test_evaluators.py::TestEvaluatorMethods -v   # installation
python -m pytest tests/test_evaluators.py::TestAudioFileHashes -v    # your audio files
```

During the NAD tests a `Wav2Vec2Model LOAD REPORT` lists some keys as **UNEXPECTED**. This is harmless: they belong to pre-training heads that are not used.

## Reproducibility

The `make` route pins everything except the GPU: espeak-ng commit [`2ea41210`](https://github.com/espeak-ng/espeak-ng/commit/2ea41210), the library versions ([`tools/constraints.txt`](tools/constraints.txt)) and the model weights. The pip route pins only the model weights. The espeak-ng version matters most: different versions phonemize some languages differently (e.g. Italian `r` vs `ɾ`), which changes PER, dPER, ArtP, DArtP and Double ASR.

# Citation

If you use PathBench in your research, please cite:

```bibtex
@inproceedings{halpern26_interspeech,
  title     = {{PathBench: Speech Intelligibility Benchmark for Automatic Pathological Speech Assessment}},
  author    = {Bence Mark Halpern and Thomas Tienkamp and Defne Abur and Tomoki Toda},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {7526--7531},
  doi       = {10.21437/Interspeech.2026-946},
  issn      = {2958-1796},
}
```

If you use ART-NAD, please also cite:

```bibtex
@misc{halpern2026artnad,
  title         = {{ART-NAD: An Articulatory Inversion-based Neural Acoustic Distance for Pathological Speech Intelligibility Assessment}},
  author        = {Bence Mark Halpern and Thomas Tienkamp and Defne Abur and Tomoki Toda},
  year          = {2026},
  eprint        = {2609.24046},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  note          = {Accepted at SLT 2026},
  url           = {https://arxiv.org/abs/2609.24046},
}
```

# License

The PathBench repository code is released under the MIT License. However, some individual evaluators include or derive from code released under more restrictive licenses. In particular, the `PraatSpeechRateEvaluator` in `pathbench/evaluators/speech_rate.py` is based on the Praat Script Syllable Nuclei by Nivja de Jong and Ton Wempe, which is licensed under the GNU General Public License v3 (GPL-3.0). The example sentence in `examples/audio/` is from LibriTTS (CC BY 4.0).

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
