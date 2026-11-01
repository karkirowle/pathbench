# -*- coding: utf-8 -*-
"""Vendored minimal subset of the ``articulatory`` package.

Upstream: Wu et al., "Speaker-independent Acoustic-to-Articulatory Inversion"
(ICASSP 2023, arXiv:2302.06774), https://github.com/articulatory/articulatory
(version 0.5.3, Apache-2.0). See ``LICENSE`` in this directory. Some leaf
modules carry upstream ParallelWaveGAN MIT headers, retained verbatim.

Only the pieces the PathBench articulatory runner actually exercises are kept:
the ``BiGRU`` inversion model, the layers it imports, ``PQMF`` (instantiated by
``load_model`` for multi-channel generators), and ``load_model`` / ``read_hdf5``.
The upstream repo also ships HiFi-GAN, MelGAN, Parallel WaveGAN, StyleMelGAN,
the full ESPnet ``nets`` tree, training ``bin`` scripts, datasets, losses, etc.
— none of which PathBench uses; those were dropped rather than vendored.

Leaf modules (``models/pytorch_models.py``, ``layers/{pqmf,residual_block,
pytorch_layers}.py``) are byte-for-byte upstream copies so they can be diffed
against upstream on update. Only the ``__init__`` files and ``utils/utils.py``
were trimmed.
"""

__version__ = "0.5.3"
