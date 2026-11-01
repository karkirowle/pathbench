# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions.

Trimmed from upstream ``articulatory/utils/utils.py``: only ``read_hdf5`` (used
by ``BiGRU.register_stats``) and ``load_model`` (used by the released-checkpoint
path of the PathBench articulatory runner) are kept. The upstream file also held
HDF5 writers, file-glob helpers, and the ``gdown``-based pretrained-model
downloader (with its ``PRETRAINED_MODEL_LIST`` of hifigan/melgan/wavegan tags)
— none of which PathBench uses. ``read_hdf5`` and ``load_model`` below are
byte-for-byte upstream.
"""

import logging
import os
import sys

from distutils.version import LooseVersion

import torch
import yaml


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    import h5py  # lazy: only the stats path needs it, which PathBench never hits

    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def load_model(checkpoint, config=None, stats=None, generator2=False):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    if generator2:
        type_key = "generator2_type"
        params_key = "generator2_params"
        generator_key = "generator2"
    else:
        type_key = "generator_type"
        params_key = "generator_params"
        generator_key = "generator"
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import articulatory.models

    # get model and load parameters
    model_class = getattr(
        articulatory.models,
        config.get(type_key, "ParallelWaveGANGenerator"),
    )
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config[params_key].items()
    }
    model = model_class(**generator_params)
    if generator2:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["model"][generator_key][0]
        )
    else:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["model"][generator_key]
        )

    # check stats existence
    if stats is None:
        dirname = os.path.dirname(checkpoint)
        if config["format"] == "hdf5":
            ext = "h5"
        else:
            ext = "npy"
        if os.path.exists(os.path.join(dirname, f"stats.{ext}")):
            stats = os.path.join(dirname, f"stats.{ext}")

    # load stats
    if stats is not None:
        model.register_stats(stats)

    # add pqmf if needed
    if config[params_key]["out_channels"] > 1:
        # lazy load for circular error
        from articulatory.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config[params_key]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model
