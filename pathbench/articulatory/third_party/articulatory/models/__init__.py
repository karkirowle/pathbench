# -*- coding: utf-8 -*-
# Trimmed from upstream: only the BiGRU inversion model is vendored. The
# upstream __init__ also re-exported hifigan/melgan/parallel_wavegan/
# style_melgan/transformer/gblock_gen, none of which PathBench uses.
from .pytorch_models import BiGRU  # NOQA
