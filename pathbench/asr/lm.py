import os
from typing import List, Optional

# Per-language KenLM n-grams, as shipped in lms.zip of the "Accompanying model
# files for PathBench" Zenodo record (https://zenodo.org/records/18738598).
LM_FILES = {
    "en": "wiki_en_token.arpa",
    "nl": "wiki_nl_token.arpa",
    "es": "wiki_es_token.arpa.bin",
    "it": "wiki_it_token.arpa.bin",
    "cmn": "wiki_zh_token.arpa",
}


def resolve_lm_path(language: str, lm_dir: str = "lms", lm_path: Optional[str] = None) -> Optional[str]:
    """Return the KenLM file to decode ``language`` with.

    An explicit ``lm_path`` is used as given. Otherwise the per-language file is
    looked up in ``lm_dir`` (languages without their own LM fall back to
    English), preferring the binary ``<name>.arpa.bin`` over ``<name>.arpa``
    when both exist.
    """
    if lm_path is not None:
        return lm_path

    lm_lang = language.split('-')[0]
    if lm_lang not in LM_FILES:
        lm_lang = 'en'

    path = os.path.join(lm_dir, LM_FILES[lm_lang])
    if path.endswith('.arpa') and os.path.exists(path + '.bin'):
        path = path + '.bin'
    return path


def build_lm_decoder(labels: List[str], lm_path: str, lazy: bool = True):
    """Build a pyctcdecode beam-search decoder with a KenLM language model.

    Equivalent to ``pyctcdecode.build_ctcdecoder(labels, kenlm_model_path=lm_path)``
    with default settings, except for how KenLM loads the file. pyctcdecode
    uses KenLM's default (POPULATE_OR_READ), which reads the whole binary into
    memory: the 14.6 GB English LM then needs ~15 GB of RAM. With ``lazy=True``
    the binary is memory-mapped and pages are read on demand, so it also runs
    on smaller machines (e.g. free Colab). The loaded model, and therefore every
    score, is the same either way.
    """
    import kenlm
    from pyctcdecode.alphabet import Alphabet, verify_alphabet_coverage
    from pyctcdecode.decoder import BeamSearchDecoderCTC
    from pyctcdecode.language_model import LanguageModel, load_unigram_set_from_arpa

    config = kenlm.Config()
    if lazy:
        config.load_method = kenlm.LoadMethod.LAZY
    kenlm_model = kenlm.Model(lm_path, config)

    # As in build_ctcdecoder: unigrams can only be read from an ARPA file.
    unigrams = load_unigram_set_from_arpa(lm_path) if lm_path.endswith(".arpa") else None
    alphabet = Alphabet.build_alphabet(labels)
    if unigrams is not None:
        verify_alphabet_coverage(alphabet, unigrams)
    return BeamSearchDecoderCTC(alphabet, LanguageModel(kenlm_model, unigrams))
