import os
from typing import Optional

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
