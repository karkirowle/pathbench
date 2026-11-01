import re
import string

# Chinese punctuation characters not covered by string.punctuation (ASCII-only).
_CJK_PUNCTUATION = (
    "\u3000\u3001\u3002\uff01\uff0c\uff0e\uff1a\uff1b\uff1f"  # 。、！，．：；？
    "\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011"          # 《》「」『』【】
    "\u2018\u2019\u201c\u201d"                                   # ''""
    "\uff08\uff09"                                               # （）
    "\u2014\u2026"                                               # —…
)

_ALL_PUNCTUATION = string.punctuation + _CJK_PUNCTUATION

def clean_text(text: str) -> str:
    """
    A function to clean text by:
    1. Lowercasing
    2. Removing punctuation (ASCII and CJK)
    3. Removing multiple spaces
    4. Stripping whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", _ALL_PUNCTUATION))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


from functools import lru_cache


@lru_cache(maxsize=10000)
def cached_phonemize(text: str, language: str) -> str:
    """Cached phonemization via espeak. Pure function: same input -> same output."""
    from phonemizer.phonemize import phonemize
    from phonemizer.separator import Separator
    separator = Separator(phone=" ", word="|")
    return phonemize(
        text, language=language, backend="espeak",
        strip=True, preserve_punctuation=False, separator=separator
    )
