import re
import string

def clean_text(text: str) -> str:
    """
    A function to clean text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing multiple spaces
    4. Stripping whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
