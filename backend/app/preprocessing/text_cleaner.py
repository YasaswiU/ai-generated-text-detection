"""
Lightweight, Unicode-safe text cleaning utilities.

These functions intentionally do very little: aggressive cleaning (e.g.
stripping punctuation) would destroy signal that the stylometric and
perplexity features rely on. We only normalise whitespace and validate size.
"""
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks for counting."""
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


def split_sentences(text: str) -> list[str]:
    """
    Unicode-aware sentence splitter.

    Splits on '.', '!', '?', and the Hindi/Telugu danda ('।') and double
    danda ('॥'), which are the standard sentence-final punctuation marks in
    Devanagari and Telugu script text.
    """
    text = text.strip()
    if not text:
        return []
    pattern = r"(?<=[.!?।॥])\s+"
    sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
    return sentences


def validate_length(text: str, min_length: int, max_length: int) -> None:
    """Raises ValueError with a user-safe message on invalid input length."""
    stripped = text.strip() if text else ""
    if not stripped:
        raise ValueError("Text must not be empty.")
    if len(stripped) < min_length:
        raise ValueError(
            f"Text is too short to analyze reliably. Please provide at least "
            f"{min_length} characters."
        )
    if len(stripped) > max_length:
        raise ValueError(
            f"Text is too long. Please provide at most {max_length} characters."
        )
