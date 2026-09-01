"""
Stylometric feature extraction.

Every function here is Unicode-aware so it produces sensible results on
English, Telugu, and Hindi text alike (no reliance on ASCII-only regexes for
letter detection; we use Python's `str.isalpha()`, `str.isdigit()`, and
`str.isupper()`/`isspace()`, which are Unicode-correct).
"""
import re
import statistics
from typing import Dict

from app.preprocessing.text_cleaner import split_paragraphs, split_sentences

_WORD_RE = re.compile(r"\S+", re.UNICODE)

_PUNCT_CHARS = {
    "comma": ",",
    "period": ".",
    "question_mark": "?",
    "exclamation": "!",
    "semicolon": ";",
    "colon": ":",
}


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def extract_stylometric_features(text: str) -> Dict[str, float]:
    words = _tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)

    word_count = len(words)
    char_count = len(text)
    sentence_count = max(len(sentences), 1)
    paragraph_count = max(len(paragraphs), 1)

    word_lengths = [len(w) for w in words] or [0]
    sentence_lengths = [len(_tokenize_words(s)) for s in sentences] or [0]

    avg_word_length = statistics.mean(word_lengths)
    avg_sentence_length = statistics.mean(sentence_lengths)
    sentence_length_variance = (
        statistics.pvariance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    )
    sentence_length_stdev = (
        statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    )

    vocabulary = {w.lower() for w in words}
    vocabulary_size = len(vocabulary)
    type_token_ratio = vocabulary_size / word_count if word_count else 0.0

    total_chars = max(char_count, 1)
    digit_count = sum(1 for c in text if c.isdigit())
    upper_count = sum(1 for c in text if c.isupper())
    lower_count = sum(1 for c in text if c.isalpha() and not c.isupper())
    whitespace_count = sum(1 for c in text if c.isspace())
    punctuation_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

    short_words = sum(1 for w in words if len(w) <= 3)
    long_words = sum(1 for w in words if len(w) >= 7)

    features: Dict[str, float] = {
        "word_count": float(word_count),
        "character_count": float(char_count),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(paragraph_count),
        "avg_word_length": float(avg_word_length),
        "avg_sentence_length": float(avg_sentence_length),
        "sentence_length_variance": float(sentence_length_variance),
        "sentence_length_stdev": float(sentence_length_stdev),
        "vocabulary_size": float(vocabulary_size),
        "type_token_ratio": float(type_token_ratio),
        "punctuation_density": float(punctuation_count / total_chars),
        "digit_ratio": float(digit_count / total_chars),
        "uppercase_ratio": float(upper_count / total_chars),
        "lowercase_ratio": float(lower_count / total_chars),
        "whitespace_ratio": float(whitespace_count / total_chars),
        "short_word_ratio": float(short_words / word_count) if word_count else 0.0,
        "long_word_ratio": float(long_words / word_count) if word_count else 0.0,
    }

    for name, char in _PUNCT_CHARS.items():
        count = text.count(char)
        features[f"{name}_frequency"] = float(count / total_chars)

    return features
