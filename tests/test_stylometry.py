"""Tests for backend/app/features/stylometry.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.features.stylometry import extract_stylometric_features


def test_basic_english_counts():
    text = "This is a test. This is only a test."
    features = extract_stylometric_features(text)
    assert features["word_count"] == 9
    assert features["sentence_count"] == 2
    assert features["character_count"] == len(text)


def test_empty_text_does_not_crash():
    features = extract_stylometric_features("")
    assert features["word_count"] == 0
    assert features["type_token_ratio"] == 0.0


def test_unicode_telugu_text():
    text = "నిన్న సాయంత్రం మా అమ్మతో కలిసి బజారుకు వెళ్ళాను."
    features = extract_stylometric_features(text)
    assert features["word_count"] > 0
    assert features["character_count"] == len(text)


def test_unicode_hindi_text():
    text = "कल रात बारिश बहुत तेज़ थी।"
    features = extract_stylometric_features(text)
    assert features["word_count"] > 0


def test_punctuation_frequency():
    text = "Hello, world! How are you? Fine; thanks: done."
    features = extract_stylometric_features(text)
    assert features["comma_frequency"] > 0
    assert features["exclamation_frequency"] > 0
    assert features["question_mark_frequency"] > 0
    assert features["semicolon_frequency"] > 0
    assert features["colon_frequency"] > 0


def test_type_token_ratio_range():
    text = "the the the cat sat on the mat"
    features = extract_stylometric_features(text)
    assert 0.0 <= features["type_token_ratio"] <= 1.0
