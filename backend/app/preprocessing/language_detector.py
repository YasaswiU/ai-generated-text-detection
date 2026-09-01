"""
Language detection for the "Auto Detect" option.

We use `langdetect` (a lightweight, well-established library) restricted to
the three languages this project currently supports: English, Telugu, Hindi.
If a fourth language is added later, extend `_LANG_MAP` and
`Settings.supported_languages` in app/core/config.py.
"""
import logging

from langdetect import DetectorFactory, LangDetectException, detect

# Make detection deterministic across requests/workers.
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
}


def detect_language(text: str, requested: str = "auto") -> tuple[str, str]:
    """
    Returns a (code, display_name) tuple, e.g. ("en", "English").

    If `requested` is not "auto", it is trusted as-is (still validated by the
    request schema/enum). Otherwise langdetect is used, falling back to
    English if detection fails or returns an unsupported language.
    """
    if requested != "auto":
        return requested, _LANG_MAP.get(requested, requested)

    try:
        code = detect(text)
    except LangDetectException:
        logger.warning("Language auto-detection failed; defaulting to English.")
        return "en", _LANG_MAP["en"]

    if code not in _LANG_MAP:
        # Unsupported language detected (e.g. French). We still analyze it
        # using the language-agnostic stylometric + perplexity pipeline, but
        # we are transparent about the fallback in the returned label.
        logger.info("Detected unsupported language '%s'; using generic analysis.", code)
        return code, code.upper()

    return code, _LANG_MAP[code]
