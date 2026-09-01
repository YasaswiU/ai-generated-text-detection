"""
Application-wide logging configuration.

IMPORTANT (privacy): logs must never contain the raw text submitted by a
user. Only metadata (timing, language, prediction, model version) is logged.
See app/api/routes.py for how this is enforced at the call site.
"""
import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Avoid duplicate handlers when reload/uvicorn workers re-import this module.
        return

    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
