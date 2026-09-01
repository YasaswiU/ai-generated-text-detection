"""
Probability calibration + calibrated-abstention decision logic.

This module is intentionally simple: the actual CalibratedClassifierCV
fitting happens offline in training/calibrate.py. At inference time we only
need to (a) trust that the loaded model's predict_proba is calibrated, and
(b) apply the abstention threshold chosen (also offline) in
training/calibrate.py and stored in model_metadata.json.

We deliberately do NOT call raw, uncalibrated scores "confidence" anywhere
in this codebase.
"""
from typing import Tuple

LABELS = ["HUMAN-WRITTEN", "AI-GENERATED"]


def decide_with_abstention(
    calibrated_proba_ai: float, threshold: float
) -> Tuple[str, float, bool]:
    """
    Args:
        calibrated_proba_ai: calibrated P(AI-GENERATED), in [0, 1].
        threshold: minimum calibrated confidence required to commit to a
            prediction (chosen via validation-set selective-risk analysis;
            see training/calibrate.py).

    Returns:
        (label, confidence, abstained)
    """
    proba_human = 1.0 - calibrated_proba_ai

    if calibrated_proba_ai >= proba_human:
        predicted_label = "AI-GENERATED"
        confidence = calibrated_proba_ai
    else:
        predicted_label = "HUMAN-WRITTEN"
        confidence = proba_human

    if confidence < threshold:
        return "UNCERTAIN", confidence, True

    return predicted_label, confidence, False
