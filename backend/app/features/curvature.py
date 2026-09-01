"""
Curvature features derived from segment-level pseudo-perplexity.

Given a sequence of per-segment pseudo-perplexity scores P = [P_1..P_n],
this module computes first and second differences and summarises their
shape. The intuition (documented, not overclaimed) is that AI-generated text
often has a flatter/more uniform predictability curve than human writing,
but this is a WEAK, probabilistic signal on its own -- it is combined with
stylometry and pseudo-perplexity in the final classifier, and the app never
claims curvature alone proves AI generation.
"""
from typing import Dict, List

import numpy as np


def compute_curvature_features(segment_scores: List[float]) -> Dict[str, float]:
    if len(segment_scores) < 2:
        # Not enough segments to compute a meaningful curve.
        return {
            "mean_first_difference": 0.0,
            "std_first_difference": 0.0,
            "mean_abs_first_difference": 0.0,
            "mean_second_difference": 0.0,
            "std_second_difference": 0.0,
            "mean_abs_second_difference": 0.0,
            "max_absolute_curvature": 0.0,
            "curvature_sign_changes": 0,
        }

    p = np.array(segment_scores, dtype=float)

    first_diff = np.diff(p)  # ΔP_i = P_(i+1) - P_i

    features: Dict[str, float] = {
        "mean_first_difference": float(np.mean(first_diff)),
        "std_first_difference": float(np.std(first_diff)),
        "mean_abs_first_difference": float(np.mean(np.abs(first_diff))),
    }

    if len(p) >= 3:
        # Δ²P_i = P_(i+1) - 2*P_i + P_(i-1)
        second_diff = p[2:] - 2 * p[1:-1] + p[:-2]
        signs = np.sign(second_diff)
        nonzero_signs = signs[signs != 0]
        sign_changes = int(np.sum(np.diff(nonzero_signs) != 0)) if len(nonzero_signs) > 1 else 0

        features.update(
            {
                "mean_second_difference": float(np.mean(second_diff)),
                "std_second_difference": float(np.std(second_diff)),
                "mean_abs_second_difference": float(np.mean(np.abs(second_diff))),
                "max_absolute_curvature": float(np.max(np.abs(second_diff))),
                "curvature_sign_changes": sign_changes,
            }
        )
    else:
        features.update(
            {
                "mean_second_difference": 0.0,
                "std_second_difference": 0.0,
                "mean_abs_second_difference": 0.0,
                "max_absolute_curvature": 0.0,
                "curvature_sign_changes": 0,
            }
        )

    return features
