"""
Pseudo-perplexity computation using a masked language model (XLM-RoBERTa).

SCIENTIFIC NOTE
----------------
XLM-RoBERTa is a masked language model (MLM), not a causal/autoregressive
language model like GPT. Standard "perplexity" is defined for autoregressive
models as exp(average negative log-likelihood of each token given only the
PRECEDING tokens). That definition does not apply to an MLM, which is
trained to predict a masked token given BOTH left and right context.

For MLMs the accepted analogue is "pseudo-perplexity" (Salazar et al., 2020):
for each token position, mask that single token, run the model, take the
negative log-probability the model assigns to the true token, average this
over all positions, and exponentiate. This module implements exactly that,
and the rest of the codebase always refers to it as "pseudo-perplexity" to
avoid the false claim that it is standard perplexity.

The model is loaded once per worker process (see app/ml/model_loader.py) and
reused across requests to avoid reloading weights on every API call.

NOTE ON IMPORTS: `torch` is imported lazily, inside the functions below,
rather than at module level. This lets the rest of the backend (and the
training scripts that import this package for its feature-name constants)
run in environments where torch/transformers are not yet installed --
degrading gracefully to stylometry-only features instead of crashing on
import.
"""
import logging
import math
from typing import List

logger = logging.getLogger(__name__)

# A cap on how many tokens we score per segment to keep CPU inference time
# bounded for real-world deployment (see requirement: performance / CPU
# inference limitations).
_MAX_TOKENS_PER_SEGMENT = 128


def pseudo_perplexity_for_text(text: str, tokenizer, model, device: str = "cpu") -> float:
    """
    Computes pseudo-perplexity for a single chunk of text.

    Returns a float. Callers should treat very short inputs (fewer than ~5
    tokens) with caution, since pseudo-perplexity is noisy on tiny samples.
    """
    import torch

    with torch.no_grad():
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS_PER_SEGMENT,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        mask_token_id = tokenizer.mask_token_id
        seq_len = input_ids.shape[1]

        # Skip special tokens (CLS/SEP) at the start/end.
        special_ids = set(tokenizer.all_special_ids)
        positions = [
            i for i in range(seq_len) if input_ids[0, i].item() not in special_ids
        ]

        if not positions:
            return float("nan")

        log_probs: List[float] = []
        for pos in positions:
            masked_input = input_ids.clone()
            true_token_id = masked_input[0, pos].item()
            masked_input[0, pos] = mask_token_id

            outputs = model(input_ids=masked_input, attention_mask=attention_mask)
            logits = outputs.logits[0, pos]
            log_softmax = torch.log_softmax(logits, dim=-1)
            token_log_prob = log_softmax[true_token_id].item()
            log_probs.append(token_log_prob)

        avg_neg_log_prob = -sum(log_probs) / len(log_probs)
        pseudo_perplexity = math.exp(avg_neg_log_prob)
        return pseudo_perplexity


def pseudo_perplexity_for_segments(
    segments: List[str], tokenizer, model, device: str = "cpu"
) -> List[float]:
    """Computes pseudo-perplexity independently for each segment of text."""
    scores = []
    for segment in segments:
        if not segment.strip():
            continue
        try:
            score = pseudo_perplexity_for_text(segment, tokenizer, model, device)
            if not math.isnan(score):
                scores.append(score)
        except Exception:  # noqa: BLE001 - a single bad segment should not fail the request
            logger.exception("Failed to compute pseudo-perplexity for a segment.")
    return scores
