"""
Validation module — source anchoring and quote verification.

For each extracted evidence item the module checks whether the
``source.quote`` field can be approximately matched in the retrieved
chunks.  This provides a lightweight sanity check that the LLM did not
hallucinate values not present in the source material.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

def _word_overlap(text_a: str, text_b: str) -> float:
    """Return the fraction of words in *text_a* that also appear in *text_b*."""
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def verify_quote(
    quote: str,
    chunks: List[Dict[str, Any]],
    substring_threshold: float = 0.8,
    overlap_threshold: float = 0.65,
) -> bool:
    """
    Return *True* if *quote* is verifiably present in one of the chunks.

    Two checks are performed (either is sufficient to pass):

    1. **Substring match** — the normalised quote is a substring of the
       normalised chunk text (handles minor whitespace differences).
    2. **Word-overlap** — at least ``overlap_threshold`` of the quote's
       words appear in a single chunk (handles OCR noise and minor
       reformatting).

    A quote of "Not reported" is always considered verified.
    """
    if not quote or quote.strip().lower() == "not reported":
        return True

    quote_norm = " ".join(quote.lower().split())

    for chunk in chunks:
        chunk_norm = " ".join(chunk["text"].lower().split())

        # Check 1: substring
        if quote_norm in chunk_norm:
            return True

        # Check 2: word overlap
        if _word_overlap(quote_norm, chunk_norm) >= overlap_threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Per-item validation
# ---------------------------------------------------------------------------

def validate_extraction(
    evidence: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Add a ``_validation`` key to *evidence* describing the verification result.

    The dict contains:
    * ``quote_verified`` (bool)
    * ``status``         ("verified" | "unverified" | "not_applicable")
    * ``warning``        (str | None)
    """
    source = evidence.get("source")
    if isinstance(source, dict):
        quote = source.get("quote", "Not reported")
    else:
        quote = "Not reported"

    verified = verify_quote(quote, chunks)

    if quote.strip().lower() == "not reported":
        status = "not_applicable"
        warning = None
    elif verified:
        status = "verified"
        warning = None
    else:
        status = "unverified"
        warning = (
            "Source quote could not be matched in the retrieved passages. "
            "Please verify manually."
        )

    evidence["_validation"] = {
        "quote_verified": verified,
        "status": status,
        "warning": warning,
    }
    return evidence


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_all(
    evidence_list: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate every item in *evidence_list* and return the annotated list."""
    return [validate_extraction(e, chunks) for e in evidence_list]
