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
# Cross-row consistency checks
# ---------------------------------------------------------------------------

def _check_cross_row_consistency(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flag potential semantic misalignment across rows.

    Detects:
    - Duplicate effect sizes assigned to different predictors from same study
    - Identical quotes used for different evidence rows (copy-paste risk)
    """
    if len(evidence_list) < 2:
        return evidence_list

    # Index quotes → rows that use them
    quote_to_rows: Dict[str, List[int]] = {}
    for idx, row in enumerate(evidence_list):
        source = row.get("source")
        if not isinstance(source, dict):
            continue
        quote = (source.get("quote", "") or "").strip().lower()
        if quote and quote != "not reported" and len(quote) > 30:
            norm_quote = " ".join(quote.split())
            quote_to_rows.setdefault(norm_quote, []).append(idx)

    # Index (study, effect_size) → rows
    study_effect_to_rows: Dict[tuple, List[int]] = {}
    for idx, row in enumerate(evidence_list):
        study = str(row.get("study_name", "")).strip().lower()
        effect = str(row.get("effect_size", "")).strip().lower()
        if study and effect and effect != "not reported":
            key = (study, effect)
            study_effect_to_rows.setdefault(key, []).append(idx)

    # Annotate warnings
    for norm_quote, indices in quote_to_rows.items():
        if len(indices) > 1:
            # Check if different predictors/biomarkers share the same quote
            predictors = set()
            for i in indices:
                p = evidence_list[i].get("predictor") or evidence_list[i].get("biomarker_name") or ""
                if p and p.lower() != "not reported":
                    predictors.add(p.lower())
            if len(predictors) > 1:
                for i in indices:
                    warnings = evidence_list[i].setdefault("_consistency_warnings", [])
                    warnings.append(
                        "Same source quote supports different predictors — "
                        "verify that each value is correctly attributed."
                    )

    for (study, effect), indices in study_effect_to_rows.items():
        if len(indices) > 1:
            predictors = set()
            for i in indices:
                p = evidence_list[i].get("predictor") or evidence_list[i].get("biomarker_name") or ""
                if p and p.lower() != "not reported":
                    predictors.add(p.lower())
            if len(predictors) > 1:
                for i in indices:
                    warnings = evidence_list[i].setdefault("_consistency_warnings", [])
                    warnings.append(
                        f"Same effect size '{effect}' from '{study}' assigned to different "
                        "predictors — possible cross-row misattribution."
                    )

    return evidence_list


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_all(
    evidence_list: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate every item in *evidence_list* and return the annotated list."""
    validated = [validate_extraction(e, chunks) for e in evidence_list]
    validated = _check_cross_row_consistency(validated)
    return validated
