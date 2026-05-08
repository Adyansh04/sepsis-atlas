"""
Backend analytics helpers for contradiction detection and knowledge graphing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _is_not_reported(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "not reported", "n/a", "na", "none", "null"}


def _extract_float(text: str) -> float | None:
    if _is_not_reported(text):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _infer_effect_direction(evidence: Dict[str, Any], use_case: int) -> str:
    effect_text = str(evidence.get("effect_size", ""))
    quote_text = str((evidence.get("source") or {}).get("quote", ""))
    payload = f"{effect_text} {quote_text}".lower()

    ratio_match = re.search(r"\b(or|hr|rr)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", payload)
    if ratio_match:
        value = float(ratio_match.group(2))
        if value > 1:
            return "increase_risk"
        if value < 1:
            return "decrease_risk"

    if any(token in payload for token in ("increased mortality", "higher mortality", "worse outcome")):
        return "increase_risk"
    if any(token in payload for token in ("decreased mortality", "lower mortality", "protective")):
        return "decrease_risk"
    if use_case == 2:
        return "phenotype_descriptor"
    return "neutral"


def _extract_cutoff(evidence: Dict[str, Any]) -> str:
    payload = f"{evidence.get('effect_size', '')} {(evidence.get('source') or {}).get('quote', '')}"
    match = re.search(r"(>=|<=|>|<)\s*\d+(?:\.\d+)?", payload)
    return match.group(0).replace(" ", "") if match else "Not reported"


def _subject_outcome(evidence: Dict[str, Any], use_case: int) -> Tuple[str, str]:
    if use_case == 1:
        subject = str(evidence.get("predictor", "Not reported"))
        outcome = str(evidence.get("outcome_definition", "Not reported"))
    elif use_case == 3:
        subject = str(evidence.get("biomarker_name", evidence.get("biomarker_or_score", "Not reported")))
        outcome = str(evidence.get("outcome", "Not reported"))
    else:
        subject = str(evidence.get("clustering_method", "Not reported"))
        outcome = str(evidence.get("assignment_feasibility", "Not reported"))
    return subject.strip(), outcome.strip()


def build_contradiction_graph(use_case: int, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for idx, row in enumerate(evidence_list):
        subject, outcome = _subject_outcome(row, use_case)
        key = (subject, outcome)
        grouped.setdefault(key, [])
        grouped[key].append({"index": idx, "row": row})

    for (subject, outcome), group in grouped.items():
        for g in group:
            row = g["row"]
            source = row.get("source") or {}
            nodes.append(
                {
                    "id": g["index"],
                    "subject": subject,
                    "outcome": outcome,
                    "study_name": row.get("study_name", "Not reported"),
                    "effect_direction": _infer_effect_direction(row, use_case),
                    "cutoff": _extract_cutoff(row),
                    "effect_size": row.get("effect_size", "Not reported"),
                    "population": row.get("population", row.get("sample_size", "Not reported")),
                    "source_page": source.get("page_number", 0),
                    "source_quote": source.get("quote", "Not reported"),
                    "confidence": row.get("_confidence_score", 0),
                }
            )

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]["row"]
                b = group[j]["row"]
                dir_a = _infer_effect_direction(a, use_case)
                dir_b = _infer_effect_direction(b, use_case)
                cutoff_a = _extract_cutoff(a)
                cutoff_b = _extract_cutoff(b)
                reasons = []
                if dir_a != dir_b and "neutral" not in {dir_a, dir_b}:
                    reasons.append("opposite_effect_direction")
                if cutoff_a != "Not reported" and cutoff_b != "Not reported" and cutoff_a != cutoff_b:
                    reasons.append("different_cutoff")
                if reasons:
                    edges.append(
                        {
                            "source_id": group[i]["index"],
                            "target_id": group[j]["index"],
                            "subject": subject,
                            "outcome": outcome,
                            "reasons": reasons,
                        }
                    )

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "contradiction_count": len(edges),
            "subjects_with_conflicts": len({(e["subject"], e["outcome"]) for e in edges}),
        },
    }


def build_evidence_knowledge_graph(use_case: int, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    entities: Dict[str, Dict[str, Any]] = {}
    relations: List[Dict[str, Any]] = []

    for idx, row in enumerate(evidence_list):
        study = str(row.get("study_name", "Not reported"))
        subject, outcome = _subject_outcome(row, use_case)
        population = str(row.get("population", "Not reported"))
        effect = str(row.get("effect_size", "Not reported"))
        confidence = int(row.get("_confidence_score", 0) or 0)

        for etype, name in (
            ("study", study),
            ("subject", subject),
            ("outcome", outcome),
            ("population", population),
        ):
            key = f"{etype}:{name}"
            entities.setdefault(key, {"id": key, "type": etype, "name": name})

        relations.extend(
            [
                {
                    "type": "predicts_outcome",
                    "source": f"subject:{subject}",
                    "target": f"outcome:{outcome}",
                    "evidence_id": idx,
                    "effect_size": effect,
                    "confidence": confidence,
                },
                {
                    "type": "measured_in_population",
                    "source": f"subject:{subject}",
                    "target": f"population:{population}",
                    "evidence_id": idx,
                    "confidence": confidence,
                },
                {
                    "type": "reported_by_study",
                    "source": f"subject:{subject}",
                    "target": f"study:{study}",
                    "evidence_id": idx,
                    "confidence": confidence,
                },
            ]
        )

    strongest = None
    if relations:
        strongest = max(
            (r for r in relations if r["type"] == "predicts_outcome"),
            key=lambda x: x.get("confidence", 0),
            default=None,
        )

    return {
        "entities": list(entities.values()),
        "relations": relations,
        "summary": {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "strongest_evidence_path": strongest,
        },
    }
