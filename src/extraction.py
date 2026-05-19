"""
Structured extraction module.

Sends retrieved chunks to an OpenRouter-compatible LLM and validates the
response against Pydantic schemas.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openai import OpenAI
from pydantic import ValidationError

from .schemas import UseCase1Evidence, UseCase2Evidence, UseCase3Evidence

# ---------------------------------------------------------------------------
# System prompt (shared across all use cases)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a strict clinical data extractor for sepsis research papers.

RULES — follow them exactly:
1. Extract ONLY values that are EXPLICITLY stated in the provided excerpts.
2. If a value is absent or unclear, write the string "Not reported".
3. Do NOT infer, guess, or interpolate missing values.
4. You MAY combine explicitly stated values across multiple sentences within the SAME excerpt.
5. For the "source.quote" field copy the EXACT sentence(s) that support the values.
6. For the "source.page_number" field use the page number given in the excerpt header.
7. Always include source.section_name and source.evidence_origin from excerpt metadata tags.
8. Return ONLY a valid JSON array (even for a single item). No explanations, no markdown fences.

CONTEXTUAL DISAMBIGUATION — critical for correctness:
9. When an excerpt reports results for MULTIPLE cohorts, subgroups, or statistical models, \
create SEPARATE evidence items for each. Never merge values from different subgroups into one row.
10. Ensure each numerical value (OR, HR, AUC, CI) is associated with the CORRECT predictor, \
population, outcome, and statistical model from the SAME analysis. Do NOT mix values across \
different regression models or subgroup analyses reported on the same page.
11. If a table presents multiple rows of results, extract each row as a separate evidence item \
with its own predictor, effect size, and confidence interval. Preserve the table structure.
12. When the same predictor is reported for different timepoints or outcome definitions, \
create separate rows and specify the timing/outcome clearly in each.\
"""

# ---------------------------------------------------------------------------
# Per-use-case extraction prompts
# ---------------------------------------------------------------------------

_PROMPT_UC1 = """\
TASK: Dynamic Extraction Mode — Counterfactual Mortality Estimation.
HUNT FOR: baseline risk equations and predictor weights (OR/HR/AUC), plus model adjustments.
PRIORITY: Extract numerical values from tables and results sections. Each row in a results \
table should become a separate evidence item. Ensure every OR/HR/AUC is linked to the correct \
predictor, cohort, and statistical model.

QUERY: {query}

EXCERPTS:
{context}

Return a JSON array where each element follows this schema exactly:
{{
  "study_name": "Author surname and year, e.g. Smith 2023",
  "population": "Patient population",
  "sample_size": "Total N or 'Not reported'",
  "setting": "Clinical setting (ICU/ED/ward) or 'Not reported'",
  "predictor": "Clinical variable or biomarker",
  "predictor_variable": "Explicit predictor variable label",
  "outcome_definition": "How the outcome is defined",
  "timing": "When predictor was measured or 'Not reported'",
  "statistical_method": "Method used or 'Not reported'",
  "effect_size": "e.g. OR 1.2, HR 2.4, AUC 0.78 or 'Not reported'",
  "performance_metrics": "Sensitivity/specificity/AUC/p-value or 'Not reported'",
  "notes": "Additional notes or 'Not reported'",
  "source": {{
    "page_number": <integer>,
    "quote": "Exact sentence(s) supporting this extraction",
    "section_name": "Section tag from excerpt metadata",
    "evidence_origin": "raw_text or vlm_summary or mixed"
  }}
}}

Return [] if no relevant data is found.\
"""

_PROMPT_UC2 = """\
TASK: Dynamic Extraction Mode — Sepsis Phenotype Extraction.
HUNT FOR: unsupervised clustering studies, reproducible assignment rules, and per-cluster descriptions.

QUERY: {query}

EXCERPTS:
{context}

Return a JSON array where each element follows this schema exactly:
{{
  "study_name": "Author surname and year",
  "country": "Country or 'Not reported'",
  "setting": "Clinical setting or 'Not reported'",
  "sample_size": "Total N or 'Not reported'",
  "sepsis_definition": "e.g. Sepsis-3 or 'Not reported'",
  "clustering_method": "e.g. k-means, latent class analysis",
  "num_clusters": "Number of clusters",
  "variables_used": ["variable 1", "variable 2"],
  "assignment_feasibility": "Assignable / Not assignable / Insufficient detail",
  "assignment_notes": "Why assignment is (not) possible or 'Not reported'",
  "assignment_is_reproducible": true,
  "reproducibility_notes": "Reasoning for boolean reproducibility flag",
    "phenotypes": [
      {{
        "cluster_id": "A / B / 1 / 2 etc.",
        "cluster_name": "Human-readable cluster name or 'Not reported'",
        "cluster_size": "Cluster size (N or %) or 'Not reported'",
        "key_features": "Key clinical/lab features",
        "clinical_description": "Clinical interpretation",
        "outcome": "Outcome for this cluster or 'Not reported'",
        "outcomes": "Outcome summary or 'Not reported'",
        "notes": "Notes or 'Not reported'"
      }}
  ],
  "source": {{
    "page_number": <integer>,
    "quote": "Exact sentence(s) supporting this extraction",
    "section_name": "Section tag from excerpt metadata",
    "evidence_origin": "raw_text or vlm_summary or mixed"
  }}
}}

Return [] if no relevant data is found.\
"""

_PROMPT_UC3 = """\
TASK: Dynamic Extraction Mode — Biomarker Selection for Risk Stratification.
HUNT FOR: head-to-head biomarker/score comparisons with AUROC-focused ranking signals.
PRIORITY: Extract each biomarker/score as a SEPARATE row. When a table compares multiple \
predictors, create one row per predictor with its own AUC/OR/HR. Ensure the cohort, \
outcome definition, and adjustment are correctly matched to each biomarker's row.

QUERY: {query}

EXCERPTS:
{context}

Return a JSON array where each element follows this schema exactly:
{{
  "study_name": "Author surname and year",
  "biomarker_or_score": "Name of biomarker or clinical score",
  "biomarker_name": "Canonical biomarker name",
  "biomarker_type": "Biomarker or Clinical Score",
  "cohort_setting": "ICU/ED/ward or 'Not reported'",
  "population": "Patient population",
  "sample_size": "Total N or 'Not reported'",
  "outcome": "Outcome measured",
  "effect_size": "e.g. OR 1.5, HR 2.1 or 'Not reported'",
  "auc": "AUC value or 'Not reported'",
  "auroc": 0.78,
  "confidence_interval": "95% CI or 'Not reported'",
  "adjustment": "Covariate adjustment or 'Not reported'",
  "statistical_method": "Method used",
  "validation_method": "Validation approach or 'Not reported'",
  "relevance_to_target_population": "Relevance to target population or 'Not reported'",
  "cohort_characteristics": "Key cohort characteristics or 'Not reported'",
  "notes": "Additional notes or 'Not reported'",
  "source": {{
    "page_number": <integer>,
    "quote": "Exact sentence(s) supporting this extraction",
    "section_name": "Section tag from excerpt metadata",
    "evidence_origin": "raw_text or vlm_summary or mixed"
  }}
}}

Return [] if no relevant data is found.\
"""

_PROMPTS: Dict[int, str] = {1: _PROMPT_UC1, 2: _PROMPT_UC2, 3: _PROMPT_UC3}
_SCHEMAS = {1: UseCase1Evidence, 2: UseCase2Evidence, 3: UseCase3Evidence}
_REQUIRED_FIELDS = {
    1: [
        "study_name",
        "population",
        "sample_size",
        "setting",
        "predictor",
        "outcome_definition",
        "effect_size",
        "source.quote",
        "source.page_number",
    ],
    2: [
        "study_name",
        "clustering_method",
        "num_clusters",
        "assignment_feasibility",
        "source.quote",
        "source.page_number",
    ],
    3: [
        "study_name",
        "biomarker_or_score",
        "population",
        "outcome",
        "effect_size",
        "auc",
        "source.quote",
        "source.page_number",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section_name", "Not reported")
        origin = chunk.get("evidence_origin", "raw_text")
        chunk_type = chunk.get("chunk_type", "child")
        parent_id = chunk.get("parent_id", "Not reported")
        parts.append(
            f"[Excerpt {i} | Source: {chunk['source']} | Page: {chunk['page_number']}]\n"
            f"[Metadata | section_name: {section} | evidence_origin: {origin} | "
            f"chunk_type: {chunk_type} | parent_id: {parent_id}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_json_response(raw: str) -> Any:
    """
    Parse a JSON value from the LLM response.

    Handles:
    * Plain JSON array / object
    * JSON wrapped in markdown fences (```json ... ```)
    * Single-object responses that should be wrapped in a list
    """
    raw = raw.strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON array or object
    array_match = re.search(r"\[[\s\S]*\]", raw)
    if array_match:
        return json.loads(array_match.group())

    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        return json.loads(obj_match.group())

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")


def _ensure_source(item: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee the 'source' key exists with valid defaults."""
    if "source" not in item or not isinstance(item["source"], dict):
        item["source"] = {
            "page_number": 0,
            "quote": "Not reported",
            "section_name": "Not reported",
            "evidence_origin": "raw_text",
        }
    else:
        item["source"].setdefault("page_number", 0)
        item["source"].setdefault("quote", "Not reported")
        item["source"].setdefault("section_name", "Not reported")
        item["source"].setdefault("evidence_origin", "raw_text")
    return item


def _missing_required_fields(item: Dict[str, Any], use_case: int) -> List[str]:
    missing: List[str] = []
    for field in _REQUIRED_FIELDS.get(use_case, []):
        if field.startswith("source."):
            src_key = field.split(".", 1)[1]
            value = (item.get("source") or {}).get(src_key)
        else:
            value = item.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def _apply_field_patch(item: Dict[str, Any], field: str, value: Any) -> None:
    if field.startswith("source."):
        item.setdefault("source", {})
        item["source"][field.split(".", 1)[1]] = value
    else:
        item[field] = value


def _schema_aware_repair(
    *,
    client: OpenAI,
    model: str,
    use_case: int,
    query: str,
    chunks: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    missing_map = []
    for idx, row in enumerate(rows):
        missing = _missing_required_fields(row, use_case)
        if missing:
            missing_map.append({"row_index": idx, "missing_fields": missing})

    if not missing_map:
        return rows

    repair_prompt = (
        f"QUERY: {query}\n\n"
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        f"CURRENT_ROWS_JSON:\n{json.dumps(rows, ensure_ascii=False)}\n\n"
        f"MISSING_FIELDS_BY_ROW:\n{json.dumps(missing_map, ensure_ascii=False)}\n\n"
        "TASK: Return ONLY a JSON array of patches. "
        "Each patch MUST include row_index and fields. "
        "Populate ONLY the listed missing fields for each row, using explicit evidence from context. "
        "If evidence is absent, set that field to 'Not reported'.\n"
        "Format:\n"
        '[{"row_index": 0, "fields": {"predictor": "...", "source.quote": "..."} }]\n'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        patches = _parse_json_response(response.choices[0].message.content or "")
    except Exception:
        return rows

    if isinstance(patches, dict):
        patches = patches.get("patches", [])
    if not isinstance(patches, list):
        return rows

    repaired = [dict(r) for r in rows]
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        row_index = patch.get("row_index")
        fields = patch.get("fields", {})
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(repaired):
            continue
        if not isinstance(fields, dict):
            continue
        for field, value in fields.items():
            if field in _missing_required_fields(repaired[row_index], use_case):
                _apply_field_patch(repaired[row_index], field, value)
    return repaired


def _extract_first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_uc1(item: Dict[str, Any]) -> Dict[str, Any]:
    predictor = item.get("predictor", "Not reported")
    predictor_variable = item.get("predictor_variable", "Not reported")
    if predictor == "Not reported" and predictor_variable != "Not reported":
        item["predictor"] = predictor_variable
    if predictor_variable == "Not reported" and predictor != "Not reported":
        item["predictor_variable"] = predictor
    return item


def _normalize_uc2(item: Dict[str, Any]) -> Dict[str, Any]:
    variables = item.get("variables_used", "Not reported")
    if isinstance(variables, str):
        if variables.strip().lower() == "not reported":
            item["variables_used"] = ["Not reported"]
        else:
            parts = [p.strip() for p in re.split(r"[,\n;]", variables) if p.strip()]
            item["variables_used"] = parts or ["Not reported"]

    feasibility = str(item.get("assignment_feasibility", "Not reported")).lower()
    if "assignable" in feasibility and "not" not in feasibility:
        item.setdefault("assignment_is_reproducible", True)
    elif any(token in feasibility for token in ("not assignable", "insufficient")):
        item.setdefault("assignment_is_reproducible", False)

    item.setdefault("reproducibility_notes", item.get("assignment_notes", "Not reported"))

    phenotypes = item.get("phenotypes", [])
    if isinstance(phenotypes, list):
        for ph in phenotypes:
            if not isinstance(ph, dict):
                continue
            cluster_id = ph.get("cluster_id", "Not reported")
            cluster_name = ph.get("cluster_name", "Not reported")
            outcome = ph.get("outcome", "Not reported")
            outcomes = ph.get("outcomes", "Not reported")
            if cluster_name == "Not reported" and cluster_id != "Not reported":
                ph["cluster_name"] = cluster_id
            if cluster_id == "Not reported" and cluster_name != "Not reported":
                ph["cluster_id"] = cluster_name
            if outcomes == "Not reported" and outcome != "Not reported":
                ph["outcomes"] = outcome
            if outcome == "Not reported" and outcomes != "Not reported":
                ph["outcome"] = outcomes
    return item


def _normalize_uc3(item: Dict[str, Any]) -> Dict[str, Any]:
    biomarker = item.get("biomarker_or_score", "Not reported")
    biomarker_name = item.get("biomarker_name", "Not reported")
    if biomarker_name == "Not reported" and biomarker != "Not reported":
        item["biomarker_name"] = biomarker
    if biomarker == "Not reported" and biomarker_name != "Not reported":
        item["biomarker_or_score"] = biomarker_name

    if "auroc" not in item or item.get("auroc") in (None, "Not reported", ""):
        item["auroc"] = _extract_first_float(item.get("auc", ""))

    item.setdefault("cohort_setting", "Not reported")
    return item


def _normalize_item(use_case: int, item: Dict[str, Any]) -> Dict[str, Any]:
    if use_case == 1:
        return _normalize_uc1(item)
    if use_case == 2:
        return _normalize_uc2(item)
    if use_case == 3:
        return _normalize_uc3(item)
    return item


def _detect_uc2_method(text: str) -> str:
    lower = text.lower()
    methods = [
        ("k-means", "k-means"),
        ("latent class analysis", "latent class analysis"),
        ("latent class", "latent class analysis"),
        ("hierarchical clustering", "hierarchical clustering"),
        ("gaussian mixture", "gaussian mixture model"),
        ("mixture model", "mixture model"),
        ("consensus clustering", "consensus clustering"),
        ("unsupervised clustering", "unsupervised clustering"),
    ]
    for token, label in methods:
        if token in lower:
            return label
    return "Not reported"


def _extract_uc2_num_clusters(text: str) -> str:
    patterns = [
        r"\b(\d+)\s+(?:clusters|phenotypes|classes|subtypes|endotypes)\b",
        r"\b(?:clusters?|phenotypes?|classes?|subtypes?|endotypes?)\s*(?:=|:)?\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not reported"


def _first_supporting_sentence(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]
    if not sentences:
        return "Not reported"
    keywords = (
        "phenotype",
        "cluster",
        "latent class",
        "subtype",
        "endotype",
        "k-means",
        "hierarchical",
        "mixture model",
    )
    for sent in sentences:
        lower = sent.lower()
        if any(k in lower for k in keywords):
            return sent
    return sentences[0]


def _salvage_uc2_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort UC2 fallback when LLM returns no parseable structured JSON."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        lower = text.lower()
        if not any(
            k in lower
            for k in (
                "phenotype",
                "cluster",
                "latent class",
                "subtype",
                "endotype",
                "unsupervised",
            )
        ):
            continue

        source = str(chunk.get("source", "Not reported"))
        page = int(chunk.get("page_number", 0) or 0)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        method = _detect_uc2_method(text)
        num_clusters = _extract_uc2_num_clusters(text)
        quote = _first_supporting_sentence(text)

        row = {
            "study_name": source.rsplit(".", 1)[0] if source else "Not reported",
            "country": "Not reported",
            "setting": "Not reported",
            "sample_size": "Not reported",
            "sepsis_definition": "Not reported",
            "clustering_method": method,
            "num_clusters": num_clusters,
            "variables_used": ["Not reported"],
            "assignment_feasibility": "Insufficient detail",
            "assignment_notes": (
                "Recovered from phenotype-related passage; assignment rule details not explicit."
            ),
            "assignment_is_reproducible": False,
            "reproducibility_notes": "Insufficient explicit assignment rules in extracted passage.",
            "phenotypes": [
                {
                    "cluster_id": "Not reported",
                    "cluster_name": "Not reported",
                    "cluster_size": "Not reported",
                    "key_features": quote if quote != "Not reported" else "Not reported",
                    "clinical_description": "Not reported",
                    "outcome": "Not reported",
                    "outcomes": "Not reported",
                    "notes": "Auto-recovered from passage-level evidence.",
                }
            ],
            "source": {
                "page_number": page,
                "quote": quote,
                "section_name": chunk.get("section_name", "Not reported"),
                "evidence_origin": chunk.get("evidence_origin", "raw_text"),
            },
        }
        rows.append(row)
        if len(rows) >= 8:
            break
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_llm_client(api_key: str, base_url: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at the OpenRouter API."""
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_evidence(
    query: str,
    chunks: List[Dict[str, Any]],
    use_case: int,
    api_key: str,
    model: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> List[Dict[str, Any]]:
    """
    Extract structured evidence from *chunks* using an LLM.

    Parameters
    ----------
    query     : The user's natural-language clinical query.
    chunks    : Retrieved text chunks (from retrieval module).
    use_case  : 1, 2, or 3 — selects the prompt and Pydantic schema.
    api_key   : OpenRouter API key.
    model     : Model identifier (OpenRouter slug).
    base_url  : OpenRouter base URL.

    Returns
    -------
    A list of validated evidence dicts (Pydantic model_dump output).
    Falls back to the raw dict if Pydantic validation fails, so the
    caller always receives *something* to display.
    """
    if not chunks:
        return []

    if use_case not in _PROMPTS:
        raise ValueError(f"use_case must be 1, 2, or 3; got {use_case}")

    client = build_llm_client(api_key, base_url)
    context = _format_context(chunks)

    user_prompt = _PROMPTS[use_case].format(query=query, context=context)
    schema_cls = _SCHEMAS[use_case]
    array_schema = {
        "type": "array",
        "items": schema_cls.model_json_schema(),
    }

    try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"use_case_{use_case}_evidence_array",
                        "strict": True,
                        "schema": array_schema,
                    },
                },
            )
        except Exception:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "401" in message or "User not found" in message:
            raise ValueError(
                "OpenRouter authentication failed (401). "
                "Check that your API key is valid and active."
            ) from exc
        raise

    raw_content = response.choices[0].message.content or ""

    try:
        data = _parse_json_response(raw_content)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[extraction] WARNING: JSON parse failed: {exc}")
        if use_case == 2:
            try:
                recovery_prompt = (
                    f"{user_prompt}\n\n"
                    "IMPORTANT: Return at least one JSON array item if any phenotype- or "
                    "cluster-related evidence exists. Use 'Not reported' for missing fields."
                )
                recovery = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": recovery_prompt},
                    ],
                    temperature=0.0,
                )
                data = _parse_json_response(recovery.choices[0].message.content or "")
            except Exception as recovery_exc:  # noqa: BLE001
                print(f"[extraction] WARNING: UC2 recovery parse failed: {recovery_exc}")
                data = _salvage_uc2_from_chunks(chunks)
        else:
            return []

    # Normalise to list
    if isinstance(data, dict):
        for key in ("items", "evidence", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data]
    elif not isinstance(data, list):
        return []

    working_rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item = _ensure_source(item)
        item = _normalize_item(use_case, item)
        working_rows.append(item)

    working_rows = _schema_aware_repair(
        client=client,
        model=model,
        use_case=use_case,
        query=query,
        chunks=chunks,
        rows=working_rows,
    )

    validated: List[Dict[str, Any]] = []
    for item in working_rows:
        try:
            evidence_obj = schema_cls(**item)
            validated.append(evidence_obj.model_dump())
        except ValidationError as exc:
            print(f"[extraction] WARNING: schema validation failed: {exc}")
            # Include with a validation warning so the user still sees data
            item["_schema_error"] = str(exc)
            validated.append(item)

    if use_case == 2 and not validated:
        fallback_rows = _salvage_uc2_from_chunks(chunks)
        if fallback_rows:
            for item in fallback_rows:
                try:
                    validated.append(schema_cls(**item).model_dump())
                except ValidationError as exc:
                    item["_schema_error"] = str(exc)
                    validated.append(item)

    return validated
