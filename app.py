"""
Sepsis Atlas — Streamlit UI entrypoint.

Run with:
    streamlit run app.py

Environment variables (see .env.example):
    OPENROUTER_API_KEY   : Your OpenRouter API key
    DEFAULT_MODEL        : LLM model slug (default: llama-3.1-8b-instruct:free)
    PDF_DIR              : Folder containing PDFs (default: ./pdfs)
    CHROMA_PERSIST_DIR   : ChromaDB storage path (default: ./chroma_db)
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st

# Ensure src/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from src.extraction import extract_evidence
from src.ingest import ingest_pdfs
from src.analytics import build_contradiction_graph, build_evidence_knowledge_graph
from src.retrieval import (
    clear_collection,
    get_collection_count,
    get_indexed_sources,
    index_chunks,
    query_chunks_enhanced,
)
from src.validation import validate_all

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sepsis Atlas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polished UI
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---- Global typography & spacing ---- */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Main header styling */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* Section headers */
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

h4 {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 1.5rem !important;
}

/* Cards for metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-radius: 8px;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border-radius: 8px;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.15);
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"], .stButton > button:not([kind]) {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    transition: all 0.2s ease;
}

/* Download buttons */
.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

.stDownloadButton > button:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
}

/* Text input styling */
.stTextInput > div > div {
    border-radius: 8px !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
}

.stTextInput > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
}

[data-testid="stSidebar"] .stDivider {
    border-color: rgba(99, 102, 241, 0.15);
}

/* Info/success/warning/error boxes */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    border-radius: 8px !important;
}

/* Select box */
.stSelectbox > div > div {
    border-radius: 8px !important;
}

/* Dividers */
hr {
    border-color: rgba(99, 102, 241, 0.1) !important;
    margin: 1.5rem 0 !important;
}

/* Caption text */
.stCaption {
    color: #64748b !important;
}

/* Hero subtitle */
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
}

/* Pipeline status card */
.pipeline-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
}

/* Scrollable log container improvements */
.log-container {
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 12px;
    background: rgba(15, 23, 42, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Use-case metadata
# ---------------------------------------------------------------------------

USE_CASES: dict = {
    1: {
        "name": "Counterfactual Mortality Estimation",
        "short": "UC1: Mortality",
        "description": (
            "Extract predictor–outcome associations, effect sizes, and "
            "statistical context to support estimation of expected mortality "
            "without a control group."
        ),
        "example_query": (
            "What is the relationship between initial lactate level and "
            "28-day mortality in septic shock?"
        ),
    },
    2: {
        "name": "Sepsis Phenotype Extraction",
        "short": "UC2: Phenotypes",
        "description": (
            "Identify published sepsis phenotypes — clustering methods, "
            "variables used, cluster characteristics, and per-phenotype outcomes."
        ),
        "example_query": (
            "What sepsis phenotypes have been identified using unsupervised "
            "clustering methods?"
        ),
    },
    3: {
        "name": "Biomarker Selection for Risk Stratification",
        "short": "UC3: Biomarkers",
        "description": (
            "Compare biomarkers and clinical scores by predictive performance "
            "(AUC, OR, HR) to support evidence-based selection of "
            "stratification variables."
        ),
        "example_query": (
            "Which biomarkers or scores best predict 28-day mortality in "
            "sepsis patients?"
        ),
    },
}

# ---------------------------------------------------------------------------
# Available OpenRouter models
# ---------------------------------------------------------------------------

MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "x-ai/grok-4.20-multi-agent",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.7"
]

UC2_KEYWORDS = [
    "phenotype",
    "phenotypes",
    "cluster",
    "clustering",
    "latent class",
    "subtype",
    "subtypes",
    "endotype",
    "endotypes",
    "unsupervised",
    "k-means",
    "hierarchical",
    "mixture model",
]

UC3_KEYWORDS = [
    "biomarker",
    "score",
    "auc",
    "auroc",
    "roc",
    "c-index",
    "sensitivity",
    "specificity",
    "hazard ratio",
    "odds ratio",
    "sofa",
    "saps ii",
    "apache",
    "lactate",
    "procalcitonin",
    "il-6",
]

NOT_REPORTED_MARKERS = {
    "",
    "n/a",
    "na",
    "not reported",
    "not available",
    "none",
    "null",
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _is_not_reported(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, list):
        return len(value) == 0
    text = str(value).strip().lower()
    return text in NOT_REPORTED_MARKERS


def _compute_confidence(evidence: dict) -> tuple[int, str]:
    validation_status = (evidence.get("_validation", {}) or {}).get("status", "")
    quote = ((evidence.get("source") or {}).get("quote", "") or "").strip()
    core_fields = {
        k: v
        for k, v in evidence.items()
        if k not in {"source", "_validation", "_schema_error", "phenotypes"}
    }
    present = sum(0 if _is_not_reported(v) else 1 for v in core_fields.values())
    total = max(len(core_fields), 1)
    completeness = present / total

    status_score = {
        "verified": 0.5,
        "unverified": 0.2,
        "not_applicable": 0.35,
    }.get(validation_status, 0.2)
    quote_score = 0.25 if not _is_not_reported(quote) and len(quote) >= 20 else 0.05
    completeness_score = 0.25 * completeness
    score = int(round((status_score + quote_score + completeness_score) * 100))
    explain = (
        f"status={validation_status or 'unknown'} ({status_score:.2f}), "
        f"quote_quality={'good' if quote_score >= 0.25 else 'weak'} ({quote_score:.2f}), "
        f"field_completeness={present}/{total} ({completeness:.0%})"
    )
    return max(0, min(score, 100)), explain


def _enrich_evidence(evidence_list: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for e in evidence_list:
        item = dict(e)
        score, explain = _compute_confidence(item)
        item["_confidence_score"] = score
        item["_confidence_explain"] = explain
        enriched.append(item)
    return enriched


def _attempt_metrics(validated: list[dict]) -> dict:
    total = len(validated)
    if total == 0:
        return {
            "evidence_count": 0,
            "verified_ratio": 0.0,
            "coverage": 0.0,
            "avg_confidence": 0.0,
        }
    verified = sum(
        1
        for e in validated
        if (e.get("_validation", {}) or {}).get("status") == "verified"
    )
    coverage_scores = []
    for e in validated:
        fields = {
            k: v
            for k, v in e.items()
            if k not in {"source", "_validation", "_schema_error", "phenotypes"}
        }
        present = sum(0 if _is_not_reported(v) else 1 for v in fields.values())
        coverage_scores.append(present / max(len(fields), 1))
    avg_conf = sum(int(e.get("_confidence_score", 0)) for e in validated) / total
    return {
        "evidence_count": total,
        "verified_ratio": verified / total,
        "coverage": sum(coverage_scores) / max(len(coverage_scores), 1),
        "avg_confidence": avg_conf / 100.0,
    }


def _retry_reason(metrics: dict) -> str | None:
    reasons = []
    if metrics["evidence_count"] == 0:
        reasons.append("no_rows")
    if metrics["verified_ratio"] < 0.35:
        reasons.append("low_quote_verification")
    if metrics["coverage"] < 0.45:
        reasons.append("low_schema_coverage")
    if metrics["avg_confidence"] < 0.45:
        reasons.append("low_confidence")
    return ",".join(reasons) if reasons else None


def _render_retry_metrics(report: dict | None) -> None:
    if not report:
        return
    attempts = report.get("attempts", [])
    if not attempts:
        return
    st.markdown("#### 🔄 Adaptive Retrieval Metrics")
    rows = []
    for a in attempts:
        m = a.get("metrics", {})
        rows.append(
            {
                "Attempt": a.get("attempt"),
                "Top-K": a.get("top_k"),
                "Rows": m.get("evidence_count", 0),
                "Verified %": round(100 * m.get("verified_ratio", 0), 1),
                "Coverage %": round(100 * m.get("coverage", 0), 1),
                "Avg Confidence": round(100 * m.get("avg_confidence", 0), 1),
                "Retry Reason": a.get("retry_reason", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_backend_graph_outputs(uc: int, validated: list[dict]) -> None:
    contradiction_graph = build_contradiction_graph(uc, validated)
    knowledge_graph = build_evidence_knowledge_graph(uc, validated)

    st.markdown("#### ⚡ Contradiction Graph Builder")
    c_summary = contradiction_graph.get("summary", {})
    st.caption(
        f"conflicts: {c_summary.get('contradiction_count', 0)} · "
        f"conflict_subjects: {c_summary.get('subjects_with_conflicts', 0)}"
    )
    edges = contradiction_graph.get("edges", [])
    if edges:
        st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
    else:
        st.info("No direct contradictions detected for this query.")

    st.markdown("#### 🕸️ Evidence Knowledge Graph")
    k_summary = knowledge_graph.get("summary", {})
    strongest = k_summary.get("strongest_evidence_path")
    st.caption(
        f"entities: {k_summary.get('entity_count', 0)} · "
        f"relations: {k_summary.get('relation_count', 0)}"
    )
    if strongest:
        st.success(
            "Strongest evidence path: "
            f"{strongest.get('source')} → {strongest.get('target')} "
            f"(confidence {strongest.get('confidence', 0)}/100)"
        )
    with st.expander("View KG relations", expanded=False):
        st.dataframe(
            pd.DataFrame(knowledge_graph.get("relations", [])).head(80),
            use_container_width=True,
            hide_index=True,
        )


def _render_trust_dashboard(evidence_list: list[dict], uc: int) -> None:
    total = len(evidence_list)
    if total == 0:
        return
    statuses = [
        (e.get("_validation", {}) or {}).get("status", "unknown")
        for e in evidence_list
    ]
    verified = sum(1 for s in statuses if s == "verified")
    unverified = sum(1 for s in statuses if s == "unverified")
    avg_conf = int(
        round(
            sum(int(e.get("_confidence_score", 0)) for e in evidence_list) / total
        )
    )

    coverage_values = []
    for e in evidence_list:
        fields = {
            k: v
            for k, v in e.items()
            if k not in {"source", "_validation", "_schema_error", "phenotypes"}
        }
        present = sum(0 if _is_not_reported(v) else 1 for v in fields.values())
        coverage_values.append(present / max(len(fields), 1))
    schema_coverage = int(round(100 * (sum(coverage_values) / max(len(coverage_values), 1))))

    st.markdown("#### 🛡️ Trust Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence Rows", total, help="Total structured evidence items extracted")
    c2.metric("Verified Quotes", f"{verified}/{total}", help="Quotes matched in source text")
    c3.metric("Avg Confidence", f"{avg_conf}/100", help="Composite score: verification + completeness + quote quality")
    c4.metric("Schema Coverage", f"{schema_coverage}%", help="Percentage of schema fields populated with real values")
    if unverified:
        st.warning(f"⚠️ {unverified} row(s) have unverified quotes. Review source evidence before downstream use.")

    _render_consensus_snapshot(uc, evidence_list)
    _render_citation_integrity(evidence_list)


def _render_consensus_snapshot(uc: int, evidence_list: list[dict]) -> None:
    st.markdown("#### 🔬 Consensus Snapshot")
    if uc == 1:
        predictors = [str(e.get("predictor", "")).strip() for e in evidence_list if not _is_not_reported(e.get("predictor"))]
        outcomes = [str(e.get("outcome_definition", "")).strip() for e in evidence_list if not _is_not_reported(e.get("outcome_definition"))]
        top_pred = Counter(predictors).most_common(1)[0][0] if predictors else "Not reported"
        top_outcome = Counter(outcomes).most_common(1)[0][0] if outcomes else "Not reported"
        st.info(f"Most frequent predictor: **{top_pred}** · Most frequent outcome: **{top_outcome}**")
    elif uc == 2:
        reproducible = sum(1 for e in evidence_list if bool(e.get("assignment_is_reproducible", False)))
        total = max(len(evidence_list), 1)
        st.info(f"Reproducible phenotype assignment in **{reproducible}/{total}** study row(s).")
    else:
        biom = [str(e.get("biomarker_name", e.get("biomarker_or_score", ""))).strip() for e in evidence_list if not _is_not_reported(e.get("biomarker_name", e.get("biomarker_or_score")))]
        top_biomarker = Counter(biom).most_common(1)[0][0] if biom else "Not reported"
        st.info(f"Most recurrent biomarker/score across extracted evidence: **{top_biomarker}**")


def _render_citation_integrity(evidence_list: list[dict]) -> None:
    quotes = []
    weak_quotes = 0
    for e in evidence_list:
        quote = ((e.get("source") or {}).get("quote", "") or "").strip()
        if _is_not_reported(quote):
            continue
        if len(quote) < 20:
            weak_quotes += 1
        quotes.append(" ".join(quote.lower().split()))
    dup_count = len(quotes) - len(set(quotes))
    if dup_count or weak_quotes:
        st.markdown("#### 🔒 Citation Integrity Check")
        if dup_count:
            st.warning(f"⚠️ {dup_count} duplicate supporting quote(s) detected across rows.")
        if weak_quotes:
            st.warning(f"⚠️ {weak_quotes} short/weak supporting quote(s) detected.")


def _render_missingness(df: pd.DataFrame) -> None:
    if df.empty:
        return
    miss = []
    for col in df.columns:
        vals = df[col]
        missing = sum(1 for v in vals if _is_not_reported(v))
        miss.append(
            {
                "Field": col,
                "Missing %": round((missing / max(len(df), 1)) * 100, 1),
            }
        )
    miss_df = pd.DataFrame(miss).sort_values("Missing %", ascending=False)
    with st.expander("📉 Missingness Overview", expanded=False):
        st.dataframe(miss_df, use_container_width=True, hide_index=True)


def _filter_df(df: pd.DataFrame, uc: int) -> pd.DataFrame:
    if df.empty:
        return df
    st.markdown("#### 🎯 Population Slice Filters")
    filtered = df.copy()
    c1, c2 = st.columns(2)
    if "Setting" in filtered.columns:
        settings = sorted([s for s in filtered["Setting"].dropna().unique() if not _is_not_reported(s)])
        selected = c1.multiselect("Setting", options=settings, key=f"filter_setting_uc{uc}")
        if selected:
            filtered = filtered[filtered["Setting"].isin(selected)]
    if "Population" in filtered.columns:
        populations = sorted([p for p in filtered["Population"].dropna().unique() if not _is_not_reported(p)])
        selected = c2.multiselect("Population", options=populations, key=f"filter_population_uc{uc}")
        if selected:
            filtered = filtered[filtered["Population"].isin(selected)]
    return filtered

def _status_badge(status: str) -> str:
    badges = {
        "verified": "✅ verified",
        "unverified": "⚠️ unverified",
        "not_applicable": "ℹ️ N/A",
    }
    return badges.get(status, status)


def _display_uc1(evidence_list: list) -> None:
    """Render Use-Case-1 evidence as a table + expandable source rows."""
    rows = []
    for e in evidence_list:
        rows.append(
            {
                "Study": e.get("study_name", "N/A"),
                "Population": e.get("population", "N/A"),
                "N": e.get("sample_size", "N/A"),
                "Setting": e.get("setting", "N/A"),
                "Predictor": e.get("predictor", "N/A"),
                "Outcome": e.get("outcome_definition", "N/A"),
                "Timing": e.get("timing", "N/A"),
                "Method": e.get("statistical_method", "N/A"),
                "Effect Size": e.get("effect_size", "N/A"),
                "Performance": e.get("performance_metrics", "N/A"),
                "Notes": e.get("notes", "N/A"),
                "Confidence": e.get("_confidence_score", 0),
            }
        )

    df = pd.DataFrame(rows)
    df = _filter_df(df, uc=1)
    st.dataframe(df.sort_values(by="Confidence", ascending=False), use_container_width=True, hide_index=True)
    _render_missingness(df)
    _display_source_evidence(evidence_list)


def _display_uc2(evidence_list: list) -> None:
    """Render Use-Case-2 evidence: study-level table + phenotype table."""

    def _fmt_variables(value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value) if value else "Not reported"
        return str(value or "Not reported")

    # --- Study-level table ---
    study_rows = []
    for e in evidence_list:
        study_rows.append(
            {
                "Study": e.get("study_name", "N/A"),
                "Country": e.get("country", "N/A"),
                "Setting": e.get("setting", "N/A"),
                "N": e.get("sample_size", "N/A"),
                "Sepsis Def.": e.get("sepsis_definition", "N/A"),
                "Method": e.get("clustering_method", "N/A"),
                "Clusters": e.get("num_clusters", "N/A"),
                "Variables": _fmt_variables(e.get("variables_used", "N/A")),
                "Assignment": e.get("assignment_feasibility", "N/A"),
                "Reproducible?": e.get("assignment_is_reproducible", False),
                "Assignment Notes": e.get("assignment_notes", "N/A"),
                "Reproducibility Notes": e.get("reproducibility_notes", "N/A"),
                "Confidence": e.get("_confidence_score", 0),
            }
        )

    study_df = pd.DataFrame(study_rows)
    study_df = _filter_df(study_df, uc=2)
    st.markdown("#### 📋 Study-Level Summary")
    st.dataframe(
        study_df.sort_values(by="Confidence", ascending=False), use_container_width=True, hide_index=True
    )
    _render_missingness(study_df)
    reproducible = int(study_df["Reproducible?"].sum()) if not study_df.empty else 0
    badge_icon = "🟢" if reproducible else "🟡"
    st.markdown(
        f"{badge_icon} **Reproducibility:** {'Strong' if reproducible else 'Limited'} "
        f"— {reproducible} reproducible study row(s)"
    )

    # --- Phenotype table ---
    pheno_rows = []
    for e in evidence_list:
        study = e.get("study_name", "N/A")
        for ph in e.get("phenotypes", []):
            pheno_rows.append(
                {
                    "Study": study,
                    "Cluster": ph.get("cluster_name", ph.get("cluster_id", "N/A")),
                    "Cluster Size": ph.get("cluster_size", "N/A"),
                    "Key Features": ph.get("key_features", "N/A"),
                    "Description": ph.get("clinical_description", "N/A"),
                    "Outcome": ph.get("outcomes", ph.get("outcome", "N/A")),
                    "Notes": ph.get("notes", "N/A"),
                    "Confidence": e.get("_confidence_score", 0),
                }
            )

    if pheno_rows:
        st.markdown("#### 🧬 Phenotype (Cluster-Level) Table")
        pheno_df = pd.DataFrame(pheno_rows)
        st.dataframe(
            pheno_df.sort_values(by="Confidence", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No cluster-level rows were extracted for this query.")

    _display_source_evidence(evidence_list)


def _display_uc3(evidence_list: list) -> None:
    """Render Use-Case-3 evidence as detailed + ranking tables + source rows."""
    rows = []
    for e in evidence_list:
        rows.append(
            {
                "Study": e.get("study_name", "N/A"),
                "Biomarker / Score": e.get("biomarker_name", e.get("biomarker_or_score", "N/A")),
                "Type": e.get("biomarker_type", "N/A"),
                "Setting": e.get("cohort_setting", "N/A"),
                "Population": e.get("population", "N/A"),
                "N": e.get("sample_size", "N/A"),
                "Outcome": e.get("outcome", "N/A"),
                "Effect Size": e.get("effect_size", "N/A"),
                "AUC": e.get("auc", "N/A"),
                "AUROC (numeric)": e.get("auroc", None),
                "95% CI": e.get("confidence_interval", "N/A"),
                "Adjustment": e.get("adjustment", "N/A"),
                "Method": e.get("statistical_method", "N/A"),
                "Validation": e.get("validation_method", "N/A"),
                "Target Pop. Relevance": e.get("relevance_to_target_population", "N/A"),
                "Cohort": e.get("cohort_characteristics", "N/A"),
                "Actionability": (
                    "risk_stratification"
                    if "mortality" in str(e.get("outcome", "")).lower()
                    else "diagnostic"
                ),
                "Notes": e.get("notes", "N/A"),
                "Confidence": e.get("_confidence_score", 0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No UC3 rows were extracted.")
        return

    df["AUROC (numeric)"] = pd.to_numeric(df["AUROC (numeric)"], errors="coerce")

    df = _filter_df(df, uc=3)
    st.markdown("#### 📊 Study-Level Comparison Table")
    st.dataframe(
        df.sort_values(by=["Confidence", "AUROC (numeric)"], ascending=[False, False], na_position="last"),
        use_container_width=True,
        hide_index=True,
    )
    _render_missingness(df)

    ranking = (
        df.groupby(["Biomarker / Score", "Setting"], dropna=False, as_index=False)
        .agg(
            Best_AUROC=("AUROC (numeric)", "max"),
            Studies=("Study", "nunique"),
            Example_CI=("95% CI", "first"),
            Validation=("Validation", "first"),
            Avg_Confidence=("Confidence", "mean"),
        )
        .sort_values(by="Best_AUROC", ascending=False, na_position="last")
    )

    st.markdown("#### 🏆 Biomarker Ranking Table")
    ranking["Avg_Confidence"] = ranking["Avg_Confidence"].round(1)
    st.dataframe(ranking, use_container_width=True, hide_index=True)
    _display_source_evidence(evidence_list)


def _display_source_evidence(evidence_list: list) -> None:
    """Show per-row expandable source evidence blocks."""
    st.markdown("#### 📎 Source Evidence & Provenance")
    for i, e in enumerate(evidence_list, 1):
        study = e.get("study_name", f"Evidence {i}")
        source = e.get("source", {}) or {}
        validation = e.get("_validation", {}) or {}
        schema_err = e.get("_schema_error")
        confidence = e.get("_confidence_score", 0)

        badge = _status_badge(validation.get("status", ""))
        # Color-coded confidence indicator
        conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 45 else "🔴"
        with st.expander(f"{conf_color} {study}  —  {badge}  ({confidence}/100)"):
            col_left, col_right = st.columns([2, 3])

            with col_left:
                page = source.get("page_number", "N/A")
                st.markdown(f"📖 **Page:** {page}")
                st.markdown(f"🔍 **Verification:** {badge}")
                st.markdown(f"📊 **Confidence:** `{confidence}/100`")
                section = source.get("section_name", "Not reported")
                if section and section != "Not reported":
                    st.markdown(f"📑 **Section:** {section}")
                origin = source.get("evidence_origin", "raw_text")
                if origin != "raw_text":
                    st.markdown(f"🔬 **Origin:** {origin}")
                if schema_err:
                    st.warning(f"Schema validation issue: {schema_err}")
                if validation.get("warning"):
                    st.warning(validation["warning"])
                explain = e.get("_confidence_explain")
                if explain:
                    st.caption(f"💡 {explain}")

            with col_right:
                st.markdown("**📝 Supporting quote:**")
                quote = source.get("quote", "Not reported")
                if quote and quote.lower() != "not reported":
                    st.markdown(
                        f'<div style="background: rgba(99, 102, 241, 0.06); '
                        f'border-left: 3px solid #6366f1; padding: 0.8rem 1rem; '
                        f'border-radius: 0 8px 8px 0; font-style: italic; '
                        f'color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">'
                        f'"{html.escape(quote)}"</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No supporting quote available.")


_DISPLAY_FNS = {1: _display_uc1, 2: _display_uc2, 3: _display_uc3}


def _expand_query(use_case: int, query: str) -> str:
    """Boost retrieval with use-case-specific terms."""
    query = query.strip()
    if use_case == 2:
        boosts = (
            "phenotype phenotypes cluster clustering latent class subtype "
            "endotype unsupervised k-means hierarchical mixture model"
        )
        return f"{query}\n\nRelated terms: {boosts}"
    if use_case == 3:
        boosts = (
            "biomarker score AUROC AUC ROC sensitivity specificity hazard ratio "
            "odds ratio SOFA SAPS II APACHE lactate procalcitonin IL-6"
        )
        return f"{query}\n\nRelated terms: {boosts}"
    return query


def _uc2_keyword_chunks(chunks: list) -> list:
    """Return UC2 keyword-focused subset from retrieved chunks."""
    keywords = tuple(k.lower() for k in UC2_KEYWORDS)
    focused = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).lower()
        if any(k in text for k in keywords):
            focused.append(chunk)
    return focused


def _extract_with_fallbacks(
    *,
    uc: int,
    query: str,
    chunks: list,
    api_key: str,
    model: str,
) -> list:
    """Run primary extraction and apply UC2 fallback passes when needed."""
    evidence = extract_evidence(
        query=query,
        chunks=chunks,
        use_case=uc,
        api_key=api_key,
        model=model,
        base_url=config.OPENROUTER_BASE_URL,
    )
    if evidence or uc != 2:
        return evidence

    fallback_sets = []
    focused = _uc2_keyword_chunks(chunks)
    if focused:
        fallback_sets.append(focused[:60])
    fallback_sets.append(chunks[:40])

    for subset in fallback_sets:
        if not subset:
            continue
        evidence = extract_evidence(
            query=query,
            chunks=subset,
            use_case=uc,
            api_key=api_key,
            model=model,
            base_url=config.OPENROUTER_BASE_URL,
        )
        if evidence:
            return evidence

    return []


def _run_adaptive_pipeline(
    *,
    uc: int,
    effective_query: str,
    api_key: str,
    model: str,
) -> tuple[list, list, dict]:
    boosted_query = _expand_query(uc, effective_query)
    keyword_fallback = UC2_KEYWORDS if uc == 2 else UC3_KEYWORDS if uc == 3 else None
    base_k = min(config.TOP_K_CHUNKS, 80) if uc == 2 else config.TOP_K_CHUNKS
    candidate_ks = [base_k, min(base_k * 2, 300), min(base_k * 3, 450)]
    attempts = []
    best_validated = []
    best_chunks = []
    best_score = -1.0

    for idx, top_k in enumerate(candidate_ks, 1):
        chunks = query_chunks_enhanced(
            boosted_query,
            config.CHROMA_PERSIST_DIR,
            n_results=top_k,
            keyword_fallback=keyword_fallback,
        )
        if not chunks:
            attempts.append(
                {
                    "attempt": idx,
                    "top_k": top_k,
                    "metrics": {"evidence_count": 0, "verified_ratio": 0.0, "coverage": 0.0, "avg_confidence": 0.0},
                    "retry_reason": "no_chunks",
                }
            )
            continue

        evidence = _extract_with_fallbacks(
            uc=uc,
            query=effective_query,
            chunks=chunks,
            api_key=api_key,
            model=model,
        )
        validated = _enrich_evidence(validate_all(evidence, chunks)) if evidence else []
        metrics = _attempt_metrics(validated)
        reason = _retry_reason(metrics)
        attempts.append(
            {
                "attempt": idx,
                "top_k": top_k,
                "metrics": metrics,
                "retry_reason": reason or "",
            }
        )

        score = metrics["verified_ratio"] + metrics["coverage"] + metrics["avg_confidence"]
        if score > best_score:
            best_score = score
            best_validated = validated
            best_chunks = chunks

        if not reason:
            break

    return best_validated, best_chunks, {"attempts": attempts}


def _init_query_history_state() -> None:
    st.session_state.setdefault("query_history", [])


def _add_query_history(*, uc: int, query: str, model: str, evidence_count: int, verified_count: int) -> None:
    _init_query_history_state()
    history = st.session_state.get("query_history", [])
    history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "use_case": uc,
            "query": query.strip(),
            "model": model,
            "evidence_count": evidence_count,
            "verified_count": verified_count,
        },
    )
    st.session_state["query_history"] = history[:20]


def _render_query_history(uc: int) -> None:
    _init_query_history_state()
    entries = [h for h in st.session_state["query_history"] if h.get("use_case") == uc]
    if not entries:
        return
    with st.expander("Query History (session)", expanded=False):
        for i, h in enumerate(entries[:8], 1):
            st.markdown(
                f"{i}. `{h['timestamp']}` · {h['evidence_count']} row(s), "
                f"{h['verified_count']} verified · `{h['query']}`"
            )


def _render_demo_scenarios(uc: int) -> None:
    scenarios = {
        1: [
            "What is the relationship between initial lactate and 28-day mortality in septic shock?",
            "Which admission predictors are linked to in-hospital mortality in sepsis?",
        ],
        2: [
            "What sepsis phenotypes were identified using unsupervised clustering?",
            "Are phenotype assignment rules reproducible across studies?",
        ],
        3: [
            "Which biomarkers or scores best predict 28-day mortality in sepsis?",
            "Compare SOFA, lactate, and IL-6 for mortality risk stratification.",
        ],
    }
    with st.expander("One-click Demo Scenarios", expanded=False):
        for idx, q in enumerate(scenarios.get(uc, [])):
            if st.button(f"Run scenario {idx + 1}", key=f"demo_uc{uc}_{idx}", use_container_width=True):
                st.session_state[f"query_uc{uc}"] = q
                st.rerun()


def _generate_submission_brief(uc: int, query: str, validated: list[dict], model: str) -> str:
    verified = sum(1 for e in validated if (e.get("_validation", {}) or {}).get("status") == "verified")
    avg_conf = int(round(sum(int(e.get("_confidence_score", 0)) for e in validated) / max(len(validated), 1)))
    lines = [
        "# Sepsis Atlas — Submission Brief",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Use case: UC{uc} — {USE_CASES[uc]['name']}",
        f"- Model: {model}",
        f"- Query: {query}",
        f"- Evidence rows: {len(validated)}",
        f"- Verified rows: {verified}",
        f"- Average confidence: {avg_conf}/100",
        "",
        "## Key evidence rows",
    ]
    for i, e in enumerate(validated[:8], 1):
        study = e.get("study_name", "N/A")
        source = e.get("source", {}) or {}
        quote = source.get("quote", "Not reported")
        lines.append(f"{i}. **{study}** — page {source.get('page_number', 'N/A')} — {quote}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_scrollable_logs(container, logs: list[str], height_px: int = 220) -> None:
    """Render multi-line logs in a vertically scrollable box."""
    if not logs:
        return
    escaped = html.escape("\n".join(logs))
    container.markdown(
        (
            f"<div style='max-height:{height_px}px; overflow-y:auto; "
            "border:1px solid rgba(99, 102, 241, 0.2); border-radius:10px; padding:12px; "
            "background:rgba(15, 23, 42, 0.6);'>"
            "<pre style='margin:0; white-space:pre-wrap; color:#e2e8f0; "
            "font-family: \"JetBrains Mono\", \"Fira Code\", monospace; "
            f"font-size: 0.78rem; line-height: 1.6;'>{escaped}</pre>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _init_indexing_ui_state() -> None:
    st.session_state.setdefault("indexing_logs", [])
    st.session_state.setdefault("indexing_summary_lines", [])
    st.session_state.setdefault("indexing_status", "")


def _render_last_indexing_output() -> None:
    _init_indexing_ui_state()
    logs: list[str] = st.session_state.get("indexing_logs", [])
    summary_lines: list[str] = st.session_state.get("indexing_summary_lines", [])
    status = st.session_state.get("indexing_status", "")

    if not logs and not summary_lines:
        return

    with st.expander("Latest indexing run logs", expanded=True):
        _render_scrollable_logs(st, logs, height_px=220)
        if summary_lines:
            summary_text = "\n".join(summary_lines)
            if status == "success":
                st.success(summary_text)
            elif status == "warning":
                st.warning(summary_text)
            else:
                st.info(summary_text)


def render_sidebar() -> tuple[str, str]:
    """Render sidebar controls; returns (api_key, model)."""
    with st.sidebar:
        _init_indexing_ui_state()
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <span style="font-size: 2.5rem;">🏥</span>
                <h2 style="margin: 0.3rem 0 0 0; font-size: 1.4rem; 
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; font-weight: 700;">
                    Sepsis Atlas
                </h2>
                <p style="margin: 0.2rem 0; font-size: 0.8rem; color: #94a3b8;">
                    AI-powered clinical evidence extractor
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # --- API configuration ---
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "OpenRouter API Key",
            value=config.OPENROUTER_API_KEY,
            type="password",
            help=(
                "Obtain a key at https://openrouter.ai/.  "
                "You can also set OPENROUTER_API_KEY in a .env file."
            ),
        )

        default_model_idx = (
            MODELS.index(config.DEFAULT_MODEL)
            if config.DEFAULT_MODEL in MODELS
            else 0
        )
        model = st.selectbox(
            "LLM Model",
            options=MODELS,
            index=default_model_idx,
            help="Free-tier models are marked ':free'.",
        )

        st.divider()

        # --- Index management ---
        st.header("📚 Document Index")

        n_chunks = get_collection_count(config.CHROMA_PERSIST_DIR)
        sources = get_indexed_sources(config.CHROMA_PERSIST_DIR)

        if sources:
            st.success(f"✅ {len(sources)} paper(s) indexed · {n_chunks} chunks")
            with st.expander("Indexed papers"):
                for s in sources:
                    st.markdown(f"- `{s}`")
        else:
            st.warning("⚠️ No papers indexed yet.")

        pdf_path = Path(config.PDF_DIR)
        pdf_files = list(pdf_path.glob("*.pdf"))

        if pdf_files:
            st.info(f"Found **{len(pdf_files)}** PDF(s) in `{config.PDF_DIR}`")

            col_idx, col_clr = st.columns(2)
            with col_idx:
                if st.button("🔄 Index PDFs", type="primary", use_container_width=True):
                    _do_index(pdf_files)
            with col_clr:
                if st.button("🗑️ Clear Index", use_container_width=True):
                    clear_collection(config.CHROMA_PERSIST_DIR)
                    st.success("Index cleared.")
                    st.rerun()
        else:
            st.error(
                f"No PDFs found in `{config.PDF_DIR}/`.  "
                "Add PDF files to that folder, then click **Index PDFs**."
            )

        _render_last_indexing_output()
        st.divider()
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0;">
                <p style="font-size: 0.72rem; color: #64748b; margin: 0;">
                    Built for the <b>Sepsis Atlas Hackathon</b><br>
                    <a href="https://github.com/Adyansh04/sepsis-atlas-hackathon" 
                       style="color: #818cf8; text-decoration: none;">
                        🔗 GitHub Repository
                    </a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return api_key, model


def _do_index(pdf_files: list) -> None:
    """Run PDF ingestion and indexing with a progress indicator."""
    _init_indexing_ui_state()
    st.session_state["indexing_logs"] = []
    st.session_state["indexing_summary_lines"] = []
    st.session_state["indexing_status"] = "running"

    progress = st.sidebar.progress(0, text="Preparing indexing…")
    log_placeholder = st.sidebar.empty()
    summary_placeholder = st.sidebar.empty()

    def add_log(message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs: list[str] = st.session_state["indexing_logs"]
        logs.append(f"[{timestamp}] {message}")
        st.session_state["indexing_logs"] = logs
        _render_scrollable_logs(log_placeholder, logs, height_px=220)

    def handle_ingest_progress(event: dict) -> None:
        kind = event.get("event")
        document = event.get("document", "unknown")
        index = int(event.get("index", 0) or 0)
        total = max(int(event.get("total", len(pdf_files)) or len(pdf_files)), 1)
        progress_value = min(70, int((index / total) * 70)) if index else 5

        if kind == "document_started":
            add_log(f"Started document {index}/{total}: {document}")
        elif kind == "document_completed":
            add_log(
                f"Completed document {index}/{total}: {document} · "
                f"{event.get('pages', 0)} pages · "
                f"{event.get('parents', 0)} parents · "
                f"{event.get('chunks', 0)} chunks"
            )
            progress.progress(progress_value, text=f"Parsing PDFs… ({index}/{total})")
        elif kind == "document_failed":
            add_log(
                f"Failed document {index}/{total}: {document} · "
                f"{event.get('error', 'unknown error')}"
            )
            progress.progress(progress_value, text=f"Parsing PDFs… ({index}/{total})")

    add_log(f"Starting indexing for {len(pdf_files)} PDF(s)")
    progress.progress(5, text="Parsing PDFs…")
    chunks, report = ingest_pdfs(
        config.PDF_DIR,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        progress_callback=handle_ingest_progress,
        return_report=True,
    )
    progress.progress(80, text="Building vector index…")
    add_log(f"Building vector index for {len(chunks)} chunk(s)")
    if chunks:
        index_chunks(chunks, config.CHROMA_PERSIST_DIR)
        add_log("Vector index build completed")
        progress.progress(100, text="Done!")
        summary_lines = [
            f"Documents completed: {report['documents_completed']}/{report['documents_discovered']}",
            f"Documents failed: {report['documents_failed']}",
            f"Pages extracted: {report['pages_extracted']}",
            f"Tables processed: {report['tables_processed']} (parse failures: {report['table_parse_failures']})",
            f"Images processed: {report['images_processed']} (parse failures: {report['image_parse_failures']})",
            (
                "VLM summaries — "
                f"success: {report['vlm_successes']}, "
                f"failed: {report['vlm_failures']}, "
                f"unavailable: {report['vlm_unavailable']}, "
                f"empty: {report['vlm_empty']}"
            ),
            f"Parents created: {report['parents_created']}",
            f"Chunks created: {report['chunks_created']}",
        ]
        if report.get("fallback_used"):
            summary_lines.append("Legacy fallback chunking was used.")
        st.session_state["indexing_summary_lines"] = summary_lines
        st.session_state["indexing_status"] = "success"
        summary_placeholder.info("\n".join(summary_lines))
        st.sidebar.success(
            f"Indexed {len(chunks)} chunks from {len(pdf_files)} PDF(s)."
        )
    else:
        progress.empty()
        summary_lines = [
            f"Documents completed: {report['documents_completed']}/{report['documents_discovered']}",
            f"Documents failed: {report['documents_failed']}",
            f"Table parse failures: {report['table_parse_failures']}",
            f"Image parse failures: {report['image_parse_failures']}",
            f"VLM failures: {report['vlm_failures']}",
        ]
        st.session_state["indexing_summary_lines"] = summary_lines
        st.session_state["indexing_status"] = "warning"
        summary_placeholder.warning("\n".join(summary_lines))
        st.sidebar.warning("No text could be extracted from the PDFs.")
    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key, model = render_sidebar()

    # --- Header ---
    st.markdown("# 🏥 Sepsis Atlas")
    st.markdown(
        '<p class="hero-subtitle">Transform published sepsis research into structured, '
        "source-grounded evidence tables.</p>",
        unsafe_allow_html=True,
    )

    # Quick status bar
    n_docs = len(get_indexed_sources(config.CHROMA_PERSIST_DIR))
    n_chunks = get_collection_count(config.CHROMA_PERSIST_DIR)
    status_cols = st.columns(4)
    with status_cols[0]:
        st.markdown(f"📄 **{n_docs}** papers indexed")
    with status_cols[1]:
        st.markdown(f"🧩 **{n_chunks}** chunks")
    with status_cols[2]:
        st.markdown(f"🤖 `{config.DEFAULT_MODEL.split('/')[-1]}`")
    with status_cols[3]:
        st.markdown(f"🔑 {'✅ Key set' if config.OPENROUTER_API_KEY else '❌ No key'}")
    st.divider()

    # --- Use-case selector ---
    st.subheader("1 · Select a Use Case")
    uc_labels = [f"UC{k} — {v['name']}" for k, v in USE_CASES.items()]
    uc_tab1, uc_tab2, uc_tab3 = st.tabs(uc_labels)

    tabs = {1: uc_tab1, 2: uc_tab2, 3: uc_tab3}

    for uc, tab in tabs.items():
        with tab:
            _render_use_case_tab(uc, api_key, model)


def _render_use_case_tab(uc: int, api_key: str, model: str) -> None:
    """Render the query + results UI for a single use-case tab."""
    info = USE_CASES[uc]
    st.markdown(
        f'<p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem;">'
        f'{info["description"]}</p>',
        unsafe_allow_html=True,
    )
    _render_query_history(uc)
    _render_demo_scenarios(uc)

    # --- Query input ---
    st.subheader("2 · Enter Query")
    col_query, col_btn = st.columns([5, 1])

    with col_query:
        query = st.text_input(
            "Query",
            placeholder=info["example_query"],
            label_visibility="collapsed",
            key=f"query_uc{uc}",
        )

    with col_btn:
        if st.button("💡 Example", key=f"example_uc{uc}", use_container_width=True):
            st.session_state[f"query_uc{uc}"] = info["example_query"]
            st.rerun()

    run = st.button(
        "🔍 Extract Evidence",
        type="primary",
        key=f"run_uc{uc}",
    )

    if not run:
        return

    # --- Validation ---
    if not api_key:
        st.error("❌ Please provide an OpenRouter API Key in the sidebar.")
        return

    effective_query = st.session_state.get(f"query_uc{uc}", query) or query
    if not effective_query.strip():
        st.warning("⚠️ Please enter a query.")
        return

    if get_collection_count(config.CHROMA_PERSIST_DIR) == 0:
        st.error(
            "❌ No documents indexed. Add PDF files to "
            f"`{config.PDF_DIR}/` and click **Index PDFs** in the sidebar."
        )
        return

    # --- Pipeline ---
    st.subheader("3 · Results")

    with st.spinner("Retrieving and extracting (adaptive mode)…"):
        try:
            validated, chunks, retry_report = _run_adaptive_pipeline(
                uc=uc,
                effective_query=effective_query,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM extraction failed: {exc}")
            return

    if not chunks:
        st.warning("No relevant passages found for this query.")
        return

    with st.expander(f"📑 Retrieved {len(chunks)} passage(s) — click to inspect", expanded=False):
        for i, chunk in enumerate(chunks[:20], 1):
            st.markdown(f"**Excerpt {i}** — `{chunk['source']}` · page {chunk['page_number']}")
            st.code(chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else ""), language=None)
        if len(chunks) > 20:
            st.caption(f"… and {len(chunks) - 20} more passages.")

    if not validated:
        st.warning(
            "The LLM returned no structured evidence for this query.  "
            "Try rephrasing or check that the PDFs contain relevant content."
        )
        return

    n_verified = sum(
        1
        for e in validated
        if e.get("_validation", {}).get("status") == "verified"
    )
    st.success(
        f"✅ Extracted **{len(validated)}** evidence item(s) — "
        f"**{n_verified}** quote-verified."
    )
    _add_query_history(
        uc=uc,
        query=effective_query,
        model=model,
        evidence_count=len(validated),
        verified_count=n_verified,
    )

    _render_retry_metrics(retry_report)
    _render_trust_dashboard(validated, uc)
    _render_backend_graph_outputs(uc, validated)

    _DISPLAY_FNS[uc](validated)

    # --- Download ---
    st.markdown("#### 💾 Export Results")
    try:
        flat = []
        for e in validated:
            row = {k: v for k, v in e.items() if k not in ("source", "_validation", "_schema_error", "phenotypes")}
            src = e.get("source") or {}
            row["source_page"] = src.get("page_number", "")
            row["source_quote"] = src.get("quote", "")
            val = e.get("_validation") or {}
            row["verification_status"] = val.get("status", "")
            flat.append(row)

        csv_data = pd.DataFrame(flat).to_csv(index=False)
        json_data = json.dumps(validated, indent=2, ensure_ascii=False)
        brief = _generate_submission_brief(uc, effective_query, validated, model)

        dl_cols = st.columns(3)
        with dl_cols[0]:
            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name=f"sepsis_atlas_uc{uc}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "🗂️ Download JSON",
                data=json_data,
                file_name=f"sepsis_atlas_uc{uc}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_cols[2]:
            st.download_button(
                "📋 Submission Brief",
                data=brief,
                file_name=f"sepsis_atlas_uc{uc}_brief.md",
                mime="text/markdown",
                use_container_width=True,
            )
    except Exception:  # noqa: BLE001
        pass

    st.divider()
    st.caption(
        "🔗 Data provenance · "
        f"model: `{model}` · "
        f"indexed_docs: `{len(get_indexed_sources(config.CHROMA_PERSIST_DIR))}` · "
        f"retrieved_chunks: `{len(chunks)}` · "
        f"generated_at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    )


if __name__ == "__main__":
    main()
