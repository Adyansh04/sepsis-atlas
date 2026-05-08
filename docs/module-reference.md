# Module Reference

This file is a practical map of what every important repository file does.

## Root files

### `app.py`

Main Streamlit entrypoint and pipeline orchestrator.

Key responsibilities:

- layout and styling
- use-case selection
- indexing controls
- adaptive retrieval loop
- query history
- trust dashboard
- contradiction graph and KG rendering
- exports

Important internal functions:

- `_run_adaptive_pipeline(uc, effective_query, api_key, model)` — multi-attempt retrieval + extraction loop
- `_attempt_metrics(validated)` — computes evidence_count, verified_ratio, coverage, avg_confidence
- `_retry_reason(metrics)` — returns retry reason string or None if quality is acceptable
- `_compute_confidence(evidence)` — composite score from status + quote quality + field completeness
- `_enrich_evidence(evidence_list)` — adds `_confidence_score` and `_confidence_explain` to each row
- `_expand_query(use_case, query)` — appends use-case-specific retrieval boost terms
- `_extract_with_fallbacks(uc, query, chunks, api_key, model)` — primary extraction + UC2 multi-pass fallback
- `_uc2_keyword_chunks(chunks)` — filters chunks to those containing UC2 keywords
- `_render_retry_metrics(report)` — displays adaptive retrieval attempts table
- `_render_trust_dashboard(evidence_list, uc)` — renders metrics + consensus + citation integrity
- `_render_backend_graph_outputs(uc, validated)` — contradiction graph + knowledge graph
- `_render_consensus_snapshot(uc, evidence_list)` — most frequent predictor/biomarker/reproducibility
- `_render_citation_integrity(evidence_list)` — duplicate/weak quote warnings
- `_render_missingness(df)` — per-column missing percentage table
- `_filter_df(df, uc)` — interactive Setting/Population multiselect filters
- `_display_uc1(evidence_list)` — UC1 evidence table renderer
- `_display_uc2(evidence_list)` — UC2 study-level + phenotype-level table renderer
- `_display_uc3(evidence_list)` — UC3 comparison table + biomarker ranking table renderer
- `_display_source_evidence(evidence_list)` — per-row expandable provenance blocks
- `_generate_submission_brief(uc, query, validated, model)` — markdown export generator
- `_do_index(pdf_files)` — PDF ingestion + indexing with progress UI
- `render_sidebar()` — sidebar layout, returns (api_key, model)

### `config.py`

Central configuration object loaded from environment variables.

Holds:

- API settings
- PDF and Chroma paths
- chunking sizes
- retrieval depth
- OCR and table-extraction toggles

### `requirements.txt`

Dependency manifest for local setup.

### `.env.example`

Template for required runtime configuration.

## Source package

### `src/ingest.py`

Purpose:

- Convert PDFs into retrieval-ready chunks.

Main concepts:

- layout-aware partitioning
- section detection
- parent/child chunk hierarchy
- visual summarization
- indexing progress reporting

Key public functions:

- `chunk_pages(...)`
- `ingest_pdfs(...)`

### `src/retrieval.py`

Purpose:

- Manage ChromaDB and return relevant chunks.

Main concepts:

- persistent local vector store
- semantic search
- reference filtering
- keyword fallback for low-recall cases

Key public functions:

- `index_chunks(...)`
- `query_chunks(...)`
- `query_chunks_enhanced(...)`
- `clear_collection(...)`
- `get_collection_count(...)`
- `get_indexed_sources(...)`

### `src/extraction.py`

Purpose:

- Convert retrieved context into schema-valid structured evidence.

Main concepts:

- strict anti-hallucination system prompt
- per-use-case JSON schemas
- JSON parsing recovery
- normalization helpers
- UC2 salvage path
- schema-aware active repair

Key public functions:

- `extract_evidence(...)`

Important internal helpers:

- `_format_context(chunks)` — formats retrieved chunks into numbered excerpts with metadata headers
- `_parse_json_response(raw)` — handles plain JSON, markdown fences, nested object/array extraction
- `_ensure_source(item)` — guarantees `source` dict exists with all required keys
- `_missing_required_fields(item, use_case)` — identifies which required fields are blank/None
- `_schema_aware_repair(client, model, use_case, query, chunks, rows)` — targeted field-level LLM repair
- `_normalize_uc1(item)` — cross-fills predictor ↔ predictor_variable
- `_normalize_uc2(item)` — converts variables_used to list, infers reproducibility, cross-fills phenotype fields
- `_normalize_uc3(item)` — cross-fills biomarker names, extracts numeric AUROC
- `_salvage_uc2_from_chunks(chunks)` — regex-based fallback when LLM extraction fails for UC2
- `_detect_uc2_method(text)` — pattern-matches clustering method names in text
- `_extract_uc2_num_clusters(text)` — regex extracts cluster count from text

### `src/validation.py`

Purpose:

- Validate provenance of extracted quotes.

Main concepts:

- exact or approximate quote matching
- lightweight verification
- validation status annotation

Key public functions:

- `verify_quote(...)`
- `validate_extraction(...)`
- `validate_all(...)` — now also runs cross-row consistency checks
- `_check_cross_row_consistency(evidence_list)` — flags quote reuse and effect size misattribution across rows

### `src/analytics.py`

Purpose:

- Provide post-extraction reasoning aids.

Main concepts:

- contradiction edges
- typed entities and relations
- strongest evidence path

Key public functions:

- `build_contradiction_graph(use_case, evidence_list)` — groups evidence by (subject, outcome) and flags opposing directions or different cutoffs
- `build_evidence_knowledge_graph(use_case, evidence_list)` — maps rows into typed entities (study, subject, outcome, population) and relations

Important internal helpers:

- `_infer_effect_direction(evidence, use_case)` — classifies effect as `increase_risk`, `decrease_risk`, `phenotype_descriptor`, or `neutral`
- `_extract_cutoff(evidence)` — extracts numeric threshold patterns (e.g. `>4`, `>=2.5`)
- `_subject_outcome(evidence, use_case)` — determines the grouping key per use case
- `_is_not_reported(value)` — checks if a value is missing/null/not-reported
- `_extract_float(text)` — extracts first numeric value from text

### `src/schemas.py`

Purpose:

- Define the structured contracts for every use case.

Main objects:

- `SourceAnchor`
- `UseCase1Evidence`
- `PhenotypeCluster`
- `UseCase2Evidence`
- `UseCase3Evidence`

## Tests

### `tests/test_pipeline.py`

Primary local test suite.

Covers:

- chunking logic
- schema validation
- quote verification
- extraction helpers
- analytics helpers
- config defaults

### `tests/test_openrouter.py`

Not a normal unit test. It is a direct live API connectivity script and raises immediately if `OPENROUTER_API_KEY` is absent.

## PDF/data assets

### `pdfs/`

Contains the paper corpus and example spreadsheet artifacts. The app expects user papers here unless `PDF_DIR` is changed.

## Recommended reading by role

| If you want to understand... | Read... |
|---|---|
| overall system behavior | `README.md`, `docs/architecture.md` |
| detailed runtime flow | `docs/data-flow.md`, `app.py` |
| every button, value, and threshold | `docs/ui-and-features.md` |
| extraction logic | `src/extraction.py`, `src/schemas.py` |
| indexing logic | `src/ingest.py`, `src/retrieval.py` |
| trust/provenance | `src/validation.py`, `app.py` |
| new analytics features | `src/analytics.py`, `app.py` |
| configuration options | `config.py`, `docs/ui-and-features.md` §7 |
| local validation | `tests/test_pipeline.py`, `docs/troubleshooting.md` |
