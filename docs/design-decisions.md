# Design Decisions

This document captures the major design points, why they exist, and what project problems they solve.

## 1. Main project goals

The system is designed to turn sepsis papers into:

- structured
- source-grounded
- analysis-ready
- exportable

evidence tables rather than free-form summaries.

The core goals are:

1. preserve traceability back to the source paper
2. avoid unsupported hallucinated values
3. support multiple hackathon use cases through shared infrastructure
4. remain local-first for indexing and retrieval
5. provide enough reliability signals that users can inspect failures

## 2. Foundational design choices

| Decision | Why it was chosen |
|---|---|
| Streamlit UI | Fast iteration and easy demo workflow |
| ChromaDB | Local persistence, simple setup, no separate service |
| Pydantic schemas | Strong structured contracts for each use case |
| OpenRouter | Unified interface for multiple models and hackathon compatibility |
| Section-aware ingestion | Better context quality than flat raw-page indexing |
| Quote verification | Explicit provenance check after extraction |

## 3. Why the pipeline became more agentic

The earlier/simple version of the pipeline had a classic failure mode:

- retrieval returns partially relevant chunks
- extraction produces sparse rows
- some rows are weakly grounded
- users cannot easily see quality issues

To address that, the pipeline now behaves more like an agentic workflow with self-checking and repair stages.

## 4. Implemented reliability improvements

### Adaptive retrieval retries

Problem:

- a single fixed Top-K often under-retrieves or over-focuses on one paper

Implemented solution:

- try multiple Top-K levels
- score each attempt by:
  - verified quote ratio
  - schema coverage
  - average confidence
- stop early once quality is acceptable

Why this matters:

- retrieval quality becomes observable
- the system can recover from weak first-pass recall

### Confidence scoring

Problem:

- users need a fast way to judge row quality

Implemented solution:

- compute a composite score using:
  - validation status
  - quote presence/quality
  - field completeness

Why this matters:

- weak rows become visible instead of silently mixed with strong rows

### Schema-aware active repair

Problem:

- models often return mostly-correct rows with a few blank required fields

Implemented solution:

- detect missing required fields per use case
- ask only for missing fields
- patch only those fields

Why this matters:

- preserves already-correct content
- reduces unnecessary regeneration drift

### Contradiction graph

Problem:

- studies can disagree, and a flat table hides those disagreements

Implemented solution:

- group rows by comparable subject/outcome pairs
- flag opposite effect directions
- flag conflicting cutoffs when detectable

Why this matters:

- reviewers can spot conflicts immediately

### Evidence knowledge graph

Problem:

- evidence tables are hard to scan when many studies and predictors are involved

Implemented solution:

- map rows into entities and relations
- summarize the strongest evidence path

Why this matters:

- improves interpretability of the extracted evidence landscape

## 5. Errors and issues encountered during development

This section documents important operational issues that influenced the current design and docs.

| Issue | Observed effect | Outcome / solution |
|---|---|---|
| Missing local Python dependencies | tests could not run initially | install required packages before running the core suite |
| `tests/test_openrouter.py` import-time key requirement | full `tests/` collection can fail without `.env` key | recommend `python -m pytest tests/test_pipeline.py -v` as core local validation |
| Weak first-pass retrieval | sparse or low-trust outputs | adaptive Top-K retries |
| Missing required fields in otherwise good rows | incomplete tables | targeted schema-aware repair |
| Conflicting study claims hidden in flat tables | reviewers miss disagreement patterns | contradiction graph output |
| Evidence importance hard to scan | difficult prioritization | knowledge graph strongest path summary |

## 6. Non-goals

The current design intentionally does **not** try to:

- fully reconcile statistical heterogeneity across studies
- replace manual clinical review
- build a production graph database
- guarantee perfect OCR or table extraction
- run as a multi-user backend service

## 7. Extension path

The architecture was kept modular so future contributors can add:

- new use cases by defining a new schema + prompt
- alternate retrieval models
- stronger contradiction heuristics
- persisted query/result history
- evaluator modules for benchmark scoring
