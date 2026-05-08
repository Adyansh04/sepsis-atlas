# UI & Features Reference

This document describes every interactive element in the Sepsis Atlas application, what each button does, how processing works end-to-end, every computed value and threshold, and all configuration knobs.

---

## 1. Sidebar Controls

### API Key Input

- **Widget:** Password text input
- **Source:** `config.OPENROUTER_API_KEY` or user-typed value
- **Effect:** Used for all LLM extraction calls and VLM visual summarization
- **If missing:** Extraction will fail with "Please provide an OpenRouter API Key"

### Model Selector

- **Widget:** Selectbox dropdown
- **Options:** Hard-coded list in `MODELS`:
  - `anthropic/claude-3.5-sonnet` (default)
  - `openai/gpt-4o`
  - `openai/gpt-4.1`
  - `openai/gpt-4o-mini`
  - `anthropic/claude-3-haiku`
  - `meta-llama/llama-3.1-8b-instruct:free`
  - `google/gemma-2-9b-it:free`
  - `mistralai/mistral-7b-instruct:free`
  - `x-ai/grok-4.20-multi-agent`
  - `anthropic/claude-sonnet-4.6`
  - `anthropic/claude-opus-4.7`
- **Effect:** Selects which LLM model is sent to OpenRouter for extraction

### 🔄 Index PDFs Button

- **Visible when:** PDFs exist in `config.PDF_DIR`
- **Processing:**
  1. Calls `ingest_pdfs()` which parses all PDFs using `unstructured` layout-aware partitioning
  2. Groups text into section-aware parent documents
  3. Splits parents into child chunks
  4. Tables/images are optionally summarized via VLM (`google/gemini-2.5-flash-image`)
  5. Calls `index_chunks()` which upserts all chunks into ChromaDB
  6. Shows real-time progress bar and scrollable log
  7. Displays summary: documents completed/failed, pages, chunks, VLM stats
- **Rerun needed:** Only when PDFs change, not for code changes

### 🗑️ Clear Index Button

- **Visible when:** PDFs exist in `config.PDF_DIR`
- **Processing:** Calls `clear_collection()` which deletes and recreates the ChromaDB collection
- **Effect:** All indexed data is lost; user must re-index

### Index Status Display

- **Shows:** Number of indexed papers, total chunk count, list of indexed source names
- **Updates:** On every page render by calling `get_collection_count()` and `get_indexed_sources()`

---

## 2. Main Page Controls

### Use Case Tabs

- **UC1 — Counterfactual Mortality Estimation:** Extracts predictor-outcome associations
- **UC2 — Sepsis Phenotype Extraction:** Extracts clustering methods and per-cluster descriptions
- **UC3 — Biomarker Selection for Risk Stratification:** Extracts biomarker/score comparisons

### Query Input

- **Widget:** Text input with placeholder showing the example query for the selected UC
- **Key:** `query_uc{1|2|3}` in session state

### 💡 Example Button

- **Effect:** Fills the query input with the pre-defined example query for the active UC
- **Processing:** Sets `st.session_state[f"query_uc{uc}"]` and triggers `st.rerun()`

### Demo Scenarios (Expander)

- **Contains:** 2 pre-written clinical queries per use case
- **Each "Run scenario N" button:** Sets the query in session state and reruns the app

### 🔍 Extract Evidence Button

- **Effect:** Triggers the full adaptive extraction pipeline
- **Pre-checks:**
  1. API key must be set
  2. Query must be non-empty
  3. ChromaDB must have indexed documents
- **Processing:** Calls `_run_adaptive_pipeline()` — see Section 4 below

### Query History (Expander)

- **Shows:** Last 8 queries for the active UC with timestamp, evidence count, verified count
- **Stored in:** `st.session_state["query_history"]` (session-only, max 20 entries)

---

## 3. Results Display

### Retrieved Passages (Expander)

- **Shows:** First 20 retrieved chunks with source name, page number, and first 500 chars
- **Purpose:** Allows user to inspect what the LLM received as context

### Trust Dashboard

Displays 4 metric cards:

| Metric | Formula | Meaning |
|---|---|---|
| Evidence Rows | `len(validated)` | Total structured rows extracted |
| Verified Quotes | `verified / total` | Rows where quote was matched in retrieved text |
| Avg Confidence | `mean(_confidence_score)` | Average composite quality score (0–100) |
| Schema Coverage | `mean(fields_present / total_fields)` | How many schema fields have non-"Not reported" values |

### Consensus Snapshot

- **UC1:** Shows the most frequent predictor and outcome across rows
- **UC2:** Shows how many studies have reproducible phenotype assignment
- **UC3:** Shows the most recurrent biomarker/score

### Citation Integrity Check

- **Triggers when:** Duplicate quotes or short/weak quotes (< 20 chars) are detected
- **Warns about:** Potential over-reliance on a single passage

### Missingness Overview (Collapsible)

- **Shows:** Per-field percentage of "Not reported" values across all rows
- **Sorted:** Highest missing % first

### Population Slice Filters

- **Widgets:** Multiselect for Setting and Population columns
- **Effect:** Filters the evidence DataFrame in real-time

### Contradiction Graph

- **Groups by:** Same (subject, outcome) pair
  - UC1: (predictor, outcome_definition)
  - UC2: (clustering_method, assignment_feasibility)
  - UC3: (biomarker_name, outcome)
- **Detects:**
  - `opposite_effect_direction` — one study shows OR>1, another shows OR<1
  - `different_cutoff` — same predictor but different thresholds (e.g. ">4" vs ">2")

### Evidence Knowledge Graph

- **Entities:** study, subject, outcome, population
- **Relations per row:**
  - `predicts_outcome` (subject → outcome)
  - `measured_in_population` (subject → population)
  - `reported_by_study` (subject → study)
- **Strongest path:** The `predicts_outcome` relation with highest confidence score

### Adaptive Retrieval Metrics Table

- **Shows per attempt:** Top-K used, rows extracted, verified %, coverage %, avg confidence, retry reason

### Export Buttons

| Button | Format | Content |
|---|---|---|
| 📊 Download CSV | `.csv` | Flat evidence table with source_page, source_quote, verification_status |
| 🗂️ Download JSON | `.json` | Full validated evidence list including nested structures |
| 📋 Submission Brief | `.md` | Markdown summary: metadata + top 8 evidence rows with quotes |

### Source Evidence (Per-Row Expanders)

Each row shows:
- Color-coded confidence indicator (🟢 ≥70, 🟡 ≥45, 🔴 <45)
- Source page number
- Verification status badge (✅ verified / ⚠️ unverified / ℹ️ N/A)
- Confidence score with explainability breakdown
- Section name and evidence origin
- Supporting quote in styled blockquote

---

## 4. Adaptive Pipeline — Processing Detail

The main extraction logic in `_run_adaptive_pipeline()`:

### Step 1: Query Expansion

`_expand_query(use_case, query)` appends use-case-specific boost terms:

- **UC1:** No expansion (raw query passed through)
- **UC2:** Appends: `phenotype phenotypes cluster clustering latent class subtype endotype unsupervised k-means hierarchical mixture model`
- **UC3:** Appends: `biomarker score AUROC AUC ROC sensitivity specificity hazard ratio odds ratio SOFA SAPS II APACHE lactate procalcitonin IL-6`

### Step 2: Retrieval with Escalating Top-K

```
candidate_ks = [base_k, base_k * 2, base_k * 3]
```

- **UC2:** `base_k = min(TOP_K_CHUNKS, 80)` (capped at 80 for UC2)
- **UC1/UC3:** `base_k = TOP_K_CHUNKS` (default 150)
- **Max values:** 300 for attempt 2, 450 for attempt 3

Each attempt calls `query_chunks_enhanced()` which:
1. Runs semantic search via ChromaDB
2. Filters reference/bibliography chunks
3. Merges keyword fallback hits if semantic results < n_results

### Step 3: Extraction with Fallbacks

`_extract_with_fallbacks()` runs the primary LLM extraction call. For UC2 specifically, if the primary call returns empty:
1. Tries UC2-keyword-focused chunk subset (max 60 chunks)
2. Tries first 40 chunks as final fallback

### Step 4: Quality Metrics & Retry Decision

After each attempt, metrics are computed:

| Metric | Threshold for retry |
|---|---|
| `evidence_count` | == 0 → retry |
| `verified_ratio` | < 0.35 → retry |
| `coverage` | < 0.45 → retry |
| `avg_confidence` | < 0.45 → retry |

If **all** thresholds pass, the pipeline stops early. Otherwise it retries with larger Top-K.

The best attempt (by composite score = verified_ratio + coverage + avg_confidence) is kept.

---

## 5. Confidence Scoring Formula

Each evidence row gets a confidence score from `_compute_confidence()`:

```
score = (status_score + quote_score + completeness_score) * 100
```

### Components:

| Component | Condition | Value |
|---|---|---|
| `status_score` | verified | 0.50 |
| `status_score` | not_applicable | 0.35 |
| `status_score` | unverified/unknown | 0.20 |
| `quote_score` | quote exists AND len ≥ 20 | 0.25 |
| `quote_score` | otherwise | 0.05 |
| `completeness_score` | `0.25 × (fields_present / total_fields)` | 0.00–0.25 |

### Result range: 0–100

- **Max possible:** `(0.50 + 0.25 + 0.25) × 100 = 100`
- **Min realistic:** `(0.20 + 0.05 + 0.00) × 100 = 25`

### Fields excluded from completeness:
`source`, `_validation`, `_schema_error`, `phenotypes`

---

## 6. "Not Reported" Detection

The constant `NOT_REPORTED_MARKERS` is used throughout the app:

```python
{"", "n/a", "na", "not reported", "not available", "none", "null"}
```

A value is considered "not reported" if:
- It is `None`
- It is `float('nan')` (pandas NaN)
- It is an empty list
- Its lowered/stripped text matches any marker above

---

## 7. Configuration Parameters

### From `config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | `""` (env) | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `DEFAULT_MODEL` | `anthropic/claude-3.5-sonnet` (env) | Default LLM model |
| `PDF_DIR` | `./pdfs` (env) | Directory containing input PDFs |
| `CHROMA_PERSIST_DIR` | `./chroma_db` (env) | ChromaDB storage path |
| `CHUNK_SIZE` | `2000` | Legacy character chunk size (used in fallback) |
| `CHUNK_OVERLAP` | `500` | Legacy character overlap between chunks |
| `TOP_K_CHUNKS` | `150` | Base number of chunks to retrieve per query |
| `TABLE_EXTRACTION_ENABLED` | `true` (env) | Enable table extraction during ingestion |
| `OCR_ENABLED` | `true` (env) | Enable OCR for low-text pages |
| `OCR_MIN_TEXT_CHARS` | `250` (env) | Minimum text before OCR triggers |
| `OCR_MIN_IMAGES` | `1` (env) | Minimum images on page before OCR triggers |
| `OCR_LANG` | `eng` (env) | Tesseract OCR language |
| `OCR_MIN_IMAGE_COVERAGE` | `0.6` (env) | Image/page ratio to treat as scan |
| `OCR_MAX_DRAWINGS` | `100` (env) | Vector drawings threshold for chart detection |

### Implicit parameters (not in config, hardcoded):

| Parameter | Value | Location | Meaning |
|---|---|---|---|
| `PARENT_CHUNK_TOKENS` | `1000` | `ingest.py` | Target parent chunk size (~4000 chars) |
| `CHILD_CHUNK_TOKENS` | `200` | `ingest.py` | Target child chunk size (~800 chars) |
| `OPENROUTER_VLM_MODEL` | `google/gemini-2.5-flash-image` | `ingest.py` | VLM model for table/image summarization |
| `_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | `retrieval.py` | Sentence-transformer embedding model |
| `COLLECTION_NAME` | `sepsis_papers` | `retrieval.py` | ChromaDB collection name |
| Cosine similarity space | `cosine` | `retrieval.py` | HNSW metric for vector search |
| Batch size for upsert | `100` | `retrieval.py` | Chunks per ChromaDB upsert call |
| Min chunk length | `200` chars | `retrieval.py` | Minimum length for `filter_chunks` |

---

## 8. Reference Chunk Filtering Heuristic

`_is_reference_chunk(text)` in `retrieval.py` returns `True` if any of:

1. Text contains "references" or "bibliography"
2. ≥40% of lines start with a number followed by `.` or `)`
3. ≥6 year patterns (19xx or 20xx) AND ≥2 citation markers (`et al`, `doi:`, `pmid`, `vol.`, `pp.`, `issn`)

---

## 9. Extraction Normalization Rules

### UC1 (`_normalize_uc1`):
- If `predictor` is "Not reported" but `predictor_variable` has a value → copy to `predictor`
- If `predictor_variable` is "Not reported" but `predictor` has a value → copy to `predictor_variable`

### UC2 (`_normalize_uc2`):
- Converts `variables_used` from string to list (splits on `,`, `\n`, `;`)
- Infers `assignment_is_reproducible` from `assignment_feasibility` text
- Defaults `reproducibility_notes` from `assignment_notes`
- Cross-fills `cluster_name` ↔ `cluster_id` and `outcome` ↔ `outcomes` in phenotype entries

### UC3 (`_normalize_uc3`):
- Cross-fills `biomarker_name` ↔ `biomarker_or_score`
- Extracts numeric `auroc` from the string `auc` field when missing
- Defaults `cohort_setting` to "Not reported"

---

## 10. UC2 Salvage Mechanism

When LLM extraction fails for UC2 (no parseable JSON or empty result), the system has a multi-level fallback:

1. **Recovery prompt:** Retry with a stronger instruction to return at least one item
2. **`_salvage_uc2_from_chunks()`:** Regex-based extraction directly from chunk text:
   - Scans for phenotype/cluster keywords
   - Detects clustering method from text patterns
   - Extracts number of clusters via regex
   - Captures first supporting sentence
   - Generates minimal schema-valid rows (max 8)

---

## 11. Keyword Lists

### UC2 Keywords (used for chunk filtering and retrieval fallback):
```
phenotype, phenotypes, cluster, clustering, latent class, subtype, subtypes,
endotype, endotypes, unsupervised, k-means, hierarchical, mixture model
```

### UC3 Keywords (used for retrieval fallback):
```
biomarker, score, auc, auroc, roc, c-index, sensitivity, specificity,
hazard ratio, odds ratio, sofa, saps ii, apache, lactate, procalcitonin, il-6
```

---

## 12. Ingestion Report Fields

The `_new_ingest_report()` dict tracks:

| Field | Meaning |
|---|---|
| `documents_discovered` | Total PDFs found in directory |
| `documents_started` | PDFs that began processing |
| `documents_completed` | PDFs successfully processed |
| `documents_failed` | PDFs that errored during parsing |
| `document_parse_errors` | Count of parse exceptions |
| `pages_extracted` | Total pages across all documents |
| `parents_created` | Section-aware parent chunks created |
| `chunks_created` | Final child chunks created for indexing |
| `tables_processed` | Table elements encountered |
| `images_processed` | Image elements encountered |
| `table_parse_failures` | Tables where no image payload was available |
| `image_parse_failures` | Images where no image payload was available |
| `vlm_successes` | VLM returned useful summary |
| `vlm_failures` | VLM call failed |
| `vlm_empty` | VLM returned empty content |
| `vlm_unavailable` | VLM could not run (no API key) |
| `fallback_used` | Whether legacy chunking was used as fallback |

---

## 13. Schema Required Fields (per Use Case)

These fields trigger schema-aware repair when missing:

### UC1:
`study_name`, `population`, `sample_size`, `setting`, `predictor`, `outcome_definition`, `effect_size`, `source.quote`, `source.page_number`

### UC2:
`study_name`, `clustering_method`, `num_clusters`, `assignment_feasibility`, `source.quote`, `source.page_number`

### UC3:
`study_name`, `biomarker_or_score`, `population`, `outcome`, `effect_size`, `auc`, `source.quote`, `source.page_number`

---

## 14. Effect Direction Inference (Contradiction Graph)

`_infer_effect_direction()` in `analytics.py`:

1. Searches for OR/HR/RR values in effect_size + quote text
2. If ratio > 1 → `increase_risk`
3. If ratio < 1 → `decrease_risk`
4. Falls back to keyword detection: "increased mortality" → `increase_risk`, "protective" → `decrease_risk`
5. UC2 default: `phenotype_descriptor`
6. Other default: `neutral`

---

## 15. Validation Status Values

| Status | Meaning | Visual Badge |
|---|---|---|
| `verified` | Quote was found (substring or ≥65% word overlap) in retrieved chunks | ✅ verified |
| `unverified` | Quote could not be matched in any chunk | ⚠️ unverified |
| `not_applicable` | Quote is "Not reported" — nothing to verify | ℹ️ N/A |
