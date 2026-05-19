# Complete Terminology Glossary — Sepsis Atlas Project

This is a reference for explaining the critical, difficult, and domain-specific terms used throughout the project. Use it when onboarding collaborators or answering technical questions.

---

## PART A: Architecture & Pipeline Terms

### **Modular Pipeline**
**Definition:** The system is deliberately split into 7 independent sequential stages, each in its own module.

**Why it matters:** Each stage can fail independently, be tested independently, and be debugged independently. Failure in one module doesn't cascade to others.

**The 7 stages:**
1. **Ingestion** (`src/ingest.py`) — PDFs → layout-aware chunks with metadata
2. **Indexing** (`src/retrieval.py`) — Chunks → ChromaDB with embeddings
3. **Retrieval** (`src/retrieval.py`) — Query → ranked relevant chunks
4. **Extraction** (`src/extraction.py`) — Chunks + schema → LLM structured evidence
5. **Validation** (`src/validation.py`) — Evidence → independent quote verification + cross-row checks
6. **Confidence Scoring** (`app.py`) — Validation results → numeric confidence 0–100
7. **Analytics** (`src/analytics.py`) — Evidence → knowledge graph + contradiction detection

---

### **Adaptive Retry (Self-Correcting Retrieval)**
**Definition:** The system doesn't retrieve once and hope; it monitors quality metrics and automatically retries with larger Top-K if needed.

**The escalation sequence:**
- **Attempt 1:** Top-K = 150 chunks (tunable, typically starts here)
- **Check quality:** verified_ratio >= 35%? coverage >= 45%? confidence >= 45%?
- **If all pass:** Use results (success on first attempt)
- **If any fail:** Escalate to larger Top-K
- **Final attempt:** Top-K = 450 (absolute max, searches most of corpus)

**Example from demo:**
> "Retrieved 127 passages on first attempt with verified_ratio=100% (49/49), coverage=90%, confidence=97 — no retry needed."

---

### **Schema-Aware Repair**
**Definition:** After extraction, identify which required fields are blank in which rows, then ask the LLM ONLY for those missing fields — doesn't regenerate the whole row.

**Why it matters:**
- **Preserves verified data:** The 90% that's already correct stays unchanged
- **Avoids LLM drift:** If we re-extract everything, values might change randomly (temperature=0 helps but doesn't eliminate all variance)
- **Efficient:** Only fills gaps, doesn't redo work

**Example:**
```
Original row: predictor="SOFA", outcome=BLANK, effect_size="OR 1.4", method=BLANK
Repair prompt: "Given this evidence, what is outcome_definition and statistical_method?"
Result: outcome → "30-day mortality", method → "logistic regression"
Updated row preserves SOFA and OR 1.4, patches only blanks
```

---

### **Cross-Row Consistency (Contextual Disambiguation Checker)**
**Definition:** Post-extraction analysis that flags when the same source quote supports different predictors, or the same effect size from the same study is assigned to different predictors.

**The problem it solves:**
A paper says: "In multivariate analysis, presepsin (OR=1.221), lactate (OR=1.296), and SOFA (OR=1.404) predicted mortality."

Without consistency checking: LLM extracts 3 rows; but did it correctly attribute each OR to its predictor? Or did it accidentally swap them?

**What the checker does:**
- Finds 3 rows sharing the same source quote
- Flags each with orange warning: "Same source quote supports different predictors — verify that each value is correctly attributed"
- Researcher sees the warning and can manually verify

**Example from live demo:**
Wen 2019 single sentence reported 3 effect sizes for 3 predictors. System extracted 3 separate rows AND flagged all 3 with consistency warnings, prompting manual verification.

---

### **ChromaDB Persistence**
**Definition:** Vector embeddings and chunk metadata are stored locally on disk in `chroma_db/` directory, surviving between sessions.

**Why it matters:**
- **No re-indexing needed for changes:** If we modify extraction prompts or validation logic, we don't re-embed (embedding is expensive)
- **Offline capability:** No API needed for retrieval; works without internet after first indexing
- **Speed:** 30 papers indexed once (~5 minutes); then queries return in milliseconds
- **Reproducibility:** Same index, same query, same results every time

**Practical example from demo:**
> "We modified `src/extraction.py` rules on Day 2. The `chroma_db/` persisted from Day 1 — no need to re-ingest all 30 papers."

---

## PART B: Hallucination Prevention & Verification

### **Temperature = 0.0**
**Definition:** LLM parameter controlling randomness. 0 = deterministic (highest-probability token every time), 1.0 = creative randomness.

**Why we use it:**
- Same prompt → same output always (deterministic)
- No creativity/hallucination via random selection
- Reproducible results for evaluation
- Enables reliable regression testing

**Code example:**
```python
response = client.chat.completions.create(
    temperature=0.0,  # ← Deterministic
    messages=[...]
)
```

**Important caveat:** Temperature=0 helps but doesn't eliminate ALL variance. Different model versions, internal tokenizer updates, or minor prompt changes can still cause output differences. That's why we also have schema validation + verification layers.

---

### **Strict System Prompt (12 Rules)**
**Definition:** Explicit rules given to the LLM at the start of every conversation, instructing it how to behave.

**Key rules:**
1. Extract ONLY values explicitly stated
2. "If absent, write 'Not reported'" (never invent)
3. "Do NOT infer, guess, or interpolate"
4. Follow schema strictly
5–8. Per-field formatting rules (effect_size format, population definition, etc.)
9. Create separate rows for multiple cohorts
10. Ensure values match correct predictor + population + model
11. Extract each table row as separate evidence item
12. Separate rows for different timepoints/outcomes

**Why 12 instead of a long rambling prompt:**
- Clear, numbered rules are easier for LLM to follow than paragraph prose
- Each rule is addressable one-at-a-time
- Easier to debug when LLM violates a specific rule ("You violated Rule 9 about separate cohorts")
- Fits in context window efficiently

---

### **"Not Reported" Marker**
**Definition:** Explicit value (not null, not blank) that means "the paper doesn't contain this information."

**Why it matters:**
- **Honest about gaps:** Missing data is not invented data
- **Transparent in analysis:** Downstream users see exactly which fields were in papers vs. which weren't
- **Prevents hallucination:** LLM has an explicit fallback that isn't "make a guess"
- **Schema valid:** Recognized by downstream analysis (not treated as error)

**Recognized values:**
```python
{"", "n/a", "na", "not reported", "not available", "none", "null"}
```

**Example:**
Paper doesn't report sample size → field becomes `"sample_size": "Not reported"` (not `null`, not inferred as "Unknown" or "See Methods")

---

### **Quote Verification (Substring + Word-Overlap)**
**Definition:** Two-level check that a claimed quote actually exists in the source text.

**Level 1: Exact Substring Match**
- Does the quote appear verbatim in any retrieved chunk?
- Normalizes whitespace and case
- If yes → `status="verified"`, move to next

**Level 2: Word-Overlap Fallback (65% threshold)**
- If exact match fails, compare word-by-word
- If >= 65% words overlap → `status="verified"` (handles OCR errors)
- If < 65% → `status="unverified"` with warning

**Why 65%?**
- Below 65%: genuinely different content (LLM likely hallucinated)
- Above 65%: likely OCR/whitespace differences, minor formatting variations, acceptable error margin
- Empirically validated against healthcare literature with OCR noise

**Example:**
- Claimed: "Binary logistic regression showed presepsin level (OR=1.221, 95% CI: 1.024–1.455)"
- Source: "Binary logistic regression showed presepsin level (OR = 1.221, 95% CI: 1.024-1.455)" (different dash, space after =)
- Word overlap: 18/18 words match = 100% → `verified`

**Demo result:** All 49 rows showed `verified` badge — 100% quote match across entire extraction.

---

### **Validation Status**
**Definition:** Three-valued field indicating confidence in a row's traceability.

| Status | Meaning | Confidence Impact |
|---|---|---|
| `verified` | Quote found in source text (exact or ≥65% overlap) | +0.50 to confidence score |
| `unverified` | Quote not found despite checks | +0.20 to confidence score |
| `not_applicable` | Row has no quote (e.g., "Not reported") | +0.35 to confidence score |

**Why three values?**
- Some values inherently can't have quotes (e.g., "Not reported" is meta-data, not an extraction claim)
- Distinguishing cases prevents false negatives
- Allows nuanced scoring: "not_applicable" isn't bad, just different

**Demo example:** All 49 rows showed `verified` status badge — high color (green).

---

## PART C: Technical Concepts

### **Determinism vs. Reproducibility**

**Determinism:**
- Same input → same output, every time
- Achieved via: `temperature=0`, persistent index, no randomness in validation
- Within-system: running query twice on same machine gives identical results
- Example: Run query "Which predictors predict mortality?" twice → identical 49 rows, identical quotes, identical confidence scores

**Reproducibility:**
- Different person, different time, different machine → same results
- Achieved via: Persistent ChromaDB (data stored), versioned models (Claude Sonnet 4.6 pinned), documented prompts, version control
- Cross-system: tomorrow, with the same 30 papers indexed, same query returns identical results
- Example: GitHub clone + same .env + run same query → same 49 rows

**Why both matter:**
- Enables independent verification across machines and teams
- Makes regression testing and debugging predictable

---

### **BGE-Large Embeddings**
**Definition:** A pre-trained sentence transformer model (`BAAI/bge-large-en-v1.5`) that converts text into 1024-dimensional dense vectors for semantic similarity comparison.

**Why BGE-large (not alternatives)?**
- **Better for medical text:** Domain-specific training on scientific papers helps; catches biomedical synonymy better
- **Local execution:** No API needed — fast and private after first download
- **Robust:** Good balance of semantic understanding and computational efficiency
- **Trade-off:** ~1.3GB model size vs. ~80MB for MiniLM, but retrieval quality worth it

**How it works:**
```
Query: "Which lactate levels predict mortality?"
        ↓ BGE-large embedding ↓
Vector: [0.12, -0.45, ..., 0.89]  (1024 float numbers)

Paper chunk: "Lactate > 4 mmol/L strongly predicted 30-day mortality"
        ↓ BGE-large embedding ↓
Vector: [0.14, -0.43, ..., 0.91]  (similar to query vector)

Cosine similarity: 0.89 (on 0–1 scale) ← High match → Retrieved
```

**Default threshold:** Chunks with similarity ≥ 0.65 are retrieved (tunable).

---

### **Word-Overlap Matching**
**Definition:** Comparing two texts by counting matching words as a percentage.

**Formula:**
```
matching_words = set(quote_words) ∩ set(source_words)
overlap = len(matching_words) / len(quote_words)
verified if overlap >= 0.65
```

**Example:**
```
Quote: "lactate level high"
Source: "lactate levels high"
Words: {lactate, level, high} vs. {lactate, levels, high}
Match: {lactate, high} = 2/3 ≈ 67% overlap ✓ verified
```

**Why not just exact match?**
- PDFs suffer OCR errors: hyphens become underscores, whitespace changes, pluralization differs, dashes change type (- vs. —)
- 65% threshold catches real quotes while rejecting fabrications
- Handles minor editing/reformatting while catching genuine hallucinations

---

### **Top-K Retrieval**
**Definition:** Return the K most-similar chunks from entire corpus, ranked by vector similarity.

**Example:**
```
Corpus: 5,000 chunks from 30 papers
Query: "Which predictors associate with mortality?"
Top-K=150: Return 150 most-similar chunks
These 150 become context for LLM extraction
```

**Why adaptive Top-K?**
- **Some queries need more context:** Rare predictors might appear in fewer papers → need higher K
- **Some queries obvious:** SOFA is very common → low K might suffice
- **Adaptive escalation:** Start small (fast), increase only if quality metrics fail
- **Balances speed vs. recall:** Early attempt is quick; if weak, escalate to thorough search

**Thresholds that trigger retry:**
- `verified_ratio < 35%` — fewer than 35% of extracted rows are quote-verified
- `coverage < 45%` — fewer than 45% of schema fields populated (too many "Not reported")
- `avg_confidence < 45%` — rows averaging below 45/100 on confidence

**Escalation path:** 150 → 300 → 450 (configurable in code)

---

## PART D: Domain & Clinical Terms

### **Predictor**
**Definition:** A measurable variable that precedes and is hypothesized to influence an outcome.

**In sepsis context:**
- Examples: SOFA score, lactate level, presepsin, lymphocyte count, procalcitonin, CRP
- UC1 goal: which clinical predictors best predict 28-day mortality?
- Measured at: admission, 24h post-diagnosis, etc.
- Can be continuous (lactate in mmol/L) or categorical (SOFA ≥3 vs. <3)

**Why critical for UC1:**
- Each numerical value must be associated with the correct predictor
- Easy to confuse when a paper tests multiple predictors (e.g., 3 ORs in one sentence)

---

### **Outcome**
**Definition:** The clinical event we're trying to predict (the dependent variable).

**In sepsis context:**
- Examples: 28-day mortality, in-hospital mortality, ICU mortality, 30-day mortality, organ failure development, septic shock development
- **Critical distinction:** Not all "mortality" outcomes are the same
  - 28-day: fixed timeframe, standardized across studies
  - In-hospital: variable length, depends on hospital LOS
  - ICU: narrower population
  - 90-day: longer follow-up, more comprehensive but rarer

**Why this matters for disambiguation:**
Same predictor (e.g., SOFA) might predict 28-day mortality with one OR and 30-day mortality with different OR. Must keep separate.

**Example:**
```
Study reports: SOFA predicts 30-day mortality (OR 1.4)
Same study also: SOFA predicts ICU mortality (OR 1.6)
These are DIFFERENT rows in UC1 output (different outcomes)
```

**UC1 schema field:** `outcome_definition` (required)

---

### **Effect Size**
**Definition:** Quantitative measure of the strength of association between predictor and outcome.

**Common formats in sepsis literature:**

| Metric | Interpretation | Example | Good range |
|---|---|---|---|
| OR (Odds Ratio) | For binary predictors; probability of outcome if present vs. absent | OR 1.3 (95% CI 1.0–1.6) | >1 protective ✗, >1.0 harmful ✓ |
| HR (Hazard Ratio) | For time-to-event; used in survival analysis | HR 1.5 (95% CI 1.2–2.0) | >1 harmful, <1 protective |
| AUC / AUROC | Discrimination ability (0.5=random, 1.0=perfect) | AUC 0.72 (95% CI 0.68–0.76) | 0.5–1.0, >0.7 good |
| β (coefficient) | In linear regression; units depend on model | β 0.15 | interpretation depends on scale |
| RR (Relative Risk) | Similar to OR; used in cohort studies | RR 1.2 | >1 harmful, <1 protective |

**Confidence interval (CI):**
- 95% CI around the effect estimate
- Example: OR 1.3 (95% CI 1.0–1.6) means we're 95% confident true OR is between 1.0 and 1.6
- If CI crosses 1.0 (for OR/HR/RR), the association may not be statistically significant

**In our schema:**
- `effect_size`: stores the reported value as string (e.g., "OR 1.221 (95% CI: 1.024–1.455)")
- `performance_metrics`: captures additional validation stats (sensitivity, specificity, etc.)

**UC3 focus:** Biomarker ranking by AUROC (higher = better predictor)

---

### **AUROC / AUC**
**Definition:** Area Under the Receiver Operating Characteristic curve — measure of how well a predictor discriminates between cases and controls.

**Scale:** 
- 0.5 = random guessing (coin flip)
- 1.0 = perfect discrimination (always correct)
- 0.6–0.7 = fair discrimination
- 0.7–0.8 = good discrimination
- >0.8 = excellent discrimination

**In UC3 context:**
- UC3 is "Biomarker Selection" — we compare biomarkers by their AUROC
- Higher AUROC = better predictor for mortality
- UC3 output: ranking table sorted by AUROC in descending order

**Example:**
```
| Biomarker | AUROC | 95% CI | # Studies |
|-----------|-------|--------|-----------|
| IL-6      | 0.86  | 0.82–0.90 | 3 |
| SOFA      | 0.72  | 0.68–0.76 | 5 |
| Lactate   | 0.68  | 0.64–0.72 | 4 |
```

**Clinical interpretation:** IL-6 is the best discriminator for mortality in this literature.

---

### **Cohort / Population**
**Definition:** The group of patients from which the study sample was drawn.

**UC1 cohort descriptions:**
- "Sepsis-3 criteria ICU patients"
- "Abdominal sepsis after surgery (Japan)"
- "ED patients with suspected infection (USA)"
- "ICU patients, mixed etiologies (Canada)"

**Why this matters:**
- **Generalizability:** Results from abdominal sepsis may not apply to respiratory sepsis
- **Confounding:** Patient demographics matter (age, comorbidities, treatment access, healthcare system)
- **Clinical context:** A predictor strong in ICU patients might be weak in ED patients
- **Selection bias:** How were patients recruited/selected?

**Examples of cohort details affecting results:**
- Median age 65 vs. 45: mortality rates differ, predictor performance may differ
- Urban teaching hospital vs. rural clinic: access to treatments differs
- Post-surgical vs. community-acquired: etiology and physiology differ

**UC1 schema field:** `population` (required)

---

### **Timepoint / Timing**
**Definition:** When relative to disease onset the predictor was measured.

**Examples in sepsis:**
- "At admission" (baseline, earliest measurement)
- "First 24 hours post-diagnosis"
- "Day 1 of ICU stay"
- "Within 6 hours of ED presentation"
- "24–48 hours post-diagnosis"
- "Not reported" (paper doesn't say when)

**Why it matters:**
- **Early lactate (0–6h):** High discrimination power (very elevated when sepsis identified)
- **Late lactate (24–48h):** Different power (may have been treated/improved)
- **SOFA at admission ≠ SOFA on day 3** — organ dysfunction can improve or worsen
- **Missing timing:** Major source of apples-to-oranges comparisons

**Example of timing impact:**
```
Study 1: Presepsin > 500 at admission predicts mortality (OR 2.1)
Study 2: Presepsin > 500 at 48h predicts mortality (OR 1.3)
Same threshold, different timepoints, different ORs
→ Must keep separate rows with clear timing distinction
```

**UC1 schema field:** `timing` (required field, "Not reported" if absent)

---

### **Semantic Misalignment**
**Definition:** Extracting a number but attributing it to the wrong context (wrong predictor, wrong cohort, wrong outcome, wrong timepoint).

**Example of semantic misalignment:**

Paper Table 1:
```
Predictor        | Cohort A (OR) | Cohort B (OR)
SOFA             | 1.3           | 1.5
Lactate          | 1.2           | 1.4
```

LLM hallucination (semantic misalignment):
```
Row 1: predictor=SOFA, OR=1.4 ← WRONG! That's lactate in cohort B
Row 2: predictor=Lactate, OR=1.3 ← WRONG! That's SOFA in cohort A
(LLM copied numbers from wrong cells)
```

**How we prevent it:**
1. **System prompt Rule 10:** "Ensure each OR is associated with correct predictor + population + model"
2. **Per-row extraction:** One row per predictor/cohort/model combo, not one row per paper
3. **Cross-row consistency check:** Flags if same study reports OR=1.3 twice for different predictors (impossible in single study, indicates copy-paste error)
4. **Researcher verification:** Can expand any row and verify the original paper matches

**Example from live demo:**
Wen 2019 single sentence reported 3 effect sizes for 3 predictors. System extracted 3 separate rows AND flagged all 3 with consistency warnings, prompting manual verification. All 3 verified correctly after review.

Preventing semantic misalignment between extracted data and source meaning is critical.

---

### **Reproducibility (of phenotype assignment)**
**Definition:** Whether other researchers, given a paper's clustering method and variable list, can re-assign new patients to the identified clusters.

**UC2-specific:**
- **Reproducible:** "Authors provide cluster centers and assignment algorithm" → Yes, new patients can be classified using algorithm
- **Not reproducible:** "Authors describe clusters descriptively but don't give assignment rules" → No way to assign new patients; only descriptive

**Example of reproducible assignment:**
```
"Clusters assigned using k-means on 5 biomarkers:
Features: IL-6, SOFA, lactate, platelet count, CRP
Centers:
  Cluster 1: IL-6=150, SOFA=8, lactate=2.0, platelets=180, CRP=250
  Cluster 2: IL-6=50, SOFA=4, lactate=1.0, platelets=300, CRP=100
New patient: Measure 5 biomarkers, find nearest center → assign to cluster"
```

**Example of non-reproducible description:**
```
"We identified 3 phenotypes characterized by:
  Phenotype A: High inflammation, organ dysfunction
  Phenotype B: Moderate inflammation
  Phenotype C: Mild inflammation
(No algorithm provided; next researcher cannot classify new patients)"
```

**UC2 schema field:** `assignment_is_reproducible` (boolean)

---

## PART E: Use-Case Specific Terms

### **UC1 — Predictor-Outcome Association**
**Definition:** The relationship between a single clinical predictor and a single mortality outcome, quantified by effect size (OR/HR/AUC).

**Structure (one row = one association):**
```json
{
  "study_name": "Wen 2019",
  "population": "Sepsis-3 criteria ICU patients",
  "predictor": "Presepsin level",
  "outcome_definition": "In-hospital mortality",
  "effect_size": "OR 1.221 (95% CI: 1.024–1.455)",
  "timing": "At admission",
  "method": "Binary logistic regression",
  "source": {
    "quote": "Binary logistic regression showed presepsin level (OR=1.221, 95% CI: 1.024–1.455) predicted in-hospital mortality",
    "page_number": 2
  }
}
```

**13-field UC1 schema:**
1. study_name
2. population
3. sample_size
4. predictor
5. outcome_definition
6. effect_size
7. performance_metrics (sensitivity, specificity, NLR, etc.)
8. timing
9. method (statistical approach)
10. sepsis_definition (Sepsis-3, consensus, etc.)
11. adjustment (univariate vs. adjusted)
12. adverse_events_reported
13. source (anchor + quote)

**Clinical use:** Estimate expected mortality for new sepsis patient using literature-derived associations (since the registry lacks controls).

**Demo output:** 49 evidence rows from single UC1 query, 100% quote-verified, 97/100 average confidence, 90% field coverage.

---

### **UC2 — Phenotype (Sepsis Subtype)**
**Definition:** A subgroup of sepsis patients identified via unsupervised clustering, defined by shared biomarker signatures and similar outcomes.

**Example phenotypes (Donzelli 2019, 4 clusters):**
| Cluster | Key Features | Clinical Description | Outcome |
|---------|--------------|----------------------|---------|
| A | Low SOFA, low lactate, high platelets | Mild inflammation | ~12% ICU mortality |
| B | Mixed markers | Moderate inflammation | ~25% mortality |
| C | High lactate, high procalcitonin | Hyper-inflammation | ~45% mortality |
| D | High SOFA, multi-organ dysfunction | Severe organ failure | ~55% mortality |

**UC2 schema structure:**

**Study-level fields (one per paper):**
- clustering_method: k-means, hierarchical, latent class, etc.
- number_of_clusters: how many phenotypes identified
- variables_used: which biomarkers/clinical variables
- assignment_is_reproducible: boolean (can new patients be classified?)

**Cluster-level fields (one per cluster in that study):**
- cluster_id
- cluster_size (N patients)
- key_features: means/medians of biomarkers
- clinical_description: narrative summary
- outcomes_per_cluster: mortality rates, other outcomes

**Clinical use:** Stratify patients into risk groups at admission; tailor treatment intensity/monitoring based on phenotype.

---

### **UC3 — Biomarker Ranking (Evidence-Based Stratification)**
**Definition:** Comparative analysis of multiple biomarkers/scores, ranked by their predictive ability (AUROC), to guide selection of best predictor for risk stratification.

**Output: Ranking Table (sorted by AUROC descending)**
```
| Biomarker | Type | Best AUROC | # Studies | Avg Confidence |
|-----------|------|-----------|-----------|-----------------|
| IL-6      | Cytokine | 0.86 | 3 | 96.5 |
| SOFA      | Clinical Score | 0.72 | 5 | 93.2 |
| Lactate   | Metabolite | 0.68 | 4 | 91.8 |
```

**Output: Detailed Comparison Table**
```
| Study | Biomarker | AUROC | 95% CI | Adjustment | Setting |
|-------|-----------|-------|--------|------------|---------|
| Donatello 2025 | IL-6 | 0.86 | 0.82–0.90 | Age, sex, Charlson | ICU sepsis |
| Raphael 2024 | SOFA | 0.72 | 0.70–0.74 | Univariate | ICU mixed |
```

**UC3 schema:**
- 18 fields per row (biomarker, AUROC, adjustment level, validation method, etc.)

**Clinical use:** Select single best biomarker for your specific patient population (e.g., "For ICU sepsis, IL-6 has best AUROC, preferred for admission risk scoring").

---

### **Biomarker Type**
**Definition:** Category of predictor variable.

| Type | Examples | Clinical Notes |
|------|----------|---|
| Clinical Score | SOFA, APACHE, SAPS II | Composite scores of vital signs + organ function; practical at bedside |
| Inflammatory Marker | IL-6, procalcitonin, CRP | Cytokine/protein levels; indicate immune activation |
| Metabolite | Lactate, glucose | Small-molecule markers; reflect cellular metabolism status |
| Cell Count | Lymphocyte count, platelet count | Blood cell population metrics; indicate immune suppression/activation |
| Coagulation Marker | D-dimer, PT/INR | Blood clotting parameters; indicate coagulopathy |
| Organ Dysfunction Marker | Creatinine, bilirubin, INR | Individual organ-specific markers |

**UC3 output includes:** biomarker_type field for classification.

---

### **Adjustment (Statistical)**
**Definition:** Whether an effect size (OR/HR) was adjusted for potential confounding variables.

**Unadjusted analysis:**
- Raw association without controlling for confounders
- Example: "Presepsin > 500 pg/mL predicted mortality (OR 1.8)"
- Problem: Older patients have higher presepsin AND higher mortality — age confounds the association

**Adjusted analysis:**
- Statistically removed confounding effects
- Example: "Presepsin > 500 pg/mL predicted mortality after adjusting for age, sex, comorbidities (adjusted OR 1.5)"
- Interpretation: Age effect removed; presepsin's independent contribution shown (smaller than unadjusted)

**Common confounders in sepsis:**
- Age, sex, comorbidities (Charlson score), ICU admission status, prior treatments, hospital type

**UC1 schema field:** `adjustment` (text, e.g., "Univariate" vs. "Adjusted for age, sex, Charlson")

**UC3 schema field:** `adjustment` (same structure)

---

## PART F: UI & Output Terms

### **Confidence Score (0–100)**
**Definition:** Composite quality metric combining three components.

**Formula:**
```
raw_score = (status_score × 0.50) + (quote_score × 0.25) + (completeness_score × 0.25)
confidence = raw_score × 100
```

**Components:**

| Component | Weight | If Verified | If Unverified | If Not Applicable |
|-----------|--------|-----------|---------------|-------------------|
| Validation status | 0.50 | 0.50 | 0.20 | 0.35 |
| Quote quality | 0.25 | 0.25 (if ≥20 chars) | 0.05 (if <20 chars) | 0.00 (no quote) |
| Field completeness | 0.25 | 0.25 × (fields_filled / total_fields) | Same | Same |

**Example calculation:**
```
Row has:
  - Validation status: verified (0.50)
  - Quote: "...full detailed quote..." (20+ chars, 0.25)
  - Fields filled: 11/13 fields (0.25 × 11/13 = 0.21)
  
Score = (0.50 + 0.25 + 0.21) × 100 = 96/100 ✓
```

**Visual badges in UI:**
- 🟢 Green (≥70): High confidence — trust this row
- 🟡 Yellow (45–69): Medium confidence — worth reviewing
- 🔴 Red (<45): Low confidence — verify manually before use

**Demo result:** Average confidence 97/100 across all 49 rows.

---

### **Schema Coverage**
**Definition:** Percentage of required fields that contain real values (not "Not reported").

**Example:**
- UC1 has 13 core fields
- 11 fields filled with real values
- 2 fields "Not reported"
- Coverage = 11/13 = **85%**

**Display:** In Trust Dashboard as "Schema Coverage: 85%"

**Why it matters:**
- High coverage (>80%): Rich, usable data
- Medium coverage (45–79%): Usable but some gaps
- Low coverage (<45%): Sparse data, risky for analysis

**Demo result:** 90% schema coverage across all 49 rows.

---

### **Consistency Warning**
**Definition:** Orange warning box appearing in source evidence when cross-row checks flag ambiguity.

**Example warning texts:**
- "Same source quote supports different predictors — verify that each value is correctly attributed."
- "Same effect size (OR 1.3) from same study (Wen 2019) used for multiple different predictors — flag for manual review."

**Action:** Researcher clicks, reads original quote, manually verifies the attribution is correct.

**Color choice:** Orange (not red) because it's a flag for review, not an error. Cross-row consistency checker doesn't auto-correct; it alerts.

**Demo result:** 22 consistency warnings flagged across 49 rows (same quote supporting multiple predictors from Wen 2019).

---

### **Evidence Origin**
**Definition:** Metadata indicating whether extracted value came from raw PDF text or from VLM summary of tables/images.

| Origin | Meaning | Confidence Implications |
|--------|---------|---|
| `raw_text` | Extracted directly from PDF text | Higher confidence (direct source) |
| `vlm_summary` | Extracted from VLM-summarized table/image | Slight risk (VLM may misunderstand) |
| `mixed` | Some fields from raw text, some from VLM | Medium confidence |

**Field location:** `source.evidence_origin`

**Example:**
```json
{
  "effect_size": "AUC 0.72",
  "source": {
    "quote": "Table 1: AUC 0.72 for lactate",
    "evidence_origin": "vlm_summary",
    "page_number": 3
  }
}
```

**Why this matters:**
- VLM (Gemini 2.5 Flash) is very good but not perfect for complex tables with multiple columns
- Researchers can deprioritize VLM-sourced values if they need maximum confidence
- Transparency about source type

---

### **Section Name**
**Definition:** Which part of the paper the quote came from.

| Section | Content | Reliability |
|---------|---------|-------------|
| Abstract | Summary of key findings | High (mirrors Results) |
| Methods | Study design, patient selection, statistical methods | Medium (not direct findings) |
| Results | Findings, tables, effect sizes | Highest (primary results) |
| Discussion | Interpretation, limitations, future work | Lower (secondary, interpretation) |
| Supplementary | Appendix data, additional analyses | Medium (may be exploratory) |
| Not reported | Not extractable from document | N/A |

---

## PART G: Process & Implementation Terms

### **Normalization (Per-Use-Case)**
**Definition:** Converting variant field names and formats into standardized form.

**UC1 normalization examples:**
- Input variants: "predictor_variable", "predictor_name", "variable" → Standardized: `predictor`
- Input: "Effect size = OR 1.3", "odds_ratio: 1.3" → Standardized: `effect_size: "OR 1.3"`
- Input: outcome might be "mortality_definition" or "outcome_definition" → Standardized: `outcome_definition`

**Why normalize?**
- **Different papers use different terminology:** No standard format in literature
- **LLM might use variant names:** Depends on prompt, model, context
- **Downstream analysis needs consistency:** SQL queries, aggregate functions, comparisons all assume consistent field names

**Code location:** `src/extraction.py` `_normalize_uc1()`, `_normalize_uc2()`, `_normalize_uc3()`

**Example normalization code:**
```python
def _normalize_uc1(raw_dict):
    normalized = raw_dict.copy()
    # Handle effect_size variants
    if "odds_ratio" in raw_dict and "effect_size" not in raw_dict:
        normalized["effect_size"] = raw_dict["odds_ratio"]
    # Handle outcome_definition variants
    if "outcome" in raw_dict and "outcome_definition" not in raw_dict:
        normalized["outcome_definition"] = raw_dict["outcome"]
    return normalized
```

---

### **Salvage Fallback (UC2-Specific)**
**Definition:** If LLM extraction returns zero phenotypes (complete failure), try regex-based salvage extraction from raw text.

**Why UC2 needs fallback:**
- Some corpora are not phenotype-focused, so signal can be sparse
- Sometimes papers don't have clear structured phenotype tables
- Salvage mode looks for keywords: "cluster", "phenotype", "subtype", "k-means", numerical patterns

**Example salvage keywords:**
```python
PHENOTYPE_KEYWORDS = [
    "phenotype", "phenotypes", "cluster", "clustering", "latent class", 
    "subtype", "subtypes", "endotype", "unsupervised", "k-means", 
    "hierarchical", "mixture model", "patient group", "patient subgroup"
]
```

**Code location:** `src/extraction.py` `_salvage_uc2_from_chunks()`

**Process:**
1. Primary LLM extraction fails (returns empty or null phenotypes)
2. Trigger salvage mode
3. Search chunks for phenotype keywords
4. Extract matching chunks
5. Use fallback schema (simplified phenotype structure)

---

### **Keyword Expansion (Query Augmentation)**
**Definition:** If initial semantic retrieval returns no/weak results, augment query with domain-specific keyword terms for keyword-based fallback retrieval.

**UC2 Keywords (phenotype-specific):**
```python
["phenotype", "phenotypes", "cluster", "clustering", "latent class", 
 "subtype", "subtypes", "endotype", "unsupervised", "k-means", 
 "hierarchical", "mixture model"]
```

**UC3 Keywords (biomarker-specific):**
```python
["biomarker", "score", "auc", "auroc", "roc", "c-index", 
 "sensitivity", "specificity", "hazard ratio", "odds ratio", 
 "sofa", "apache", "lactate", "procalcitonin", "il-6"]
```

**Process:**
1. **Semantic retrieval:** Query vector matches chunks (typically retrieves well)
2. **If results weak:** Append UC-specific keywords to query text
3. **Keyword fallback:** Text-based matching using BM25 or simple substring search in chunk content
4. **Merge results:** Combine semantic + keyword results, deduplicate

**Example query expansion for UC3:**
```
Original: "Which biomarkers predict ICU mortality?"
Expanded: "Which biomarkers predict ICU mortality? biomarker score auc auroc roc 
           c-index sensitivity specificity hazard ratio odds ratio sofa apache 
           lactate procalcitonin il-6"
```

---

### **Per-Row Extraction**
**Definition:** For each predictor/biomarker/cluster in a paper, create one separate evidence row (not one row per paper).

**Example (Wen 2019 reports 3 ORs):**
```
Paper: "Presepsin (OR=1.221), lactate (OR=1.296), SOFA (OR=1.404) predicted mortality"

Our output (CORRECT):
Row 1: study="Wen 2019", predictor="Presepsin", effect_size="OR 1.221"
Row 2: study="Wen 2019", predictor="Lactate", effect_size="OR 1.296"
Row 3: study="Wen 2019", predictor="SOFA", effect_size="OR 1.404"
(all 3 flagged with consistency warning due to shared quote)

Alternative (WRONG):
Row 1: study="Wen 2019", predictor="Multiple", effect_size="Various ORs" ← Bad
```

**Why per-row is critical:**
- Each predictor is comparable on its own row
- Downstream analysis can filter by predictor
- Rankings and aggregations work correctly

**System Prompt Rule 11:** "If a table presents multiple rows of results, extract each row as a separate evidence item with its own predictor, effect size, and confidence interval."

---

### **Session State**
**Definition:** Streamlit's in-memory storage that persists within a user session but clears when the app restarts or browser session ends.

**We use it for:**
```python
st.session_state[f"results_uc{uc}"]        # Cached extraction results
st.session_state[f"query_uc{uc}"]          # Current query text
st.session_state[f"query_history_uc{uc}"]  # Past queries for quick re-run
st.session_state["indexing_logs"]          # PDF indexing progress
```

**Benefit:** User switches between UC1/UC2/UC3 tabs — results stay cached until explicitly cleared.

**Lifetime:** Session persists until:
- Browser tab closes
- User clears browser cookies
- App restarts (developer edits code)
- "Clear Results" button clicked by user

---

### **Retry Report**
**Definition:** Metadata returned from adaptive retrieval showing retry history.

**Example retry report:**
```json
{
  "attempt_count": 1,
  "retrieval_attempts": [
    {
      "top_k": 150,
      "evidence_count": 49,
      "verified_ratio": 1.0,
      "coverage": 0.90,
      "avg_confidence": 97,
      "passed_thresholds": true,
      "retry_reason": null
    }
  ]
}
```

**Displayed to user in UI:**
> "Retrieved 127 passages on attempt 1 with 49 evidence, 100% verified, 90% coverage — quality thresholds passed."

**Multi-attempt example:**
```
Attempt 1: Top-K=150, verified_ratio=25% ← Below 35% threshold, retry
Attempt 2: Top-K=300, verified_ratio=45% ← Above 35%, passed
Result: Used Attempt 2 results
```

---

## PART H: Clinical/Research Context Terms

### **Sepsis-3 Criteria**
**Definition:** Standardized clinical definition of sepsis adopted in 2016 by a consensus conference.

**Key components:**
- Suspected infection (likely or documented)
- SOFA score increase by ≥ 2 points (indicates new organ dysfunction)

**Why papers mention it:**
- Standardized definition helps compare cohorts across studies
- Replaces older definitions (Sepsis-1, Sepsis-2)
- Clinical reproducibility: clinicians worldwide use this definition

**In schema:** `sepsis_definition` field captures which definition (Sepsis-3, consensus, older, not specified)

---

### **Septic Shock**
**Definition:** Subset of sepsis with severe vasodilation requiring vasopressor support to maintain blood pressure, plus hyperlactatemia.

**Clinical significance:**
- More severe than simple sepsis
- Higher mortality
- Different pathophysiology (profound vasodilation vs. moderate inflammation)

**In UC1 context:** Some papers study "sepsis" broadly, others focus on "septic shock" specifically.

---

### **Hemoadsorption**
**Definition:** Extracorporeal blood purification technique that removes inflammatory mediators (cytokines) from blood via charged plastic beads.

**Registry context:** When a registry includes hemoadsorption-treated sepsis patients, literature evidence can estimate counterfactual outcomes without treatment.

**UC1 relevance:** Effect sizes from literature (predictors of natural sepsis mortality) must be separated from hemoadsorption-treated patients.

---

### **In-Hospital Mortality vs. 28-Day Mortality**
**Definition:** Two different outcome measures.

| Measure | Definition | Pros | Cons |
|---------|-----------|------|------|
| In-hospital mortality | Death during hospital stay | Practical, easy to measure | Variable length by hospital/patient LOS |
| 28-day mortality | Death within 28 days of diagnosis | Standardized, fixed timeframe | May miss later deaths, requires follow-up |

**Why both exist:**
- 28-day is standardized (comparable across studies)
- In-hospital is practical (no follow-up needed)
- Registry might have one but not the other

**In UC1:** Must distinguish which outcome each predictor predicts (they're different rows if same predictor tested for both).

---

## Summary Quick-Reference Table

| Term | Category | Definition |
|------|----------|---|
| Modular pipeline | Architecture | 7 independent stages for separation of concerns |
| Adaptive retry | Architecture | Auto-escalates Top-K if quality weak |
| Schema-aware repair | Architecture | Fix only missing fields, preserve correct values |
| Cross-row consistency | Architecture | Flags same quote supporting different predictors |
| Temperature = 0 | Hallucination prevention | Deterministic LLM (no randomness) |
| Strict system prompt | Hallucination prevention | 12 explicit rules for LLM behavior |
| Quote verification | Hallucination prevention | Exact match or 65% word-overlap threshold |
| BGE-large | Technical | Local embedding model for medical text |
| Top-K retrieval | Technical | Return K most-similar chunks from corpus |
| Predictor | Domain | Variable hypothesized to influence outcome |
| Effect size | Domain | OR/HR/AUC quantifying predictor-outcome association |
| AUROC | Domain | Discrimination ability (0.5=random, 1.0=perfect) |
| Cohort | Domain | Group of patients studied |
| Timepoint | Domain | When predictor measured relative to disease onset |
| Semantic misalignment | Domain | Extracting number but attributing to wrong context |
| Determinism | Quality | Same input → same output every time |
| Reproducibility | Quality | Different person/time/machine → same results |
| UC1/UC2/UC3 | Use cases | Mortality / Phenotypes / Biomarker ranking |
| Confidence score | Output | Composite metric (0–100) combining verification + completeness |
| Schema coverage | Output | % of fields with real values |
| Evidence origin | Output | Whether from raw text or VLM summary |
| Session state | Implementation | Streamlit memory persisting within user session |
| Normalization | Implementation | Converting variant field names to standard form |
| Per-row extraction | Implementation | One row per predictor/biomarker, not one row per paper |

---

## Using This Glossary

**Strategy:**
1. When a reviewer asks a technical question, reference the relevant section
2. Give definition first, then clinical/architectural context
3. Use examples from your demo (Wen 2019 data if possible)
4. Explain why the term matters for the task
5. Be precise — conflating related terms reduces trust

**Example question:**
> "What do you mean by 'verified' status? How is that different from 'unverified'?"

**Sample answer:**
> "Verified means we found the exact quote in the source text, or achieved ≥65% word-overlap accounting for OCR errors. Unverified means we tried both checks and neither succeeded — the quote wasn't found, suggesting possible LLM hallucination. Not applicable is for values that don't have quotes, like 'Not reported' which is metadata, not an extraction claim. The three-way distinction prevents false negatives on the 'not_applicable' category."

---

This glossary covers difficult and critical terms. Use it to answer technical questions with precision.
