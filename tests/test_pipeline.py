"""
Unit tests for the Sepsis Atlas pipeline.

These tests cover pure Python logic and do NOT require:
- A running LLM / OpenRouter API key
- PDF files on disk
- A ChromaDB instance

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# src.ingest tests
# ---------------------------------------------------------------------------

class TestChunkPages:
    """Tests for the chunk_pages helper in src.ingest."""

    def _make_pages(self, texts: list[str], source: str = "test") -> list[dict]:
        return [
            {"text": t, "page_number": i + 1, "source": source}
            for i, t in enumerate(texts)
        ]

    def test_basic_chunking(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["A" * 3000])
        chunks = chunk_pages(pages, chunk_size=1000, overlap=200)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk["text"]) <= 1000

    def test_short_page_produces_single_chunk(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["Short text."])
        chunks = chunk_pages(pages, chunk_size=1000, overlap=200)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_empty_page_skipped(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["   \n\t  "])
        chunks = chunk_pages(pages, chunk_size=1000, overlap=200)

        assert len(chunks) == 0

    def test_chunk_metadata(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["Hello world"], source="my_paper")
        chunks = chunk_pages(pages, chunk_size=1000, overlap=200)

        assert chunks[0]["source"] == "my_paper"
        assert chunks[0]["page_number"] == 1
        assert "chunk_id" in chunks[0]
        assert chunks[0]["chunk_id"].startswith("my_paper_p1_")

    def test_overlap_creates_shared_content(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["X" * 2000])
        chunks = chunk_pages(pages, chunk_size=1200, overlap=400)

        assert len(chunks) >= 2
        # Each chunk must be non-empty
        assert all(c["text"] for c in chunks)

    def test_multiple_pages(self):
        from src.ingest import chunk_pages

        pages = self._make_pages(["Page one content.", "Page two content."])
        chunks = chunk_pages(pages, chunk_size=1000, overlap=100)

        page_nums = {c["page_number"] for c in chunks}
        assert 1 in page_nums
        assert 2 in page_nums

    def test_ingest_pdfs_returns_empty_for_missing_dir(self, tmp_path):
        from src.ingest import ingest_pdfs

        result = ingest_pdfs(str(tmp_path))
        assert result == []

    def test_ingest_pdfs_returns_empty_when_no_pdfs(self, tmp_path):
        from src.ingest import ingest_pdfs

        (tmp_path / "notes.txt").write_text("not a pdf")
        result = ingest_pdfs(str(tmp_path))
        assert result == []

    def test_ingest_pdfs_return_report_for_empty_dir(self, tmp_path):
        from src.ingest import ingest_pdfs

        chunks, report = ingest_pdfs(str(tmp_path), return_report=True)
        assert chunks == []
        assert report["documents_discovered"] == 0
        assert report["documents_completed"] == 0
        assert report["tables_processed"] == 0

    def test_record_visual_stats_counts_failures_and_successes(self):
        from src.ingest import _new_ingest_report, _record_visual_stats

        report = _new_ingest_report()
        _record_visual_stats(
            report,
            visual_type="Table",
            has_payload=False,
            summary="[VLM SUMMARY FAILED]\nCould not summarize table: boom",
        )
        _record_visual_stats(
            report,
            visual_type="Image",
            has_payload=True,
            summary="## Figure summary\n- Finding",
        )

        assert report["tables_processed"] == 1
        assert report["images_processed"] == 1
        assert report["table_parse_failures"] == 1
        assert report["vlm_failures"] == 1
        assert report["vlm_successes"] == 1


# ---------------------------------------------------------------------------
# src.schemas tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Tests for Pydantic schema validation."""

    def _base_source(self):
        return {"page_number": 3, "quote": "Lactate was elevated in 80% of patients."}

    def test_uc1_valid(self):
        from src.schemas import UseCase1Evidence

        data = {
            "study_name": "Smith 2023",
            "population": "ICU sepsis patients",
            "sample_size": "N=200",
            "setting": "ICU",
            "predictor": "Lactate",
            "outcome_definition": "28-day mortality",
            "timing": "Admission",
            "statistical_method": "Logistic regression",
            "effect_size": "OR 2.1",
            "performance_metrics": "AUC 0.78 (95% CI 0.72–0.84)",
            "notes": "Not reported",
            "source": self._base_source(),
        }
        e = UseCase1Evidence(**data)
        assert e.study_name == "Smith 2023"
        assert e.effect_size == "OR 2.1"

    def test_uc1_notes_defaults_to_not_reported(self):
        from src.schemas import UseCase1Evidence

        data = {
            "study_name": "Jones 2022",
            "population": "Septic shock",
            "sample_size": "N=100",
            "setting": "ICU",
            "predictor": "SOFA score",
            "outcome_definition": "In-hospital mortality",
            "timing": "Not reported",
            "statistical_method": "ROC analysis",
            "effect_size": "AUC 0.81",
            "performance_metrics": "Not reported",
            "source": self._base_source(),
        }
        e = UseCase1Evidence(**data)
        assert e.notes == "Not reported"

    def test_uc2_valid_with_phenotypes(self):
        from src.schemas import PhenotypeCluster, UseCase2Evidence

        data = {
            "study_name": "Donzelli 2019",
            "country": "Norway",
            "setting": "ICU",
            "sample_size": "N=1476",
            "sepsis_definition": "Sepsis-3",
            "clustering_method": "k-means clustering",
            "num_clusters": "4",
            "variables_used": "18 clinical and laboratory variables",
            "assignment_feasibility": "Insufficient detail",
            "assignment_notes": "Not reported",
            "phenotypes": [
                {
                    "cluster_id": "A",
                    "cluster_size": "Not reported",
                    "key_features": "Low SOFA, low lactate",
                    "clinical_description": "Low severity",
                    "outcome": "ICU mortality ~12%",
                    "notes": "Not reported",
                }
            ],
            "source": self._base_source(),
        }
        e = UseCase2Evidence(**data)
        assert len(e.phenotypes) == 1
        assert e.phenotypes[0].cluster_id == "A"

    def test_uc3_valid(self):
        from src.schemas import UseCase3Evidence

        data = {
            "study_name": "Lee 2021",
            "biomarker_or_score": "Lactate",
            "biomarker_type": "Biomarker",
            "population": "Septic shock patients",
            "sample_size": "N=350",
            "outcome": "28-day mortality",
            "effect_size": "OR 1.8",
            "auc": "0.79",
            "confidence_interval": "95% CI 0.73–0.85",
            "adjustment": "Not reported",
            "statistical_method": "Logistic regression",
            "validation_method": "Internal cross-validation",
            "relevance_to_target_population": "Not reported",
            "cohort_characteristics": "Mean age 65, 55% male",
            "notes": "Not reported",
            "source": self._base_source(),
        }
        e = UseCase3Evidence(**data)
        assert e.auc == "0.79"

    def test_source_anchor_valid(self):
        from src.schemas import SourceAnchor

        s = SourceAnchor(page_number=5, quote="Patients had elevated lactate.")
        assert s.page_number == 5

    def test_uc1_model_dump_is_dict(self):
        from src.schemas import UseCase1Evidence

        data = {
            "study_name": "X 2020",
            "population": "P",
            "sample_size": "N=50",
            "setting": "ED",
            "predictor": "IL-6",
            "outcome_definition": "30-day mortality",
            "timing": "24h",
            "statistical_method": "ROC",
            "effect_size": "AUC 0.7",
            "performance_metrics": "Sens 0.8",
            "source": {"page_number": 1, "quote": "IL-6 was measured."},
        }
        d = UseCase1Evidence(**data).model_dump()
        assert isinstance(d, dict)
        assert d["study_name"] == "X 2020"


# ---------------------------------------------------------------------------
# src.validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for quote verification and batch validation."""

    def _make_chunks(self, texts: list[str]) -> list[dict]:
        return [
            {"text": t, "source": "paper", "page_number": i + 1}
            for i, t in enumerate(texts)
        ]

    def test_exact_match_verified(self):
        from src.validation import verify_quote

        chunks = self._make_chunks(["Lactate above 4 mmol/L was a strong predictor."])
        assert verify_quote("Lactate above 4 mmol/L was a strong predictor.", chunks)

    def test_not_reported_always_verified(self):
        from src.validation import verify_quote

        assert verify_quote("Not reported", [])
        assert verify_quote("not reported", [])
        assert verify_quote("", [])

    def test_missing_quote_fails(self):
        from src.validation import verify_quote

        chunks = self._make_chunks(["Completely different text about something else."])
        assert not verify_quote("Lactate above 4 mmol/L was a strong predictor.", chunks)

    def test_high_overlap_verified(self):
        from src.validation import verify_quote

        quote = "Elevated lactate was associated with increased mortality in all patients."
        chunk_text = "Elevated lactate was associated with increased mortality in all patients admitted to the ICU."
        chunks = self._make_chunks([chunk_text])
        assert verify_quote(quote, chunks)

    def test_validate_extraction_adds_validation_key(self):
        from src.validation import validate_extraction

        evidence = {
            "study_name": "Test 2023",
            "source": {"page_number": 1, "quote": "Not reported"},
        }
        result = validate_extraction(evidence, [])
        assert "_validation" in result
        assert result["_validation"]["status"] == "not_applicable"

    def test_validate_extraction_unverified(self):
        from src.validation import validate_extraction

        evidence = {
            "source": {
                "page_number": 1,
                "quote": "This sentence does not appear in any chunk.",
            }
        }
        chunks = [{"text": "Something completely different.", "source": "p", "page_number": 1}]
        result = validate_extraction(evidence, chunks)
        assert result["_validation"]["status"] == "unverified"
        assert result["_validation"]["warning"] is not None

    def test_validate_all_returns_same_length(self):
        from src.validation import validate_all

        evidence_list = [
            {"source": {"page_number": 1, "quote": "Not reported"}},
            {"source": {"page_number": 2, "quote": "Another not reported"}},
        ]
        result = validate_all(evidence_list, [])
        assert len(result) == len(evidence_list)
        assert all("_validation" in e for e in result)


# ---------------------------------------------------------------------------
# src.extraction helpers (no LLM call)
# ---------------------------------------------------------------------------

class TestExtractionHelpers:
    """Tests for pure helper functions in src.extraction."""

    def test_format_context(self):
        from src.extraction import _format_context

        chunks = [
            {"text": "Text A", "source": "paper1", "page_number": 2},
            {"text": "Text B", "source": "paper2", "page_number": 5},
        ]
        ctx = _format_context(chunks)
        assert "Excerpt 1" in ctx
        assert "paper1" in ctx
        assert "Page: 2" in ctx
        assert "Text A" in ctx
        assert "Excerpt 2" in ctx

    def test_parse_json_response_plain_array(self):
        from src.extraction import _parse_json_response

        raw = '[{"key": "value"}]'
        result = _parse_json_response(raw)
        assert isinstance(result, list)
        assert result[0]["key"] == "value"

    def test_parse_json_response_with_fences(self):
        from src.extraction import _parse_json_response

        raw = '```json\n[{"key": "value"}]\n```'
        result = _parse_json_response(raw)
        assert result[0]["key"] == "value"

    def test_parse_json_response_single_object(self):
        from src.extraction import _parse_json_response

        raw = '{"key": "value"}'
        result = _parse_json_response(raw)
        assert isinstance(result, dict)

    def test_parse_json_response_invalid_raises(self):
        from src.extraction import _parse_json_response

        with pytest.raises((ValueError, Exception)):
            _parse_json_response("not json at all !!!!")

    def test_ensure_source_adds_missing_key(self):
        from src.extraction import _ensure_source

        item = {"study_name": "Test"}
        result = _ensure_source(item)
        assert "source" in result
        assert result["source"]["page_number"] == 0

    def test_ensure_source_preserves_existing(self):
        from src.extraction import _ensure_source

        item = {"source": {"page_number": 7, "quote": "Some quote."}}
        result = _ensure_source(item)
        assert result["source"]["page_number"] == 7

    def test_ensure_source_handles_non_dict(self):
        from src.extraction import _ensure_source

        item = {"source": "not a dict"}
        result = _ensure_source(item)
        assert isinstance(result["source"], dict)

    def test_salvage_uc2_from_chunks_extracts_method_and_clusters(self):
        from src.extraction import _salvage_uc2_from_chunks

        chunks = [
            {
                "source": "Donzelli2019.pdf",
                "page_number": 8,
                "section_name": "Results",
                "text": (
                    "We identified 4 clusters using k-means clustering in sepsis patients. "
                    "Cluster A had lower SOFA and lactate."
                ),
            }
        ]
        rows = _salvage_uc2_from_chunks(chunks)
        assert rows
        assert rows[0]["study_name"] == "Donzelli2019"
        assert rows[0]["clustering_method"] == "k-means"
        assert rows[0]["num_clusters"] == "4"
        assert rows[0]["source"]["page_number"] == 8

    def test_salvage_uc2_from_chunks_returns_empty_for_irrelevant_text(self):
        from src.extraction import _salvage_uc2_from_chunks

        chunks = [
            {
                "source": "paper.pdf",
                "page_number": 1,
                "text": "This study reports antibiotic stewardship and ICU LOS.",
            }
        ]
        rows = _salvage_uc2_from_chunks(chunks)
        assert rows == []

    def test_missing_required_fields_detects_blank_required_values(self):
        from src.extraction import _missing_required_fields

        item = {
            "study_name": "Study A",
            "population": "",
            "sample_size": "N=100",
            "setting": "ICU",
            "predictor": "Lactate",
            "outcome_definition": "28-day mortality",
            "effect_size": "OR 1.8",
            "source": {"page_number": 3, "quote": ""},
        }
        missing = _missing_required_fields(item, 1)
        assert "population" in missing
        assert "source.quote" in missing


# ---------------------------------------------------------------------------
# src.analytics tests
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_build_contradiction_graph_detects_opposite_effect_direction(self):
        from src.analytics import build_contradiction_graph

        evidence = [
            {
                "study_name": "A 2023",
                "predictor": "Lactate",
                "outcome_definition": "28-day mortality",
                "effect_size": "OR 2.1",
                "population": "ICU",
                "source": {"page_number": 2, "quote": "Lactate increased mortality."},
                "_confidence_score": 80,
            },
            {
                "study_name": "B 2024",
                "predictor": "Lactate",
                "outcome_definition": "28-day mortality",
                "effect_size": "OR 0.7",
                "population": "ICU",
                "source": {"page_number": 4, "quote": "Lactate was protective."},
                "_confidence_score": 75,
            },
        ]
        graph = build_contradiction_graph(1, evidence)
        assert graph["summary"]["contradiction_count"] >= 1
        assert "opposite_effect_direction" in graph["edges"][0]["reasons"]

    def test_build_evidence_knowledge_graph_has_strongest_path(self):
        from src.analytics import build_evidence_knowledge_graph

        evidence = [
            {
                "study_name": "A 2023",
                "biomarker_or_score": "IL-6",
                "biomarker_name": "IL-6",
                "outcome": "28-day mortality",
                "population": "ICU sepsis",
                "effect_size": "HR 4.1",
                "_confidence_score": 92,
            }
        ]
        kg = build_evidence_knowledge_graph(3, evidence)
        assert kg["summary"]["entity_count"] >= 4
        assert kg["summary"]["relation_count"] >= 3
        assert kg["summary"]["strongest_evidence_path"] is not None


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_has_expected_fields(self):
        from config import Config

        c = Config()
        assert hasattr(c, "OPENROUTER_API_KEY")
        assert hasattr(c, "OPENROUTER_BASE_URL")
        assert hasattr(c, "DEFAULT_MODEL")
        assert hasattr(c, "PDF_DIR")
        assert hasattr(c, "CHROMA_PERSIST_DIR")
        assert hasattr(c, "CHUNK_SIZE")
        assert hasattr(c, "CHUNK_OVERLAP")
        assert hasattr(c, "TOP_K_CHUNKS")

    def test_default_base_url(self):
        from config import Config

        c = Config()
        assert "openrouter.ai" in c.OPENROUTER_BASE_URL

    def test_chunk_size_positive(self):
        from config import Config

        c = Config()
        assert c.CHUNK_SIZE > 0
        assert c.CHUNK_OVERLAP > 0
        assert c.CHUNK_OVERLAP < c.CHUNK_SIZE
