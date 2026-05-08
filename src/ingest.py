"""
Structure-aware PDF ingestion for scientific/medical papers.

Pipeline overview:
1) Parse PDFs with `unstructured` in layout-aware mode.
2) Route detected tables/images through OpenRouter VLM for markdown summaries.
3) Build section-aware parent documents.
4) Split parent documents into child chunks for vector indexing.

The return format remains backward-compatible with the rest of the app
(`text`, `source`, `page_number`, `chunk_id`) while adding richer metadata
(parent IDs, section names, evidence origin, etc.).
"""

from __future__ import annotations

import base64
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import config

try:
    from unstructured.partition.pdf import partition_pdf
except Exception:  # noqa: BLE001
    partition_pdf = None


# ---------------------------------------------------------------------------
# OpenRouter VLM
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[Dict[str, Any]], None]

def _get_openrouter_client() -> OpenAI:
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )


def _guess_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    return "image/png"


def _to_data_url(image_bytes: bytes) -> str:
    mime = _guess_mime(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _new_ingest_report(total_documents: int = 0) -> Dict[str, Any]:
    return {
        "documents_discovered": total_documents,
        "documents_started": 0,
        "documents_completed": 0,
        "documents_failed": 0,
        "document_parse_errors": 0,
        "pages_extracted": 0,
        "parents_created": 0,
        "chunks_created": 0,
        "tables_processed": 0,
        "images_processed": 0,
        "table_parse_failures": 0,
        "image_parse_failures": 0,
        "vlm_successes": 0,
        "vlm_failures": 0,
        "vlm_empty": 0,
        "vlm_unavailable": 0,
        "fallback_used": False,
    }


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    callback({"event": event, **payload})


def _vlm_summary_status(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return "empty"
    if text.startswith("[VLM SUMMARY FAILED]"):
        return "failed"
    if text.startswith("[VLM SUMMARY EMPTY]"):
        return "empty"
    if text.startswith("[VLM SUMMARY UNAVAILABLE]"):
        return "unavailable"
    return "success"


def _record_visual_stats(
    report: Dict[str, Any] | None,
    *,
    visual_type: str,
    has_payload: bool,
    summary: str,
) -> None:
    if report is None:
        return

    counter_key = "tables_processed" if visual_type == "Table" else "images_processed"
    report[counter_key] += 1

    if not has_payload:
        parse_key = "table_parse_failures" if visual_type == "Table" else "image_parse_failures"
        report[parse_key] += 1

    status = _vlm_summary_status(summary)
    if status == "success":
        report["vlm_successes"] += 1
    elif status == "failed":
        report["vlm_failures"] += 1
    elif status == "unavailable":
        report["vlm_unavailable"] += 1
    elif status == "empty":
        report["vlm_empty"] += 1


def _summarize_visual_with_vlm(
    image_bytes: bytes,
    *,
    visual_type: str,
    title: str,
    section: str,
    page_number: int,
) -> str:
    if not image_bytes:
        return ""

    model = getattr(config, "OPENROUTER_VLM_MODEL", "google/gemini-2.5-flash-image")
    if not config.OPENROUTER_API_KEY:
        return (
            f"[VLM SUMMARY UNAVAILABLE]\n"
            f"OpenRouter API key missing for {visual_type.lower()} summary."
        )

    prompt = (
        "You are extracting evidence from a clinical paper. "
        "Return a detailed markdown summary of this visual element. "
        "For tables: preserve key rows/columns and numeric values. "
        "For charts/curves: summarize trends, axes, cohorts, and clinical findings. "
        "Be faithful to the image. Do not hallucinate."
    )

    user_text = (
        f"Document title: {title or 'Unknown'}\n"
        f"Section: {section or 'Unknown'}\n"
        f"Page: {page_number or 0}\n"
        f"Element type: {visual_type}\n"
        "Provide markdown with a short heading and bullet points."
    )

    try:
        client = _get_openrouter_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": _to_data_url(image_bytes)},
                        },
                    ],
                },
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if content:
            return content
    except Exception as exc:  # noqa: BLE001
        return f"[VLM SUMMARY FAILED]\nCould not summarize {visual_type.lower()}: {exc}"

    return f"[VLM SUMMARY EMPTY]\nNo summary produced for {visual_type.lower()}."


# ---------------------------------------------------------------------------
# Unstructured parsing helpers
# ---------------------------------------------------------------------------

def _partition_layout_aware(pdf_path: str) -> List[Any]:
    if partition_pdf is None:
        raise ImportError(
            "unstructured is not installed. Install requirements and retry ingestion."
        )

    kwargs = {
        "filename": pdf_path,
        "strategy": "hi_res",
        "infer_table_structure": True,
    }

    # Best-effort extraction of image/table payloads (API differs slightly by version).
    variants = [
        {
            **kwargs,
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_to_payload": True,
        },
        {
            **kwargs,
            "extract_images_in_pdf": True,
        },
        kwargs,
    ]

    for variant in variants:
        try:
            return partition_pdf(**variant)
        except TypeError:
            continue

    return partition_pdf(**kwargs)


def _element_category(element: Any) -> str:
    cat = getattr(element, "category", None)
    if cat:
        return str(cat)
    return element.__class__.__name__


def _element_text(element: Any) -> str:
    return (getattr(element, "text", "") or "").strip()


def _metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _extract_image_bytes(element: Any) -> bytes | None:
    metadata = getattr(element, "metadata", None)
    b64_payload = _metadata_value(metadata, "image_base64")
    if b64_payload:
        try:
            return base64.b64decode(b64_payload)
        except Exception:  # noqa: BLE001
            return None

    image_path = _metadata_value(metadata, "image_path")
    if image_path:
        try:
            return Path(image_path).read_bytes()
        except Exception:  # noqa: BLE001
            return None

    return None


def _looks_like_section_header(text: str) -> bool:
    if not text:
        return False
    clean = text.strip().lower().rstrip(":")
    canonical = {
        "abstract",
        "introduction",
        "background",
        "methods",
        "materials and methods",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
    }
    if clean in canonical:
        return True
    return len(text) <= 80 and text[:1].isupper() and text.upper() != text


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Structural extraction
# ---------------------------------------------------------------------------

def extract_pages(
    pdf_path: str,
    report: Dict[str, Any] | None = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """
    Parse a PDF into page-level records with layout-aware extraction.

    Returns page dicts with keys:
        - text
        - page_number
        - source
        - source_path
        - section_name
        - document_title
        - evidence_origin (raw_text | vlm_summary)
    """
    path = Path(pdf_path).resolve()
    source_name = path.stem

    elements = _partition_layout_aware(str(path))

    document_title = source_name
    current_section = "Unknown"
    page_fragments: Dict[int, List[Dict[str, str]]] = defaultdict(list)

    for element in elements:
        category = _element_category(element)
        metadata = getattr(element, "metadata", None)
        page_number = _safe_int(_metadata_value(metadata, "page_number", 0), 0)
        if page_number <= 0:
            page_number = 1

        text = _element_text(element)

        if category == "Title" and text and document_title == source_name:
            document_title = text

        if category in {"Title", "Header"} and _looks_like_section_header(text):
            current_section = text.strip().rstrip(":")

        if category in {"Image", "Table"}:
            image_bytes = _extract_image_bytes(element)
            summary = _summarize_visual_with_vlm(
                image_bytes or b"",
                visual_type=category,
                title=document_title,
                section=current_section,
                page_number=page_number,
            )
            _record_visual_stats(
                report,
                visual_type=category,
                has_payload=bool(image_bytes),
                summary=summary,
            )
            if summary.strip():
                page_fragments[page_number].append(
                    {
                        "text": f"[{category.upper()} SUMMARY]\n{summary}",
                        "section_name": current_section,
                        "evidence_origin": "vlm_summary",
                    }
                )
            continue

        if text:
            page_fragments[page_number].append(
                {
                    "text": text,
                    "section_name": current_section,
                    "evidence_origin": "raw_text",
                }
            )

    pages: List[Dict[str, Any]] = []
    for page_number in sorted(page_fragments.keys()):
        fragments = page_fragments[page_number]
        merged = "\n\n".join(f["text"] for f in fragments if f["text"].strip()).strip()
        if not merged:
            continue

        # Use the latest known section for the page; mixed pages are marked mixed origin.
        section_name = fragments[-1]["section_name"] if fragments else "Unknown"
        origins = {f["evidence_origin"] for f in fragments}
        evidence_origin = "mixed" if len(origins) > 1 else next(iter(origins))

        pages.append(
            {
                "text": merged,
                "page_number": page_number,
                "source": source_name,
                "source_path": str(path),
                "section_name": section_name,
                "document_title": document_title,
                "evidence_origin": evidence_origin,
            }
        )

    if report is not None:
        report["pages_extracted"] += len(pages)

    return pages


# ---------------------------------------------------------------------------
# Backward-compatible page chunker (kept for tests + compatibility)
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1500,
    overlap: int = 300,
) -> List[Dict[str, Any]]:
    """Split page text into overlapping character chunks."""
    chunks: List[Dict[str, Any]] = []

    for page_data in pages:
        text = (page_data.get("text") or "").strip()
        if not text:
            continue

        page_num = page_data.get("page_number", 0)
        source = page_data.get("source", "unknown")

        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()

            if piece:
                chunks.append(
                    {
                        "text": piece,
                        "page_number": page_num,
                        "source": source,
                        "chunk_id": f"{source}_p{page_num}_{start}",
                    }
                )

            next_start = end - overlap
            if next_start <= start:
                break
            start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Parent/child chunk construction
# ---------------------------------------------------------------------------

def _group_pages_for_parents(
    pages: Iterable[Dict[str, Any]],
    parent_target_chars: int,
) -> List[Dict[str, Any]]:
    parents: List[Dict[str, Any]] = []

    current: Dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n\n".join(current.pop("parts")).strip()
        if not text:
            current = None
            return
        current["text"] = text
        current["chunk_type"] = "parent"
        current["parent_id"] = current.get("parent_id") or f"parent_{uuid4().hex}"
        parents.append(current)
        current = None

    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue

        section = page.get("section_name") or "Unknown"
        source = page.get("source") or "unknown"
        title = page.get("document_title") or source
        page_number = _safe_int(page.get("page_number", 0), 0)
        origin = page.get("evidence_origin", "raw_text")

        if current is None:
            current = {
                "source": source,
                "document_title": title,
                "section_name": section,
                "page_start": page_number,
                "page_end": page_number,
                "evidence_origin": origin,
                "parts": [text],
            }
            continue

        same_section = current["section_name"] == section
        projected_size = sum(len(p) for p in current["parts"]) + len(text)

        if not same_section or projected_size > parent_target_chars:
            flush()
            current = {
                "source": source,
                "document_title": title,
                "section_name": section,
                "page_start": page_number,
                "page_end": page_number,
                "evidence_origin": origin,
                "parts": [text],
            }
            continue

        current["parts"].append(text)
        current["page_end"] = page_number
        if current.get("evidence_origin") != origin:
            current["evidence_origin"] = "mixed"

    flush()
    return parents


def _child_chunks_from_parents(
    parents: List[Dict[str, Any]],
    *,
    child_target_chars: int,
    child_overlap_chars: int,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_target_chars,
        chunk_overlap=child_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    children: List[Dict[str, Any]] = []
    for parent in parents:
        source = parent["source"]
        page_number = parent.get("page_start", 0)
        parent_text = parent.get("text", "")
        parent_id = parent["parent_id"]

        segments = splitter.split_text(parent_text)
        for i, segment in enumerate(segments):
            text = segment.strip()
            if not text:
                continue
            children.append(
                {
                    "text": text,
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": f"{source}_{parent_id}_c{i}",
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "parent_chunk_id": parent_id,
                    "parent_text": parent_text,
                    "section_name": parent.get("section_name", "Unknown"),
                    "document_title": parent.get("document_title", source),
                    "evidence_origin": parent.get("evidence_origin", "raw_text"),
                    "page_start": parent.get("page_start", page_number),
                    "page_end": parent.get("page_end", page_number),
                }
            )

    return children


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------

def ingest_pdfs(
    pdf_dir: str,
    chunk_size: int = 1500,
    overlap: int = 300,
    progress_callback: ProgressCallback | None = None,
    return_report: bool = False,
    **_: Any,
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ingest all PDFs from `pdf_dir` and return child chunks.

    Parent documents are built section-wise and split into child chunks for
    vector indexing. Each child chunk carries metadata linking back to its
    parent chunk for context-rich downstream retrieval.
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    report = _new_ingest_report(len(pdf_files))

    if not pdf_files:
        return ([], report) if return_report else []

    # ~1000 tokens parent, ~200 tokens child (roughly 4 chars/token).
    parent_target_chars = max(
        1200,
        int(getattr(config, "PARENT_CHUNK_TOKENS", 1000)) * 4,
    )
    child_target_chars = max(400, int(getattr(config, "CHILD_CHUNK_TOKENS", 200)) * 4)
    child_overlap_chars = max(80, int(overlap))

    all_children: List[Dict[str, Any]] = []
    for idx, pdf_path in enumerate(pdf_files, start=1):
        report["documents_started"] += 1
        _emit_progress(
            progress_callback,
            "document_started",
            document=pdf_path.name,
            index=idx,
            total=len(pdf_files),
            report=report.copy(),
        )
        try:
            pages = extract_pages(str(pdf_path), report=report)
            if not pages:
                report["documents_completed"] += 1
                _emit_progress(
                    progress_callback,
                    "document_completed",
                    document=pdf_path.name,
                    index=idx,
                    total=len(pdf_files),
                    pages=0,
                    parents=0,
                    chunks=0,
                    report=report.copy(),
                )
                continue

            parents = _group_pages_for_parents(pages, parent_target_chars=parent_target_chars)
            children = _child_chunks_from_parents(
                parents,
                child_target_chars=child_target_chars,
                child_overlap_chars=child_overlap_chars,
            )
            report["parents_created"] += len(parents)
            report["chunks_created"] += len(children)
            report["documents_completed"] += 1
            all_children.extend(children)
            _emit_progress(
                progress_callback,
                "document_completed",
                document=pdf_path.name,
                index=idx,
                total=len(pdf_files),
                pages=len(pages),
                parents=len(parents),
                chunks=len(children),
                report=report.copy(),
            )
        except Exception as exc:  # noqa: BLE001
            report["documents_failed"] += 1
            report["document_parse_errors"] += 1
            print(f"[ingest] WARNING: could not parse {pdf_path.name}: {exc}")
            _emit_progress(
                progress_callback,
                "document_failed",
                document=pdf_path.name,
                index=idx,
                total=len(pdf_files),
                error=str(exc),
                report=report.copy(),
            )

    # Fallback to legacy chunking if layout-aware ingestion produced no chunks.
    if not all_children:
        report["fallback_used"] = True
        legacy_chunks: List[Dict[str, Any]] = []
        for pdf_path in pdf_files:
            try:
                pages = extract_pages(str(pdf_path))
                legacy_chunks.extend(
                    chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest] WARNING: fallback chunking failed for {pdf_path.name}: {exc}")
        report["chunks_created"] = len(legacy_chunks)
        return (legacy_chunks, report) if return_report else legacy_chunks

    return (all_children, report) if return_report else all_children
