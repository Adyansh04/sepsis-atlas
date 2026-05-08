"""
Vector-store retrieval module using ChromaDB.

ChromaDB is used because it is fully local (no server needed), supports
persistent storage out-of-the-box, and ships with a built-in sentence-
transformer embedding function so no extra LLM calls are required for indexing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "sepsis_papers"

# Use the lightweight all-MiniLM-L6-v2 model (~80 MB, no GPU required).
# It is downloaded automatically on first use.
# _EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def _get_collection(
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_chunks(
    chunks: List[Dict[str, Any]],
    persist_dir: str = "./chroma_db",
) -> None:
    """
    Add (or update) *chunks* in the persistent ChromaDB collection.

    Uses ``upsert`` so re-indexing the same PDFs is idempotent.
    Chunks are batched to avoid memory spikes on large corpora.
    """
    if not chunks:
        return

    client = _get_client(persist_dir)
    collection = _get_collection(client)

    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page_number": c["page_number"]}
        for c in chunks
    ]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            documents=documents[i : i + batch_size],
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def query_chunks(
    query: str,
    persist_dir: str = "./chroma_db",
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieve the *n_results* most relevant chunks for *query*.

    Returns a list of dicts with keys: ``text``, ``source``,
    ``page_number``, ``distance``.
    """
    client = _get_client(persist_dir)
    collection = _get_collection(client)

    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, total),
    )

    chunks: List[Dict[str, Any]] = []
    if results and results.get("documents"):
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = (
                results["distances"][0][i]
                if results.get("distances")
                else None
            )
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page_number": meta.get("page_number", 0),
                    "distance": dist,
                }
            )

    return chunks


def _is_reference_chunk(text: str) -> bool:
    """Heuristic filter for reference/bibliography sections."""
    lower = text.lower()

    if "references" in lower or "bibliography" in lower:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    numeric_lines = sum(1 for line in lines if re.match(r"^\d+[\.\)]", line))
    if len(lines) >= 4 and numeric_lines / len(lines) >= 0.4:
        return True

    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", lower))
    citation_hits = sum(
        lower.count(token)
        for token in ("et al", "doi:", "pmid", "vol.", "pp.", "issn")
    )

    return year_hits >= 6 and citation_hits >= 2


def filter_chunks(
    chunks: List[Dict[str, Any]],
    *,
    exclude_references: bool = True,
    min_length: int = 200,
) -> List[Dict[str, Any]]:
    """Filter out reference-like or very short chunks."""
    filtered: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if len(text) < min_length:
            continue
        if exclude_references and _is_reference_chunk(text):
            continue
        filtered.append(chunk)
    return filtered


def keyword_search(
    keywords: List[str],
    persist_dir: str = "./chroma_db",
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fallback keyword search across the entire collection.

    Returns chunks that contain any keyword (case-insensitive), prioritised
    by the number of keyword hits.
    """
    client = _get_client(persist_dir)
    collection = _get_collection(client)

    total = collection.count()
    if total == 0:
        return []

    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    scored: List[Dict[str, Any]] = []
    for doc, meta in zip(documents, metadatas):
        lower = doc.lower()
        hits = sum(lower.count(k) for k in keywords)
        if hits <= 0:
            continue
        chunk = {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number", 0),
            "distance": None,
            "_keyword_hits": hits,
        }
        if _is_reference_chunk(doc):
            continue
        scored.append(chunk)

    scored.sort(key=lambda c: c["_keyword_hits"], reverse=True)
    for chunk in scored:
        chunk.pop("_keyword_hits", None)
    return scored[:n_results]


def query_chunks_enhanced(
    query: str,
    persist_dir: str = "./chroma_db",
    n_results: int = 10,
    *,
    keyword_fallback: List[str] | None = None,
    exclude_references: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with reference filtering + optional keyword fallback.
    """
    chunks = query_chunks(query, persist_dir=persist_dir, n_results=n_results)
    chunks = filter_chunks(chunks, exclude_references=exclude_references)

    if keyword_fallback and len(chunks) < n_results:
        keyword_hits = keyword_search(
            keyword_fallback, persist_dir=persist_dir, n_results=n_results
        )
        chunks = _merge_chunks(chunks, keyword_hits, limit=n_results)

    return chunks[:n_results]


def _merge_chunks(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Merge chunk lists while preserving order and avoiding duplicates."""
    seen = set()
    merged: List[Dict[str, Any]] = []

    for chunk in primary + secondary:
        key = (
            chunk.get("source"),
            chunk.get("page_number"),
            chunk.get("text", "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break

    return merged


def get_indexed_sources(persist_dir: str = "./chroma_db") -> List[str]:
    """Return the sorted list of source document names in the index."""
    try:
        client = _get_client(persist_dir)
        collection = _get_collection(client)

        if collection.count() == 0:
            return []

        all_meta = collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta})
    except Exception:  # noqa: BLE001
        return []


def get_collection_count(persist_dir: str = "./chroma_db") -> int:
    """Return total number of indexed chunks (0 if index does not exist)."""
    try:
        client = _get_client(persist_dir)
        collection = _get_collection(client)
        return collection.count()
    except Exception:  # noqa: BLE001
        return 0


def clear_collection(persist_dir: str = "./chroma_db") -> None:
    """Delete and recreate the collection (full re-index)."""
    client = _get_client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    _get_collection(client)
