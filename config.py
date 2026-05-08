"""
Configuration management for Sepsis Atlas.

Loads settings from environment variables / .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    # OpenRouter API
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Default LLM model (free tier on OpenRouter)
    DEFAULT_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_MODEL", "anthropic/claude-3.5-sonnet"
        )
    )

    # File paths
    PDF_DIR: str = field(default_factory=lambda: os.getenv("PDF_DIR", "./pdfs"))
    CHROMA_PERSIST_DIR: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    # Chunking parameters
    CHUNK_SIZE: int = 2000   # characters per chunk
    CHUNK_OVERLAP: int = 500  # overlap between consecutive chunks

    # Retrieval
    TOP_K_CHUNKS: int = 50   # number of chunks to retrieve per query

    # Table extraction
    TABLE_EXTRACTION_ENABLED: bool = field(
        default_factory=lambda: _env_bool("TABLE_EXTRACTION_ENABLED", True)
    )

    # OCR (for figure/table images on low-text pages)
    OCR_ENABLED: bool = field(
        default_factory=lambda: _env_bool("OCR_ENABLED", True)
    )
    OCR_MIN_TEXT_CHARS: int = field(
        default_factory=lambda: int(os.getenv("OCR_MIN_TEXT_CHARS", "250"))
    )
    OCR_MIN_IMAGES: int = field(
        default_factory=lambda: int(os.getenv("OCR_MIN_IMAGES", "1"))
    )
    OCR_LANG: str = field(
        default_factory=lambda: os.getenv("OCR_LANG", "eng")
    )
    OCR_MIN_IMAGE_COVERAGE: float = field(
        default_factory=lambda: float(os.getenv("OCR_MIN_IMAGE_COVERAGE", "0.6"))
    )
    OCR_MAX_DRAWINGS: int = field(
        default_factory=lambda: int(os.getenv("OCR_MAX_DRAWINGS", "100"))
    )


# Singleton config instance
config = Config()
