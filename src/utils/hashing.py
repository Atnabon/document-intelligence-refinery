"""Content hashing utilities for chunk integrity verification.

Provides SHA-256 based hashing for:
- Individual chunks (content + page refs + bounding box)
- Full documents (file-level hashing)
- Integrity verification (detect tampering)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.provenance import BoundingBox


def generate_chunk_hash(
    content: str,
    page_refs: List[int],
    bbox: Optional["BoundingBox"] = None,
) -> str:
    """Generate a deterministic SHA-256 hash for a chunk.

    The hash includes content, page references, and bounding box so that
    any change to the chunk's content or location produces a new hash.

    Args:
        content: The text content of the chunk.
        page_refs: List of page numbers the chunk spans.
        bbox: Optional bounding box.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(content.encode("utf-8"))
    hasher.update(str(sorted(page_refs)).encode("utf-8"))

    if bbox is not None:
        bbox_str = f"{bbox.x0:.2f},{bbox.y0:.2f},{bbox.x1:.2f},{bbox.y1:.2f}"
        hasher.update(bbox_str.encode("utf-8"))

    return hasher.hexdigest()


def verify_chunk_hash(chunk) -> bool:
    """Verify that a chunk's content_hash matches its current content.

    Args:
        chunk: An LDU-like object with content, page_number or page_refs,
               bbox, and content_hash attributes.

    Returns:
        True if the hash matches, False if the content has been tampered with.
    """
    page_refs = getattr(chunk, "page_refs", None)
    if page_refs is None:
        page_refs = [getattr(chunk, "page_number", 1)]
    bbox = getattr(chunk, "bbox", None)

    expected = generate_chunk_hash(chunk.content, page_refs, bbox)
    return chunk.content_hash == expected


def generate_document_hash(file_path: str | Path) -> str:
    """Generate a SHA-256 hash of an entire file.

    Reads the file in 8 KB blocks for memory efficiency.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(8192)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def hash_file(file_path: str | Path) -> str:
    """Alias for generate_document_hash (convenience)."""
    return generate_document_hash(file_path)


def hash_text(text: str) -> str:
    """Generate a short (16-char) SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
