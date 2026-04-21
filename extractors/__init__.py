"""
Document Extractors Package
============================
Provides format-specific extractors with a unified ExtractionResult schema.

Usage:
    from extractors import FileRouter, SUPPORTED_EXTENSIONS

    router = FileRouter()
    result = router.extract("report.pdf")
"""

from .base import (
    BaseExtractor,
    ExtractionResult,
    ExtractedImage,
    ExtractedTable,
)
from .router import FileRouter, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExtractedImage",
    "ExtractedTable",
    "FileRouter",
    "SUPPORTED_EXTENSIONS",
]
