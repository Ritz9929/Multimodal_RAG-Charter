"""
Base Extractor & Unified Data Schema
======================================
All format-specific extractors inherit from BaseExtractor
and return the same ExtractionResult schema.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents a single image extracted from any document format."""
    filename: str              # e.g. "page3_img1.png"
    filepath: str              # full path inside mock_s3_storage
    page_number: int           # 0-indexed page/slide/section the image came from
    position_index: int        # insertion order on that page
    source_type: str = "embedded"  # "embedded", "rendered", "slide_image", "shape_image"
    summary: str = ""          # filled in by the summarizer


@dataclass
class ExtractedTable:
    """Represents a structured table extracted from any document format."""
    markdown: str              # Full markdown-formatted table
    page_number: int           # Page/slide/section index
    position_index: int        # Insertion order on that page
    source_type: str = "table" # "table"
    row_count: int = 0
    col_count: int = 0
    caption: str = ""          # Table caption if available


@dataclass
class ExtractionResult:
    """Unified extraction output for any document format."""
    page_texts: dict[int, str] = field(default_factory=dict)   # {page/section: text}
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_format: str = ""    # "pdf", "csv", "pptx", "docx"


class BaseExtractor(ABC):
    """Abstract base class for all document format extractors."""

    def __init__(self, output_dir: str = "./mock_s3_storage", **kwargs):
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(self, file_path: str, source_doc: str = "") -> ExtractionResult:
        """
        Parse the input file and return a unified ExtractionResult.

        Args:
            file_path:  Absolute or relative path to the input file.
            source_doc: Filename for tracking (defaults to basename of file_path).

        Returns:
            ExtractionResult with page_texts, images, tables populated.
        """
        ...

    def _make_doc_output_dir(self, source_doc: str) -> Path:
        """Create a document-specific subfolder for extracted artifacts."""
        folder = source_doc.replace(".", "_") if source_doc else "unknown"
        doc_dir = self.base_output_dir / folder
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir
