"""
File Router
============
Dispatches files to the appropriate extractor based on extension.

Usage:
    router = FileRouter(output_dir="./mock_s3_storage")
    result = router.extract("report.pdf")
    result = router.extract("data.csv")
    result = router.extract("deck.pptx")
"""

import logging
from pathlib import Path

from .base import BaseExtractor, ExtractionResult
from .pdf import PDFExtractor
from .csv_ext import CSVExtractor
from .docx_ext import DOCXExtractor
from .pptx_ext import PPTXExtractor
from .excel_ext import ExcelExtractor

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".docx", ".pptx", ".xlsx", ".xls"}

# Extension → Extractor class mapping
EXTRACTOR_MAP: dict[str, type[BaseExtractor]] = {
    ".pdf": PDFExtractor,
    ".csv": CSVExtractor,
    ".docx": DOCXExtractor,
    ".pptx": PPTXExtractor,
    ".xlsx": ExcelExtractor,
    ".xls": ExcelExtractor,
}


class FileRouter:
    """
    Dispatches files to the appropriate extractor based on extension.

    Usage:
        router = FileRouter(output_dir="./mock_s3_storage")
        result = router.extract("report.pdf")
        result = router.extract("data.csv")
        result = router.extract("deck.pptx")
    """

    def __init__(self, output_dir: str = "./mock_s3_storage", **kwargs):
        self.output_dir = output_dir
        self.kwargs = kwargs  # Passed to extractor constructors (e.g. render_dpi)

    def extract(self, file_path: str, source_doc: str = "") -> ExtractionResult:
        """Route file to the appropriate extractor."""
        ext = Path(file_path).suffix.lower()

        if ext not in EXTRACTOR_MAP:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        extractor_cls = EXTRACTOR_MAP[ext]
        extractor = extractor_cls(output_dir=self.output_dir, **self.kwargs)

        logger.info(f"Routing '{Path(file_path).name}' → {extractor_cls.__name__}")
        return extractor.extract(file_path, source_doc=source_doc)

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file format is supported."""
        return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS
