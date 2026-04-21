"""
PDF Extractor
==============
Uses PyMuPDF (fitz) to parse PDF files.

Extraction methods:
  - Text: page.get_text("text")
  - Embedded raster images: page.get_images() (photos, logos)
  - Vector graphics: page.get_pixmap() (charts, graphs rendered as PNG)
  - Tables: page.find_tables() → markdown format
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from .base import BaseExtractor, ExtractionResult, ExtractedImage, ExtractedTable

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    PDF extraction using PyMuPDF.

    Extraction methods:
      - Text: page.get_text("text")
      - Embedded raster images: page.get_images() (photos, logos)
      - Vector graphics: page.get_pixmap() (charts, graphs rendered as PNG)
      - Tables: page.find_tables() → markdown format
    """

    # Minimum number of vector drawing operations on a page to consider it
    # as having a chart/graph/table worth rendering.
    MIN_DRAWINGS_THRESHOLD = 10

    def __init__(self, output_dir: str = "./mock_s3_storage", render_dpi: int = 200, **kwargs):
        super().__init__(output_dir)
        self.render_dpi = render_dpi

    def _has_vector_graphics(self, page) -> bool:
        """Check if a page has significant vector drawings (charts/graphs/tables)."""
        try:
            drawings = page.get_drawings()
            return len(drawings) >= self.MIN_DRAWINGS_THRESHOLD
        except Exception:
            return False

    def _render_page_as_image(self, page, page_num: int, img_index: int,
                               doc_output_dir: Path) -> ExtractedImage:
        """Render an entire page as a PNG image (captures vector graphics)."""
        zoom = self.render_dpi / 72  # 72 DPI is the PDF default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        filename = f"page{page_num}_rendered.png"
        filepath = doc_output_dir / filename
        pix.save(str(filepath))

        return ExtractedImage(
            filename=filename,
            filepath=str(filepath),
            page_number=page_num,
            position_index=img_index,
            source_type="rendered",
        )

    def _extract_tables(self, page, page_num: int) -> list[ExtractedTable]:
        """Extract structured tables from a PDF page using PyMuPDF."""
        tables = []
        try:
            tab_finder = page.find_tables()
            for idx, table in enumerate(tab_finder.tables):
                df = table.to_pandas()
                if df.empty:
                    continue
                markdown = df.to_markdown(index=False)
                tables.append(ExtractedTable(
                    markdown=markdown,
                    page_number=page_num,
                    position_index=idx,
                    row_count=len(df),
                    col_count=len(df.columns),
                ))
        except Exception:
            pass  # find_tables() may not be available in older PyMuPDF
        return tables

    def extract(self, file_path: str, source_doc: str = "") -> ExtractionResult:
        """Parse the PDF and return text + images + tables."""
        doc = fitz.open(file_path)
        result = ExtractionResult(source_format="pdf")

        if not source_doc:
            source_doc = Path(file_path).name
        doc_output_dir = self._make_doc_output_dir(source_doc)

        logger.info(f"Opened PDF: {file_path} ({len(doc)} pages)")
        logger.info(f"  Images will be saved to: {doc_output_dir}")

        for page_num in range(len(doc)):
            page = doc[page_num]

            # ── Extract text ──
            result.page_texts[page_num] = page.get_text("text")

            # ── Extract tables (structured) ──
            page_tables = self._extract_tables(page, page_num)
            result.tables.extend(page_tables)

            # ── Extract embedded raster images ──
            image_list = page.get_images(full=True)
            has_embedded_images = False
            seen_xrefs = set()
            saved_img_count = 0
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue
                    image_bytes = base_image["image"]
                    if len(image_bytes) < 1024:
                        continue

                    image_ext = base_image.get("ext", "png")
                    filename = f"page{page_num}_img{saved_img_count}.{image_ext}"
                    filepath = doc_output_dir / filename

                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    result.images.append(ExtractedImage(
                        filename=filename,
                        filepath=str(filepath),
                        page_number=page_num,
                        position_index=saved_img_count,
                        source_type="embedded",
                    ))
                    has_embedded_images = True
                    saved_img_count += 1
                    logger.info(f"  Saved embedded image: {filename}")

                except Exception as e:
                    logger.warning(f"  Skipping image xref {xref} on page {page_num}: {e}")

            # ── Render pages with vector graphics ──
            if self._has_vector_graphics(page) and not has_embedded_images:
                next_index = len(image_list)
                rendered = self._render_page_as_image(
                    page, page_num, next_index, doc_output_dir
                )
                result.images.append(rendered)
                logger.info(f"  Rendered page {page_num} as image (vector graphics detected)")

        doc.close()

        result.metadata = {
            "page_count": len(result.page_texts),
            "embedded_images": sum(1 for i in result.images if i.source_type == "embedded"),
            "rendered_pages": sum(1 for i in result.images if i.source_type == "rendered"),
            "tables_found": len(result.tables),
        }

        embedded_count = result.metadata["embedded_images"]
        rendered_count = result.metadata["rendered_pages"]
        logger.info(
            f"Extraction complete — {len(result.page_texts)} pages, "
            f"{embedded_count} embedded images, {rendered_count} rendered pages, "
            f"{len(result.tables)} tables"
        )
        return result
