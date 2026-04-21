"""
CSV Extractor
==============
Uses pandas for CSV parsing with intelligent row-windowing.

Strategy:
  Page 0: Schema summary (column names, types, stats)
  Pages 1..N: Rows in windows of ROWS_PER_PAGE as markdown tables

Why markdown tables?
  Raw CSV text loses column alignment in embeddings. Markdown format
  preserves header-to-value relationships, improving retrieval accuracy
  for column-specific queries like "show me rows where Revenue > 1M".
"""

import logging
from pathlib import Path

import pandas as pd
from .base import BaseExtractor, ExtractionResult, ExtractedTable

logger = logging.getLogger(__name__)


class CSVExtractor(BaseExtractor):
    """
    CSV extraction with intelligent row-windowing.

    Strategy:
      Page 0: Schema summary (column names, types, stats)
      Pages 1..N: Rows in windows of ROWS_PER_PAGE as markdown tables
    """

    ROWS_PER_PAGE = 50  # Number of rows per "page" (configurable)

    def __init__(self, output_dir: str = "./mock_s3_storage", rows_per_page: int = 50, **kwargs):
        super().__init__(output_dir)
        self.ROWS_PER_PAGE = rows_per_page

    def _build_schema_summary(self, df: pd.DataFrame, file_path: str) -> str:
        """Build a comprehensive schema summary for Page 0."""
        lines = [
            f"# CSV Schema Summary: {Path(file_path).name}",
            f"",
            f"**Total Rows**: {len(df)}",
            f"**Total Columns**: {len(df.columns)}",
            f"",
            f"## Column Definitions",
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_pct = (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0

            lines.append(f"- **{col}** ({dtype}): {non_null} non-null values ({null_pct:.1f}% missing)")

            # Sample values
            samples = df[col].dropna().unique()[:5]
            if len(samples) > 0:
                sample_str = ", ".join(str(s) for s in samples)
                lines.append(f"  - Sample values: {sample_str}")

        # Numeric statistics
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            lines.append("")
            lines.append("## Numeric Column Statistics")
            stats_df = df[numeric_cols].describe()
            lines.append(stats_df.to_markdown())

        return "\n".join(lines)

    def extract(self, file_path: str, source_doc: str = "") -> ExtractionResult:
        """Parse the CSV and return schema + windowed rows."""
        result = ExtractionResult(source_format="csv")

        if not source_doc:
            source_doc = Path(file_path).name

        # Read CSV with pandas
        df = pd.read_csv(file_path)
        logger.info(f"Opened CSV: {file_path} ({len(df)} rows, {len(df.columns)} cols)")

        # Page 0: Schema summary
        result.page_texts[0] = self._build_schema_summary(df, file_path)

        # Pages 1..N: Row windows as markdown tables
        for i, start in enumerate(range(0, len(df), self.ROWS_PER_PAGE)):
            window = df.iloc[start:start + self.ROWS_PER_PAGE]
            page_num = i + 1

            md_table = window.to_markdown(index=False)
            header = f"## Rows {start + 1}–{start + len(window)} of {len(df)}\n\n"
            result.page_texts[page_num] = header + md_table

            result.tables.append(ExtractedTable(
                markdown=md_table,
                page_number=page_num,
                position_index=0,
                row_count=len(window),
                col_count=len(window.columns),
            ))

        result.metadata = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "pages_created": len(result.page_texts),
            "rows_per_page": self.ROWS_PER_PAGE,
        }

        logger.info(f"CSV extraction complete — {len(result.page_texts)} pages created")
        return result
