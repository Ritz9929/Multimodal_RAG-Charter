"""
Multimodal RAG Ingestion Pipeline
==================================
A modular pipeline that:
  1. Extracts text, images, and tables from documents (PDF, CSV, DOCX, PPTX)
     via the extractors package (format-agnostic FileRouter)
  2. Summarizes images via a Vision Language Model (NVIDIA NIM API)
  3. Reassembles the document with image and table reference tags
  4. Chunks the text intelligently (never splitting reference tags)
  5. Stores chunks in a PGVector database (persistent)
"""

import os
import re
import hashlib
import json
import base64
import time
import logging
from pathlib import Path
from io import BytesIO

from PIL import Image
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from config import cfg
from extractors import FileRouter, ExtractionResult, ExtractedImage

logger = logging.getLogger(__name__)


# NOTE: PDFExtractor has been moved to extractors/pdf.py
# All format-specific extractors are in the extractors/ package.
# Use FileRouter to dispatch files to the correct extractor.


# ═══════════════════════════════════════════════════════════════════════════════
#  2. IMAGE SUMMARIZER (NVIDIA NIM API — OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageSummarizer:
    """
    Sends each extracted image to a Vision Language Model via the
    NVIDIA NIM Inference API and returns a detailed, factual summary.

    Uses NVIDIA NIM's OpenAI-compatible API — 40 RPM free tier.
    The model is swappable via the constructor.
    """

    DEFAULT_PROMPT = (
        "You are an expert data analyst and document parsing specialist. "
        "Your task is to analyze the provided image and extract its contents "
        "into a dense, highly detailed text summary.\n\n"
        "This summary will be embedded into a Vector Database for a "
        "Retrieval-Augmented Generation (RAG) system. It is critical that "
        "your output contains all exact keywords, numbers, and relationships "
        "present in the image so that semantic search can accurately find it later.\n\n"
        "Please analyze the image and output your response strictly using "
        "the following Markdown structure:\n\n"
        "### 1. Image Type & Core Subject\n"
        "[State the type of image: e.g., Bar Chart, Line Graph, Flowchart, "
        "Architecture Diagram, Financial Table, or Photograph. State the main "
        "title or core subject in one sentence.]\n\n"
        "### 2. Explicit Text & Labels\n"
        "[Extract and list all literal text visible in the image. This includes:\n"
        "- Main titles and subtitles\n"
        "- X and Y axis labels (including units of measurement)\n"
        "- Legend items\n"
        "- Node labels in flowcharts\n"
        "- Column and row headers in tables]\n\n"
        "### 3. Data & Relationships (The \"Meat\")\n"
        "[Translate the visual data into descriptive text.\n"
        "- For charts: State the specific values, trends, peaks, and valleys "
        "(e.g., \"Revenue peaked in Q3 2023 at $4.5M, a 15% increase from Q2\").\n"
        "- For flowcharts/diagrams: Describe the step-by-step flow, connections, "
        "and logic (e.g., \"The API Gateway routes traffic to the Auth Service, "
        "which then queries the Redis Cache\").\n"
        "- For tables: Summarize the key data points or anomalies; if the table "
        "is small, represent it entirely in Markdown format.]\n\n"
        "### 4. Semantic Keywords\n"
        "[Provide a comma-separated list of 5-10 highly specific keywords, "
        "jargon, or entities found in the image to aid in vector similarity matching.]"
    )

    MAX_RETRIES = 5          # Number of retry attempts on errors
    INITIAL_BACKOFF = 5.0    # Initial wait (seconds) before first retry
    DELAY_BETWEEN = 1.5      # Delay (seconds) between consecutive API calls (40 RPM = 1.5s/req)

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: NVIDIA NIM model ID for a vision-language model.
        """
        model_name = model_name or cfg.vlm_model_name
        if not cfg.nvidia_vlm_api_key:
            raise ValueError(
                "NVIDIA_VLM_API_KEY (or NVIDIA_API_KEY) not found in environment. "
                "Set it in .env"
            )

        self.client = OpenAI(
            base_url=cfg.nvidia_base_url,
            api_key=cfg.nvidia_vlm_api_key,
            timeout=cfg.vlm_timeout,
        )
        self.model_name = model_name
        logger.info(f"ImageSummarizer initialized with model: {model_name} (NVIDIA NIM)")

    def _image_to_base64_url(self, filepath: str, max_size: int = 1024) -> str:
        """Convert an image file to a base64 data URL, resizing if too large."""
        img = Image.open(filepath)

        # Resize large images to speed up API calls
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            logger.info(f"    Resized image to {img.size[0]}x{img.size[1]}")

        # Convert to JPEG for smaller payload (unless PNG with transparency)
        buffer = BytesIO()
        if img.mode == "RGBA":
            img.save(buffer, format="PNG")
            mime_type = "image/png"
        else:
            img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=85)
            mime_type = "image/jpeg"

        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{image_data}"

    def summarize(self, image: ExtractedImage) -> str:
        """Send an image to the VLM and return the summary string."""
        logger.info(f"  Summarizing {image.filename} ...")

        image_url = self._image_to_base64_url(image.filepath)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                ],
            }
        ]

        # Retry with exponential backoff for transient errors
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.2,
                    timeout=120.0,
                )
                summary = response.choices[0].message.content.strip()
                logger.info(f"    ✓ Summary length: {len(summary)} chars")
                return summary
            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    "429" in error_str
                    or "503" in error_str
                    or "502" in error_str
                    or "rate" in error_str.lower()
                    or "overloaded" in error_str.lower()
                )

                if is_retryable and attempt < self.MAX_RETRIES:
                    wait_time = self.INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(
                        f"    ⏳ Error (attempt {attempt + 1}/{self.MAX_RETRIES}): "
                        f"{type(e).__name__}. Waiting {wait_time:.0f}s ..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"    ✗ Failed to summarize {image.filename}: {e}")
                    return f"[Error: could not summarize image — {type(e).__name__}]"

    def summarize_all(
        self,
        images: list[ExtractedImage],
        source_doc: str = "unknown",
        cache_path: str = "./mock_s3_storage/summaries_cache.json",
    ) -> list[ExtractedImage]:
        """
        Summarize every image, with disk caching.
        Already-summarized images are loaded from cache instantly.
        New summaries are saved to cache after each image.

        Cache keys include the source document name to prevent
        collisions between same-numbered pages from different PDFs.
        """
        # Load existing cache
        cache = {}
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.info(f"  Loaded {len(cache)} cached summaries from {cache_path}")
            except Exception:
                cache = {}

        total = len(images)
        cached_count = 0
        for idx, img in enumerate(images):
            # Cache key = "doc4.pdf::page4_rendered.png" (unique per document)
            cache_key = f"{source_doc}::{img.filename}"

            # Check cache first
            if cache_key in cache:
                img.summary = cache[cache_key]
                cached_count += 1
                logger.info(f"  [{idx + 1}/{total}] {img.filename} — cached ✓")
                continue

            logger.info(f"  [{idx + 1}/{total}] Processing {img.filename}")
            img.summary = self.summarize(img)

            # Save to cache immediately (so progress is never lost)
            cache[cache_key] = img.summary
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

            # Rate-limit delay
            if idx < total - 1:
                time.sleep(self.DELAY_BETWEEN)

        logger.info(f"  Summary complete: {cached_count} cached, {total - cached_count} new")
        return images


# ═══════════════════════════════════════════════════════════════════════════════
#  3. DOCUMENT REASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentReassembler:
    """
    Reconstructs the full document text, injecting reference tags
    at the positions where images and tables were extracted.

    Tag formats:
      [IMAGE_REFERENCE | URL: /mock_s3_storage/{filename} | SUMMARY: {summary}]
      [TABLE_REFERENCE | PAGE: {n} | ROWS: {r} | COLS: {c} | CONTENT: {markdown}]
    """

    @staticmethod
    def reassemble(extraction: ExtractionResult) -> str:
        """Return the full document text with image and table references injected."""
        full_text_parts = []

        # Group images and tables by page
        images_by_page: dict[int, list] = {}
        for img in extraction.images:
            images_by_page.setdefault(img.page_number, []).append(img)

        tables_by_page: dict[int, list] = {}
        for tbl in extraction.tables:
            tables_by_page.setdefault(tbl.page_number, []).append(tbl)

        # Sort pages
        for page_num in sorted(extraction.page_texts.keys()):
            page_text = extraction.page_texts[page_num].strip()

            # Append the page text
            if page_text:
                full_text_parts.append(page_text)

            # Append table references for this page
            if page_num in tables_by_page:
                for tbl in sorted(tables_by_page[page_num], key=lambda t: t.position_index):
                    clean_md = re.sub(r"\s+", " ", tbl.markdown).strip()
                    tag = (
                        f"[TABLE_REFERENCE | PAGE: {page_num} "
                        f"| ROWS: {tbl.row_count} | COLS: {tbl.col_count} "
                        f"| CONTENT: {clean_md}]"
                    )
                    full_text_parts.append(tag)

            # Append image references for this page (after the page text)
            if page_num in images_by_page:
                # Sort by position index to maintain original order
                page_images = sorted(images_by_page[page_num], key=lambda x: x.position_index)
                for img in page_images:
                    # Clean summary: collapse internal newlines so the tag stays on one line
                    clean_summary = re.sub(r"\s+", " ", img.summary).strip()
                    tag = (
                        f"[IMAGE_REFERENCE | URL: {img.filepath} "
                        f"| SUMMARY: {clean_summary}]"
                    )
                    full_text_parts.append(tag)

        return "\n\n".join(full_text_parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SMART CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

class SmartChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter.

    Key guarantee: Reference tags are NEVER split in half.
    Protected tags:
      - [IMAGE_REFERENCE ...]
      - [TABLE_REFERENCE ...]
      - [SPEAKER_NOTES: ...]
    """

    # Regex matching ALL injected reference tags (never split these)
    TAG_PATTERN = re.compile(
        r"\[IMAGE_REFERENCE\s*\|[^\]]+\]"
        r"|\[TABLE_REFERENCE\s*\|[^\]]+\]"
        r"|\[SPEAKER_NOTES:[^\]]+\]"
    )

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # The recursive splitter uses these separators in order.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",    # double newline (paragraph break)
                "\n",      # single newline
                ". ",      # sentence boundary
                " ",       # word boundary
                "",        # character-level fallback
            ],
            keep_separator=True,
            is_separator_regex=False,
        )

    def chunk(self, text: str) -> list[str]:
        """
        Split text into chunks, guaranteeing image tags stay intact.

        Strategy:
          1. Split the text into segments around IMAGE_REFERENCE tags.
          2. Each tag becomes its own atomic segment.
          3. We then feed segments to the recursive splitter, but any segment
             that is a complete image tag is kept whole (even if > chunk_size).
        """
        # Split into interleaved [text, tag, text, tag, ...] segments
        segments = self.TAG_PATTERN.split(text)
        tags = self.TAG_PATTERN.findall(text)

        # Interleave segments and tags
        parts = []
        for i, segment in enumerate(segments):
            if segment.strip():
                parts.append(("text", segment.strip()))
            if i < len(tags):
                parts.append(("tag", tags[i]))

        # Now chunk each text segment and keep tags whole
        final_chunks = []
        text_buffer = ""

        for part_type, content in parts:
            if part_type == "tag":
                # Flush any accumulated text first
                if text_buffer.strip():
                    final_chunks.extend(self.splitter.split_text(text_buffer.strip()))
                    text_buffer = ""
                # Add tag as its own chunk (never split)
                final_chunks.append(content)
            else:
                text_buffer += "\n\n" + content if text_buffer else content

        # Flush remaining text
        if text_buffer.strip():
            final_chunks.extend(self.splitter.split_text(text_buffer.strip()))

        logger.info(f"Chunking complete — {len(final_chunks)} chunks produced")
        return final_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  5. VECTOR STORE MANAGER (PGVector — Persistent PostgreSQL Storage)
# ═══════════════════════════════════════════════════════════════════════════════


class NvidiaEmbeddings(Embeddings):
    """
    Custom embeddings class for NVIDIA NIM asymmetric models.
    Passes input_type='passage' for documents, 'query' for queries.
    Uses the raw OpenAI client — no Pydantic conflicts.

    Supports Matryoshka truncation: the model produces 2048-dim embeddings
    but we truncate to `truncate_dim` (default 1024) for HNSW compatibility
    and faster queries with negligible quality loss (~1.7%).
    """

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None,
                 batch_size: int = 50, truncate_dim: int = None):
        model = model or cfg.nvidia_embed_model
        base_url = base_url or cfg.nvidia_base_url
        api_key = api_key or cfg.nvidia_embed_api_key
        truncate_dim = truncate_dim if truncate_dim is not None else cfg.embed_truncate_dim
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.model = model
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim  # Matryoshka truncation dimension

    def _truncate(self, embedding: list[float]) -> list[float]:
        """Truncate embedding to target dimension (Matryoshka)."""
        if self.truncate_dim and len(embedding) > self.truncate_dim:
            return embedding[:self.truncate_dim]
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (passages)."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                extra_body={"input_type": "passage"},
            )
            all_embeddings.extend([
                self._truncate(item.embedding) for item in response.data
            ])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            extra_body={"input_type": "query"},
        )
        return self._truncate(response.data[0].embedding)


class VectorStoreManager:
    """
    Stores embeddings in a PostgreSQL database with the pgvector extension.
    Data persists across runs — no need to re-ingest every time.

    Uses NVIDIA NIM embedding model (free with your NVIDIA API key).
    """

    def __init__(
        self,
        collection_name: str = None,
        connection_string: str = None,
        embedding_model=None,
    ):
        """
        Args:
            collection_name:   Name for the pgvector collection.
            connection_string: PostgreSQL connection string.
            embedding_model:   Any LangChain Embeddings instance.
                               Defaults to NVIDIA NIM llama-nemotron-embed-1b-v2.
        """
        if embedding_model is None:
            self.embedding = NvidiaEmbeddings()
        else:
            self.embedding = embedding_model

        self.collection_name = collection_name or cfg.collection_name
        self.connection_string = connection_string or cfg.pg_connection_string

    def delete_document(self, source_doc: str):
        """Delete all chunks belonging to a specific document (for re-ingestion)."""
        from sqlalchemy import create_engine, text
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source_doc' = :doc"),
                {"doc": source_doc},
            )
            conn.commit()
            deleted = result.rowcount
        engine.dispose()
        if deleted > 0:
            logger.info(f"Deleted {deleted} old chunks for '{source_doc}'")
        return deleted

    def ingest(self, chunks: list[str], source_doc: str = "unknown",
               doc_hash: str = "", page_count: int = 0,
               source_format: str = "pdf") -> PGVector:
        """
        Insert chunks into the PGVector collection (incremental — keeps existing data).

        Args:
            chunks:        List of text chunks to embed and store.
            source_doc:    Filename of the source document (for tracking & targeted deletion).
            doc_hash:      SHA-256 hash of the source file (for change detection).
            page_count:    Number of pages/sections in the source document.
            source_format: File format ("pdf", "csv", "docx", "pptx").
        """
        logger.info(f"Connecting to PGVector (collection: {self.collection_name})")

        # Delete old chunks for this document (if re-ingesting an updated file)
        self.delete_document(source_doc)

        # Create metadata for each chunk — tracks source document and format
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "chunk_index": i,
                "has_image_ref": "[IMAGE_REFERENCE" in chunk,
                "has_table_ref": "[TABLE_REFERENCE" in chunk,
                "has_speaker_notes": "[SPEAKER_NOTES:" in chunk,
                "char_count": len(chunk),
                "source_doc": source_doc,
                "source_format": source_format,
                "doc_hash": doc_hash,
                "page_count": page_count,
            })

        vectorstore = PGVector.from_texts(
            texts=chunks,
            embedding=self.embedding,
            metadatas=metadatas,
            collection_name=self.collection_name,
            connection=self.connection_string,
            pre_delete_collection=False,  # INCREMENTAL: keep existing data
        )

        logger.info(f"Ingested {len(chunks)} chunks for '{source_doc}' into PGVector")
        return vectorstore

    def connect(self) -> PGVector:
        """Connect to an existing PGVector collection (for querying)."""
        logger.info(f"Connecting to existing PGVector collection: {self.collection_name}")
        return PGVector(
            embeddings=self.embedding,
            collection_name=self.collection_name,
            connection=self.connection_string,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    file_path: str,
    output_dir: str = None,
    vlm_model_name: str = None,
    embedding_model=None,
    connection_string: str = None,
    collection_name: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> tuple[PGVector, list[str]]:
    """
    End-to-end pipeline: Route → Extract → Summarize → Reassemble → Chunk → Store.

    Supports: .pdf, .csv, .pptx, .docx

    Supports incremental ingestion — each document's chunks are tracked by source
    document name. Re-ingesting a file automatically replaces its old chunks.

    Args:
        file_path:         Path to the input file (any supported format).
        output_dir:        Directory for extracted images.
        vlm_model_name:    NVIDIA NIM model ID for VLM (swappable).
        embedding_model:   Optional custom embedding model.
        connection_string: PostgreSQL connection string.
        collection_name:   Name for the pgvector collection.
        chunk_size:        Chunk size for the text splitter.
        chunk_overlap:     Overlap for the text splitter.

    Returns:
        (vectorstore, chunks) — the PGVector store and the raw chunk list.
    """
    # Apply config defaults
    output_dir = output_dir or cfg.output_dir
    vlm_model_name = vlm_model_name or cfg.vlm_model_name
    connection_string = connection_string or cfg.pg_connection_string
    collection_name = collection_name or cfg.collection_name
    chunk_size = chunk_size or cfg.chunk_size
    chunk_overlap = chunk_overlap or cfg.chunk_overlap

    # ── Compute source document identity ──────────────────────────────────
    source_doc = Path(file_path).name
    with open(file_path, "rb") as f:
        doc_hash = hashlib.sha256(f.read()).hexdigest()

    logger.info("=" * 60)
    logger.info(f"MULTIMODAL RAG INGESTION PIPELINE — START")
    logger.info(f"  Source: {source_doc}")
    logger.info(f"  Hash:   {doc_hash[:16]}...")
    logger.info("=" * 60)

    # ── Step 1: Extract (format-agnostic via FileRouter) ──────────────────
    logger.info("\n📄 STEP 1: Extracting content ...")
    router = FileRouter(output_dir=output_dir)
    extraction = router.extract(file_path, source_doc=source_doc)
    page_count = len(extraction.page_texts)

    # ── Step 2: Summarize images ──────────────────────────────────────────
    logger.info("\n🔍 STEP 2: Summarizing images via VLM ...")
    if extraction.images:
        summarizer = ImageSummarizer(model_name=vlm_model_name)
        summarizer.summarize_all(extraction.images, source_doc=source_doc)
    else:
        logger.info("  No images found — skipping summarization.")

    # ── Step 3: Reassemble document ───────────────────────────────────────
    logger.info("\n📝 STEP 3: Reassembling document with reference tags ...")
    reassembled_text = DocumentReassembler.reassemble(extraction)
    logger.info(f"  Reassembled text length: {len(reassembled_text)} chars")

    # ── Step 4: Chunk ─────────────────────────────────────────────────────
    logger.info("\n✂️  STEP 4: Chunking text (preserving reference tags) ...")
    chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(reassembled_text)

    # ── Step 5: Vector Store (PGVector — Incremental) ─────────────────────
    logger.info("\n🗄️  STEP 5: Ingesting chunks into PGVector ...")
    store_manager = VectorStoreManager(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_model=embedding_model,
    )
    vectorstore = store_manager.ingest(
        chunks,
        source_doc=source_doc,
        doc_hash=doc_hash,
        page_count=page_count,
        source_format=extraction.source_format,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"PIPELINE COMPLETE ✅ — {source_doc}: {len(chunks)} chunks ingested")
    logger.info("=" * 60)

    return vectorstore, chunks
