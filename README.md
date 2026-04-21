# 📄 Multimodal RAG Ingestion & Retrieval Pipeline

A modular, format-agnostic Python pipeline that extracts text, images, tables, and speaker notes from **PDFs, CSVs, Excel, DOCX, and PPTX** files, summarizes visual content using a Vision Language Model (NVIDIA NIM), and stores everything in a persistent PostgreSQL vector database (PGVector) for accurate Retrieval-Augmented Generation (RAG).

> **Evolution**: This pipeline was originally PDF-only. It has been refactored into a multimodal architecture using the **Strategy Pattern** — adding new file formats requires only writing a single extractor class.

---

## 🏗️ Architecture

### Ingestion Pipeline (`main.py` → `pipeline.py`)

```
                                          extractors/ package
                                    ┌─────────────────────────────────┐
 ┌───────────────┐    ┌───────────┐ │ ┌─────────────────────────────┐ │    ┌───────────────────────────────────┐
 │  File Input   ├───►│FileRouter │─┤ │ .pdf  → PDFExtractor        │ │───►│  ImageSummarizer                  │
 │               │    │           │ │ │         (PyMuPDF)            │ │    │  (NVIDIA NIM)                     │
 │ .pdf          │    │ Dispatches│ │ │  • Text per page             │ │    │                                   │
 │ .csv          │    │ by ext    │ │ │  • Embedded images           │ │    │  llama-3.1-nemotron-nano-vl-8b-v1 │
 │ .xlsx / .xls  │    │           │ │ │  • Rendered pages (charts)   │ │    │  (VLM — 8B params)                │
 │ .docx         │    │           │ │ │  • Structured tables ★       │ │    │  + Summary caching                │
 │ .pptx         │    │           │ │ ├─────────────────────────────┤ │    │                                   │
 │               │    │           │ │ │ .csv  → CSVExtractor        │ │    │  Only runs if images exist        │
 └───────────────┘    └───────────┘ │ │         (pandas)             │ │    └─────────────────┬─────────────────┘
                                    │ │  • Schema summary (pg 0)    │ │                      │
                                    │ │  • Row-windowed md tables   │ │                      ▼
                                    │ ├─────────────────────────────┤ │    ┌──────────────────────────────────────┐
                                    │ │ .xlsx → ExcelExtractor      │ │    │  DocumentReassembler                 │
                                    │ │         (openpyxl/pandas)   │ │    │                                      │
                                    │ │  • Multi-sheet support      │ │    │  Injects:                            │
                                    │ │  • Row-windowed md tables   │ │    │   [IMAGE_REFERENCE | URL | SUMMARY]  │
                                    │ │  • Embedded images          │ │    │   [TABLE_REFERENCE | PAGE | CONTENT] │
                                    │ ├─────────────────────────────┤ │    │   [SPEAKER_NOTES: ...]          ★    │
                                    │ │ .docx → DOCXExtractor       │ │    └─────────────────┬────────────────────┘
                                    │ │         (python-docx)        │ │                      │
                                    │ │  • Heading-aware sections   │ │                      ▼
                                    │ │  • Tables → markdown        │ │    ┌──────────────────────────────────────┐
                                    │ │  • Embedded images          │ │    │  SmartChunker (800/100)               │
                                    │ ├─────────────────────────────┤ │    │                                      │
                                    │ │ .pptx → PPTXExtractor       │ │    │  Never splits:                       │
                                    │ │         (python-pptx)        │ │    │   • IMAGE_REFERENCE tags              │
                                    │ │  • Slide-per-page           │ │    │   • TABLE_REFERENCE tags          ★   │
                                    │ │  • Tables, images           │ │    │   • SPEAKER_NOTES tags            ★   │
                                    │ │  • Speaker notes        ★  │ │    └─────────────────┬────────────────────┘
                                    │ └─────────────────────────────┘ │                      │
                                    └─────────────────────────────────┘                      ▼
                                                                          ┌──────────────────────────────────────┐
                                             ExtractionResult             │  VectorStoreManager (PGVector)       │
                                              (unified schema)            │                                      │
                                        page_texts + images + tables ★    │  Metadata per chunk:                  │
                                                                          │   • source_doc, source_format     ★  │
                                                                          │   • has_image_ref, has_table_ref  ★  │
                                                                          │   • doc_hash, chunk_index            │
                                                                          │                                      │
                                                                          │  Embeddings: llama-nemotron-embed-    │
                                                                          │  1b-v2 (NVIDIA NIM, 1024-dim)        │
                                                                          └──────────────────────────────────────┘
                                                                                     pipeline.py

★ = New in multimodal refactor
```

### Query Pipeline (`query.py`)

```
┌──────────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────────────────────────────────┐
│  Answer Synthesis                │◄─│  Cross-Encoder Reranker  │◄─│  HYBRID SEARCH                                      │
│  (NVIDIA NIM)                    │  │                          │  │                                                      │
│                                  │  │  ms-marco-MiniLM-L-6-v2  │  │  ┌─ Semantic (PGVector cosine similarity)            │
│ llama-3.1-nemotron-nano          │  │  (22M params, local CPU) │  │  └─ Keyword  (BM25) → Reciprocal Rank Fusion        │
│ -vl-8b-v1                        │  │  Disk: ~90 MB            │  │                                                      │
│ (8B params)                      │  │  RAM:  ~90 MB            │  │  RRF uses content-based MD5 hashing                ★ │
│ Disk: ~16 GB | VRAM: ~16 GB     │  │                          │  │  (fixes the old id()-based matching bug)            ★ │
│                                  │  │  Top 20 → Top 5          │  │                                                      │
│ Format-aware source display    ★ │  │                          │  │  Embeddings: llama-nemotron-embed-1b-v2 (NVIDIA NIM) │
│ 📄PDF 📊CSV 📗XLSX 📽️PPTX 📝DOCX ★ │  │                          │  │  (1B params) Disk: ~2 GB | VRAM: ~2 GB FP16         │
└──────────────────────────────────┘  └──────────────────────────┘  └──────────────────────────────────────────────────────┘
         query.py                              query.py                              query.py + pipeline.py

★ = New in multimodal refactor
```

---

## 🤖 Models Used

| Component | Model | Provider | Details |
|-----------|-------|----------|---------|
| **VLM (Image Summarization)** | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA NIM | 8B params, document-focused VLM, fast inference |
| **Embeddings** | `nvidia/llama-nemotron-embed-1b-v2` | NVIDIA NIM | 1B param, asymmetric model (passage/query), 1024-dim Matryoshka |
| **Answer Synthesis** | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA NIM | Same VLM for generating coherent answers from retrieved chunks |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local (HuggingFace) | Runs locally, reranks top-20 → top-5 for better relevance |
| **Keyword Search** | BM25 (rank-bm25) | Local | BM25Okapi for keyword matching, merged with semantic via RRF |

> **Prototype → Production**: All NVIDIA NIM calls use the OpenAI-compatible API. For production, the same Nemotron models will be self-hosted on AWS — only `NVIDIA_BASE_URL` in `.env` needs to change. Zero code modifications required.

---

## 📂 Supported File Formats

| Format | Extractor | Library | What It Captures |
|--------|-----------|---------|-----------------|
| `.pdf` | `PDFExtractor` | PyMuPDF | Text per page, embedded images, vector-rendered charts, **structured tables** |
| `.csv` | `CSVExtractor` | pandas | Schema summary (page 0), row-windowed markdown tables (50 rows/page) |
| `.xlsx`/`.xls` | `ExcelExtractor` | openpyxl/pandas | Multi-sheet support, schema summary, row-windowed markdown tables, embedded images |
| `.docx` | `DOCXExtractor` | python-docx | Heading-aware sections, tables → markdown, embedded images |
| `.pptx` | `PPTXExtractor` | python-pptx | Slide-per-page text, tables, images, **speaker notes** |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker Desktop (for PostgreSQL + pgvector)
- NVIDIA NIM API keys (free from [build.nvidia.com](https://build.nvidia.com))

### 2. Setup

```bash
# Navigate to the project
cd IP_new

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# or: venv\Scripts\activate     # Windows (CMD)
# or: source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup (One-time)

```bash
# Start a PostgreSQL container with pgvector
docker run --name local-rag-db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg17

# Enable the vector extension
docker exec local-rag-db psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Configure API Keys

Create a `.env` file:

```env
# Get keys from each model's page on https://build.nvidia.com
NVIDIA_VLM_API_KEY=nvapi-xxxxx      # From llama-3.1-nemotron-nano-vl-8b-v1 model page
NVIDIA_EMBED_API_KEY=nvapi-xxxxx    # From llama-nemotron-embed-1b-v2 model page

# PostgreSQL connection
PG_CONNECTION_STRING=postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres
```

### 5. Run

```bash
# Make sure Docker container is running
docker start local-rag-db

# Ingest a single file
python main.py sample.pdf

# Ingest multiple formats at once
python main.py sample.pdf data.csv data.xlsx deck.pptx report.docx

# Query interactively (instant — uses existing embeddings)
python query.py
```

---

## 📁 Project Structure

```
IP_new/
├── config.py                    # Centralized configuration (PipelineConfig dataclass)
├── pipeline.py                  # Core pipeline (ImageSummarizer, Reassembler, Chunker, VectorStore, orchestrator)
├── query.py                     # Interactive query tool (hybrid search + reranker + LLM synthesis)
├── main.py                      # Entry point — multi-format ingestion with progress summary
├── requirements.txt             # Python dependencies
├── .env                         # Your API keys (not committed)
│
├── extractors/                  # Format-specific extractors (Strategy Pattern)
│   ├── __init__.py              # Package exports: FileRouter, SUPPORTED_EXTENSIONS
│   ├── base.py                  # BaseExtractor ABC + ExtractionResult + ExtractedTable
│   ├── pdf.py                   # PDFExtractor (PyMuPDF) — text, images, tables
│   ├── csv_ext.py               # CSVExtractor (pandas) — schema + row-windowed tables
│   ├── excel_ext.py             # ExcelExtractor (openpyxl/pandas) — multi-sheet
│   ├── docx_ext.py              # DOCXExtractor (python-docx) — heading-aware sections
│   ├── pptx_ext.py              # PPTXExtractor (python-pptx) — slides, speaker notes
│   └── router.py                # FileRouter — dispatches by file extension
│
├── mock_s3_storage/             # Auto-created — extracted images stored here
│   └── summaries_cache.json     # Cached VLM summaries (never re-summarize)
│
├── MULTIMODAL_IMPLEMENTATION_GUIDE.md  # Full technical reference (all code templates)
├── PARSING_COMPARATIVE_STUDY.md        # Parser comparison (Individual libs vs Unstructured vs Docling)
│
├── sample.pdf                   # Test files
├── doc2.pdf
├── doc3.pdf
├── doc4.pdf
└── doc5.pdf
```

---

## 🧩 Pipeline Modules

### `config.py` — Centralized Configuration

All environment variables, model names, API URLs, and pipeline parameters in one `PipelineConfig` dataclass:

```python
from config import cfg

cfg.nvidia_base_url      # "https://integrate.api.nvidia.com/v1"
cfg.vlm_model_name       # "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
cfg.nvidia_embed_model   # "nvidia/llama-nemotron-embed-1b-v2"
cfg.pg_connection_string # From .env
cfg.chunk_size           # 800
cfg.chunk_overlap        # 100
```

### `extractors/` — Format-Specific Extractors

All extractors inherit from `BaseExtractor` and return the same `ExtractionResult` schema:

```python
from extractors import FileRouter

router = FileRouter(output_dir="./mock_s3_storage")
result = router.extract("report.pdf")     # → PDFExtractor
result = router.extract("data.csv")       # → CSVExtractor
result = router.extract("data.xlsx")      # → ExcelExtractor
result = router.extract("deck.pptx")      # → PPTXExtractor
result = router.extract("paper.docx")     # → DOCXExtractor

# Unified output schema (same for all formats):
result.page_texts   # {page_num: text}
result.images       # [ExtractedImage(...)]
result.tables       # [ExtractedTable(markdown=..., rows=..., cols=...)]
result.source_format  # "pdf" | "csv" | "excel" | "docx" | "pptx"
```

#### PDFExtractor

| Method | What It Captures | How |
|--------|-----------------|-----|
| **Text** | Raw text per page | `page.get_text("text")` |
| **Embedded images** | Photos, logos, raster graphics | `page.get_images()` — extracts raw image data |
| **Rendered pages** | Charts, graphs (vector-drawn) | `page.get_pixmap()` — renders page at 200 DPI |
| **Tables** | Structured tabular data | `page.find_tables()` → pandas → markdown |

#### ExcelExtractor

| Page | Content |
|------|---------|
| Page 0 | Workbook overview — summary of total rows and column counts across all sheets |
| Per Sheet | Sheet schema (column stats), followed by row windows of 50 rows each (markdown tables) |
| Elements | Embedded images via `openpyxl` drawing package |

#### CSVExtractor

| Page | Content |
|------|---------|
| Page 0 | Schema summary — column names, data types, statistics, sample values |
| Pages 1..N | Row windows of 50 rows each, formatted as markdown tables |

#### DOCXExtractor

- **Heading-aware sectioning**: Heading 1/2 triggers a new "page" boundary
- **Tables**: Converted to markdown format preserving column alignment
- **Images**: Extracted from embedded media parts (`.png`, `.jpeg`, etc.)
- **Heading markers**: `#`, `##`, `###` preserved for context

#### PPTXExtractor

- **Slide-per-page**: Each slide becomes one entry in `page_texts`
- **Tables**: Extracted from table shapes → markdown
- **Images**: Extracted from picture shapes (including group shapes)
- **Speaker notes**: Captured with `[SPEAKER_NOTES: ...]` tag

---

### `pipeline.py` — Ingestion Components

#### ImageSummarizer (NVIDIA NIM VLM)

Sends each image to the VLM for structured summarization. Features:
- Image resizing (>1024px → thumbnailed), JPEG compression
- Exponential backoff retry (5 retries, 5s → 160s)
- Rate limiting (1.5s delay, 40 RPM compliance)
- Disk-based caching (`summaries_cache.json`)

#### DocumentReassembler

Reconstructs document text with three types of reference tags:

```
[IMAGE_REFERENCE | URL: /mock_s3_storage/page2_rendered.png | SUMMARY: The chart shows ...]
[TABLE_REFERENCE | PAGE: 3 | ROWS: 12 | COLS: 5 | CONTENT: | Col1 | Col2 | ...]
[SPEAKER_NOTES: Detailed explanation of the slide content ...]
```

#### SmartChunker (800/100)

Wraps LangChain's `RecursiveCharacterTextSplitter` with a **guarantee**: reference tags are **never split in half**.

Protected tags: `[IMAGE_REFERENCE]`, `[TABLE_REFERENCE]`, `[SPEAKER_NOTES]`

#### NvidiaEmbeddings + VectorStoreManager

- **Asymmetric embedding**: `input_type="passage"` for documents, `"query"` for queries
- **Matryoshka truncation**: 2048-dim → 1024-dim (negligible quality loss)
- **Incremental ingestion**: Re-ingesting a file deletes only that file's old chunks
- **Format-aware metadata**: `source_format`, `has_table_ref`, `has_speaker_notes`

---

## 🔍 Query Module (query.py)

### Stage 1: Hybrid Search (Semantic + Keyword → RRF)

| Strategy | Method | What It Catches |
|----------|--------|-----------------|
| **Semantic** | PGVector cosine similarity | Meaning-based matches |
| **Keyword** | BM25 (rank-bm25) | Exact term matches |
| **Fusion** | RRF with content-based MD5 hashing | Merges both → top 20 |

> **Bug fix**: The old pipeline used `id(doc)` (memory addresses) for RRF fusion, causing BM25 and semantic results to never properly merge. Now uses `hashlib.md5(content)` for stable, content-based matching with O(1) hash-map lookups.

### Stage 2: Cross-Encoder Reranking

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, 22M params). Top 20 → Top 5.

### Stage 3: LLM Answer Synthesis

Model: `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (NVIDIA NIM). Format-aware source display:

| Icon | Format | Example |
|------|--------|---------|
| 📄 | PDF | `📄 Source 1 (score: 0.87) [📄 report.pdf → Chunk #12]` |
| 📊 | CSV | `📊 Source 2 (score: 0.75) [📊 data.csv → Chunk #3]` |
| 📗 | XLSX | `📗 Source 3 (score: 0.73) [📗 sheet.xlsx → Chunk #5]` |
| 📽️ | PPTX | `📽️ Source 4 (score: 0.71) [📽️ deck.pptx → Chunk #5]` |
| 📝 | DOCX | `📝 Source 5 (score: 0.68) [📝 paper.docx → Chunk #8]` |

---

## ⚙️ Configuration

### Multi-Format Ingestion

```bash
# Single file
python main.py report.pdf

# Multiple formats
python main.py report.pdf financials.csv slides.pptx whitepaper.docx

# All PDFs in directory
python main.py *.pdf
```

### Swapping the VLM Model

```python
vectorstore, chunks = run_pipeline(
    file_path="report.pdf",
    vlm_model_name="qwen/qwen3.5-122b-a10b",
)
```

### Custom Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

custom_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore, chunks = run_pipeline("report.pdf", embedding_model=custom_emb)
```

### CSV Row Window Size

Set via `config.py` → `cfg.csv_rows_per_page` (default: 50 rows per page).

---

## 🔧 Troubleshooting

### SSL Certificate Errors (Corporate Networks)

The pipeline includes automatic SSL handling via `truststore` (loaded in `config.py`):

```bash
pip install truststore
```

### NVIDIA NIM 401 Unauthorized

- Ensure your API key is from the **correct model page** on [build.nvidia.com](https://build.nvidia.com)
- `NVIDIA_VLM_API_KEY` → from the VLM model page
- `NVIDIA_EMBED_API_KEY` → from the embedding model page

### NVIDIA NIM 429 Rate Limit

Automatic retry with exponential backoff (5 retries, 5s → 160s). If persistent:
- Wait a few minutes and try again
- Generate a new API key
- Increase `vlm_delay_between` in `config.py`

### Docker / PGVector Issues

```bash
docker ps                          # Check if running
docker start local-rag-db         # Start container
docker exec local-rag-db psql -U postgres -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` | PDF parsing — text, images, rendered pages, table extraction |
| `python-docx` | DOCX parsing — heading-aware sections, tables, images |
| `python-pptx` | PPTX parsing — slides, tables, images, speaker notes |
| `pandas` | CSV parsing + markdown table formatting |
| `openpyxl` | Excel extraction including embedded images |
| `tabulate` | `df.to_markdown()` dependency |
| `openai` | NVIDIA NIM API client (OpenAI-compatible protocol) |
| `langchain` | Core LangChain framework |
| `langchain-openai` | LangChain OpenAI integration |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter for chunking |
| `langchain-postgres` | PGVector integration for persistent vector storage |
| `psycopg[binary]` | PostgreSQL driver |
| `sentence-transformers` | Cross-encoder reranker (local, query.py) |
| `rank-bm25` | BM25Okapi keyword search for hybrid retrieval |
| `Pillow` | Image resizing and format conversion |
| `python-dotenv` | Environment variable loading from `.env` |
| `truststore` | OS-level SSL certificate handling |

---

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Image summarization** | ~2-4 sec/image (NVIDIA NIM), instant (cached) |
| **Embedding generation** | ~200ms per batch (NVIDIA NIM) |
| **File extraction** | < 5s per non-PDF file |
| **Semantic search latency** | < 20ms |
| **BM25 keyword search** | < 5ms |
| **Query response time** | ~5-8s (hybrid + rerank + LLM synthesis) |
| **NVIDIA NIM rate limit** | 40 RPM per API key |

---

## 🔄 What Changed from the PDF-Only Version

| Component | Before (PDF-only) | After (Multimodal) |
|-----------|-------------------|-------------------|
| **Input** | `.pdf` only | `.pdf`, `.csv`, `.xlsx`, `.docx`, `.pptx` |
| **Extraction** | `PDFExtractor` in pipeline.py | `extractors/` package with FileRouter |
| **Tables** | Flat text (lost column structure) | Structured markdown via `ExtractedTable` |
| **Config** | Scattered `os.environ` calls | Centralized `config.py` → `PipelineConfig` |
| **Reassembler** | `[IMAGE_REFERENCE]` only | + `[TABLE_REFERENCE]` + `[SPEAKER_NOTES]` |
| **Chunker** | Protects image tags | Protects image + table + speaker note tags |
| **RRF Fusion** | `id(doc)` (broken — memory addresses) | `hashlib.md5(content)` (correct — O(1)) |
| **Metadata** | `source_doc`, `has_image_ref` | + `source_format`, `has_table_ref`, `has_speaker_notes` |
| **Source Display** | 📄 only | 📄 PDF, 📊 CSV, 📗 XLSX, 📽️ PPTX, 📝 DOCX |

---

## 📄 License

This is a prototype/proof-of-concept for internal use.
