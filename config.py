"""
Centralized Configuration
=========================
Single source of truth for all environment variables, model configs,
connection strings, and pipeline parameters.

Usage:
    from config import cfg, SUPPORTED_EXTENSIONS
    print(cfg.pg_connection_string)
    print(cfg.nvidia_base_url)
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Fix SSL (corporate networks) ──
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".docx", ".pptx", ".xlsx", ".xls"}


@dataclass
class PipelineConfig:
    """All pipeline configuration in one place."""

    # ── NVIDIA NIM API Keys ──
    nvidia_vlm_api_key: str = field(default_factory=lambda: (
        os.environ.get("NVIDIA_VLM_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
    ))
    nvidia_embed_api_key: str = field(default_factory=lambda: (
        os.environ.get("NVIDIA_EMBED_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
    ))
    nvidia_base_url: str = field(default_factory=lambda: os.environ.get(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    ))

    # ── NVIDIA NIM Models ──
    vlm_model_name: str = field(default_factory=lambda: os.environ.get(
        "VLM_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    ))
    nvidia_embed_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embed_truncate_dim: int = 1024
    llm_model_name: str = field(default_factory=lambda: os.environ.get(
        "LLM_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    ))

    # ── PostgreSQL ──
    pg_connection_string: str = field(default_factory=lambda: os.environ.get(
        "PG_CONNECTION_STRING",
        "postgresql+psycopg://langchain:langchain@localhost:5432/langchain",
    ))
    collection_name: str = "multimodal_rag"

    # ── Pipeline Parameters ──
    output_dir: str = "./mock_s3_storage"
    chunk_size: int = 800
    chunk_overlap: int = 100
    render_dpi: int = 200
    csv_rows_per_page: int = 50

    # ── VLM Parameters ──
    vlm_max_retries: int = 5
    vlm_initial_backoff: float = 5.0
    vlm_delay_between: float = 1.5
    vlm_timeout: float = 120.0
    vlm_max_image_size: int = 1024

    # ── Search Parameters ──
    hybrid_alpha: float = 0.5
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


cfg = PipelineConfig()
