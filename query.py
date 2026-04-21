"""
Interactive Query Tool for the Multimodal RAG Pipeline
=======================================================
Connects to the EXISTING PGVector database — no re-ingestion needed.

Features:
  - HYBRID SEARCH: Combines semantic (vector) + keyword (BM25) retrieval
  - Reciprocal Rank Fusion (RRF) to merge results from both search methods
  - Cross-encoder reranking for better relevance
  - LLM answer synthesis via NVIDIA NIM (generates a coherent answer from chunks)
  - Metadata-aware retrieval (separates text vs image results)

Usage:
    python query.py
"""

import os
import re
import hashlib
import logging
import textwrap
from openai import OpenAI
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from pipeline import VectorStoreManager
from config import cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  HYBRID SEARCHER — Combines Semantic (Vector) + Keyword (BM25) Search
# ═══════════════════════════════════════════════════════════════════════════════

class HybridSearcher:
    """
    Combines two retrieval strategies:
      1. Semantic Search (Vector/Embedding) — finds meaning-similar chunks
      2. Keyword Search (BM25) — finds exact term matches

    Results are merged using Reciprocal Rank Fusion (RRF), which assigns
    scores based on rank position rather than raw scores, making it easy
    to combine results from different scoring systems.

    Why hybrid?
      - Semantic search: "provisioning timelines" → matches "SLA is 4 hours"
      - Keyword search:  "ERR-4102" → matches exact error code (semantic may miss)
      - Together: catches both meaning AND exact matches
    """

    RRF_K = 60  # RRF constant — controls how much rank position matters

    def __init__(self, vectorstore, all_docs: list, alpha: float = 0.5):
        """
        Args:
            vectorstore: PGVector instance for semantic search.
            all_docs:    All Document objects from the vectorstore (for BM25 index).
            alpha:       Balance between semantic (1.0) and keyword (0.0) search.
                         0.5 = equal weight (default), 0.7 = favor semantic, 0.3 = favor keyword.
        """
        self.vectorstore = vectorstore
        self.all_docs = all_docs
        self.alpha = alpha

        # Build BM25 index from all document chunks
        print("  Building BM25 keyword index ...")
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  BM25 index built: {len(all_docs)} documents indexed")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization for BM25."""
        # Remove special characters, lowercase, split on whitespace
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    def search(self, query: str, k: int = 20) -> list:
        """
        Perform hybrid search and return merged results.

        Args:
            query: User's search query.
            k:     Number of results to return.

        Returns:
            List of (Document, rrf_score) tuples, sorted best-first.
        """
        # ── Semantic search (vector similarity) ───────────────────────────
        print("    📐 Semantic search (vector similarity) ...")
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # ── Keyword search (BM25) ────────────────────────────────────────
        print("    🔤 Keyword search (BM25) ...")
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k BM25 results by score
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:k]
        keyword_results = [
            (self.all_docs[i], float(bm25_scores[i]))
            for i in top_bm25_indices
            if bm25_scores[i] > 0  # Only include docs with at least one keyword match
        ]

        # ── Reciprocal Rank Fusion (RRF) ─────────────────────────────────
        print("    🔀 Merging with Reciprocal Rank Fusion (RRF) ...")

        def _content_hash(doc):
            return hashlib.md5(doc.page_content.encode()).hexdigest()

        rrf_scores = {}  # content_hash → fused score
        all_docs_map = {}  # content_hash → Document

        # Score semantic results by rank
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = _content_hash(doc)
            rrf_score = self.alpha * (1.0 / (self.RRF_K + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            all_docs_map[doc_id] = doc

        # Score keyword results by rank (O(1) hash lookup instead of O(n²))
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = _content_hash(doc)
            rrf_score = (1 - self.alpha) * (1.0 / (self.RRF_K + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            if doc_id not in all_docs_map:
                all_docs_map[doc_id] = doc

        # Sort by fused RRF score
        fused_results = [
            (all_docs_map[doc_id], score)
            for doc_id, score in rrf_scores.items()
        ]
        fused_results.sort(key=lambda x: x[1], reverse=True)

        # Log stats
        sem_hashes = {_content_hash(d) for d, _ in semantic_results}
        kw_hashes = {_content_hash(d) for d, _ in keyword_results}
        overlap = len(sem_hashes & kw_hashes)
        sem_only = len(sem_hashes - kw_hashes)
        kw_only = len(kw_hashes - sem_hashes)
        print(f"    📊 Results: {sem_only} semantic-only, {kw_only} keyword-only, {overlap} overlap")

        return fused_results[:k]



# ═══════════════════════════════════════════════════════════════════════════════
#  RERANKER — Cross-Encoder for better relevance scoring
# ═══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Uses a cross-encoder model to rerank retrieved chunks.
    Cross-encoders process (query, document) pairs jointly, giving
    far more accurate relevance scores than bi-encoder similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"  Loading reranker: {model_name} ...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k: int = 5) -> list:
        """
        Rerank documents by relevance to query.

        Args:
            query:  The user's search query.
            docs:   List of (Document, score) tuples from hybrid search.
            top_k:  Number of top results to return after reranking.

        Returns:
            List of (Document, rerank_score) tuples, sorted best-first.
        """
        if not docs:
            return []

        # Score each (query, chunk) pair with the cross-encoder
        pairs = [(query, doc.page_content) for doc, _ in docs]
        scores = self.model.predict(pairs)

        # Attach rerank scores and sort (higher = more relevant)
        reranked = [(doc, float(score)) for (doc, _), score in zip(docs, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
#  ANSWER SYNTHESIZER — NVIDIA NIM LLM generates a coherent response
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerSynthesizer:
    """
    Takes retrieved chunks + user question and generates a coherent
    answer using an LLM via NVIDIA NIM API.
    """

    SYSTEM_PROMPT = (
        "You are a helpful research assistant. Answer the user's question "
        "based ONLY on the provided context. If the context doesn't contain "
        "enough information, say so honestly. Cite specific details from the "
        "context. Be concise but thorough."
    )

    def __init__(self, model_name: str = None):
        model_name = model_name or cfg.llm_model_name
        self.client = OpenAI(
            base_url=cfg.nvidia_base_url,
            api_key=cfg.nvidia_vlm_api_key,
            timeout=cfg.vlm_timeout,
        )
        self.model_name = model_name

    def synthesize(self, query: str, docs: list) -> str:
        """Generate an answer from the top retrieved chunks."""
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(docs):
            source_type = "IMAGE" if "[IMAGE_REFERENCE" in doc.page_content else "TEXT"
            context_parts.append(
                f"[Source {i+1} ({source_type})]:\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"---\n\n"
                    f"Question: {query}\n\n"
                    f"Answer based on the context above:"
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating answer: {type(e).__name__}: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Interactive Query Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Initialize components ─────────────────────────────────────────────
    print("🔗 Initializing RAG query engine ...")
    print("  Connecting to PGVector database ...")
    store = VectorStoreManager()
    vectorstore = store.connect()

    # Load all documents for BM25 indexing (direct SQL, no embedding call needed)
    print("  Loading all chunks for hybrid search ...")
    from langchain_core.documents import Document
    from sqlalchemy import create_engine, text
    
    pg_conn = os.environ.get("PG_CONNECTION_STRING", "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres")
    engine = create_engine(pg_conn)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT document, cmetadata FROM langchain_pg_embedding "
            "WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = 'multimodal_rag')"
        )).fetchall()
    all_docs = [Document(page_content=row[0], metadata=row[1] if row[1] else {}) for row in rows]
    print(f"  Loaded {len(all_docs)} chunks from database")

    # Pre-filter image chunks + build BM25 index for instant image retrieval
    image_docs = [doc for doc in all_docs if "[IMAGE_REFERENCE" in doc.page_content]
    if image_docs:
        image_tokenized = [HybridSearcher._tokenize(doc.page_content) for doc in image_docs]
        image_bm25 = BM25Okapi(image_tokenized)
    else:
        image_bm25 = None
    print(f"  Found {len(image_docs)} image chunks (BM25-indexed for instant retrieval)")

    print("  Loading cross-encoder reranker ...")
    reranker = Reranker()

    # Initialize hybrid searcher (semantic + keyword)
    hybrid_searcher = HybridSearcher(vectorstore, all_docs, alpha=0.5)

    print("  Connecting to NVIDIA NIM LLM for answer synthesis ...")
    synthesizer = AnswerSynthesizer()

    print("\n✅ Ready!")
    print("=" * 60)
    print("🔍 ASK ANYTHING ABOUT YOUR DOCUMENT")
    print("   Search mode: HYBRID (Semantic + Keyword)")
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # ── Interactive query loop ────────────────────────────────────────────
    while True:
        try:
            query = input("\n📝 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        # Step 1: Hybrid search (semantic + keyword → RRF fusion) → top 20
        print("\n  🔎 Hybrid search ...")
        hybrid_results = hybrid_searcher.search(query, k=20)

        if not hybrid_results:
            print("  No results found.")
            continue

        # Step 2: Rerank with cross-encoder → top 5
        print("  🏆 Reranking with cross-encoder ...")
        reranked = reranker.rerank(query, hybrid_results, top_k=5)

        # Step 3: Synthesize answer with LLM
        print("  🤖 Generating answer ...\n")
        answer = synthesizer.synthesize(query, reranked)

        # ── Display the answer ────────────────────────────────────────────
        print("═" * 60)
        print("💡 ANSWER")
        print("─" * 60)
        # Word-wrap the answer for clean display
        for paragraph in answer.split("\n"):
            if paragraph.strip():
                wrapped = textwrap.fill(paragraph.strip(), width=80, initial_indent="  ", subsequent_indent="  ")
                print(wrapped)
            else:
                print()
        print("═" * 60)

        # ── Show the supporting sources ───────────────────────────────────
        print(f"\n── Supporting Sources ({len(reranked)} chunks) ──\n")
        for i, (doc, score) in enumerate(reranked):
            content = doc.page_content
            chunk_idx = doc.metadata.get("chunk_index", "?")
            source_doc = doc.metadata.get("source_doc", "unknown")
            source_fmt = doc.metadata.get("source_format", "pdf")
            is_image = "[IMAGE_REFERENCE" in content
            is_table = "[TABLE_REFERENCE" in content

            # Format-aware icon
            fmt_icons = {"pdf": "📄", "csv": "📊", "pptx": "📽️", "docx": "📝", "excel": "📈"}
            icon = fmt_icons.get(source_fmt, "📄")

            if is_image:
                # Extract filename and summary from the image tag
                url_match = re.search(r"URL:\s*(\S+)", content)
                summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\]|$)", content)
                filename = url_match.group(1) if url_match else "unknown"
                summary = summary_match.group(1)[:150] + "..." if summary_match else "No summary"
                print(f"  🖼️  Source {i+1}  (score: {score:.2f})  [{icon} {source_doc} → Chunk #{chunk_idx}]")
                print(f"      Image: {filename}")
                print(f"      Summary: {summary}")
            elif is_table:
                # Extract table info from the table tag
                rows_match = re.search(r"ROWS:\s*(\d+)", content)
                cols_match = re.search(r"COLS:\s*(\d+)", content)
                rows = rows_match.group(1) if rows_match else "??"
                cols = cols_match.group(1) if cols_match else "??"
                print(f"  📊 Source {i+1}  (score: {score:.2f})  [{icon} {source_doc} → Chunk #{chunk_idx}]")
                print(f"      Table: {rows} rows × {cols} columns")
            else:
                # Text source — show clean preview
                preview = " ".join(content.split())[:200]
                if len(content) > 200:
                    preview += " ..."
                print(f"  {icon} Source {i+1}  (score: {score:.2f})  [{icon} {source_doc} → Chunk #{chunk_idx}]")
                print(f"      {preview}")

            if i < len(reranked) - 1:
                print()

        # ── Show related images (BM25 keyword match — instant) ───────────
        # Uses a dedicated BM25 index built ONLY on image chunks.
        # Runs in < 1ms (no cross-encoder, no API call).
        # Only shows images from source documents that appeared in the main results.
        print(f"\n── 📎 Related Images ──")

        # Collect source docs from main results to scope image suggestions
        result_source_docs = {
            doc.metadata.get("source_doc", "")
            for doc, _ in reranked
            if doc.metadata.get("source_doc")
        }

        if image_bm25 and image_docs:
            img_query_tokens = HybridSearcher._tokenize(query)
            img_scores = image_bm25.get_scores(img_query_tokens)
            top_img_indices = sorted(
                range(len(img_scores)),
                key=lambda i: img_scores[i],
                reverse=True
            )[:10]  # Check more candidates since we'll filter by source

            shown = 0
            print()
            for idx in top_img_indices:
                if shown >= 3:
                    break
                if img_scores[idx] <= 1.0:  # Minimum relevance threshold
                    continue
                doc = image_docs[idx]
                img_source = doc.metadata.get("source_doc", "unknown")

                # Only show images from source documents in the main results
                if img_source not in result_source_docs:
                    continue

                url_match = re.search(r"URL:\s*(\S+)", doc.page_content)
                summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\]|$)", doc.page_content)
                chunk_idx = doc.metadata.get("chunk_index", "?")
                filename = url_match.group(1) if url_match else "unknown"
                summary = summary_match.group(1)[:150] + "..." if summary_match else "No summary"
                print(f"  🖼️  [{img_source} → Chunk #{chunk_idx}]")
                print(f"      Image: {filename}")
                print(f"      {summary}")
                shown += 1
                if shown < 3:
                    print()

            if shown == 0:
                print("  No related images found for this query.")
        else:
            print("  No image chunks found in database.")


if __name__ == "__main__":
    main()
