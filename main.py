"""
Main entry point for the Multimodal RAG Ingestion Pipeline.
============================================================
Usage:
    Single file:
        python main.py sample.pdf

    Multiple files (any supported format):
        python main.py doc1.pdf data.csv deck.pptx paper.docx

    All PDFs in current directory:
        python main.py *.pdf

    No arguments (defaults to sample.pdf):
        python main.py
"""

import sys
import time
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from pipeline import run_pipeline


def main():
    # ── Collect file paths from command-line arguments ────────────────────
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        file_paths = ["sample.pdf"]

    # ── Validate all files exist and are supported ────────────────────────
    valid_files = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            print(f"  ⚠️  Skipping '{fp}' — file not found")
        elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"  ⚠️  Skipping '{fp}' — unsupported format (use: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
        else:
            valid_files.append(fp)

    if not valid_files:
        print("❌ ERROR: No valid files found.")
        print(f"   Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        print("   Usage: python main.py doc1.pdf data.csv deck.pptx paper.docx")
        sys.exit(1)

    print(f"\n📚 Ingestion Queue: {len(valid_files)} file(s)")
    for i, fp in enumerate(valid_files, 1):
        size_mb = Path(fp).stat().st_size / (1024 * 1024)
        ext = Path(fp).suffix.upper()
        print(f"   {i}. {fp} ({ext}, {size_mb:.1f} MB)")

    # ── Ingest each file sequentially ─────────────────────────────────────
    total_chunks = 0
    results = []
    overall_start = time.time()

    for i, fp in enumerate(valid_files, 1):
        print(f"\n{'━' * 60}")
        print(f"📄 [{i}/{len(valid_files)}] Ingesting: {fp}")
        print(f"{'━' * 60}")

        start = time.time()
        vectorstore, chunks = run_pipeline(fp)
        elapsed = time.time() - start

        total_chunks += len(chunks)
        results.append({
            "file": fp,
            "chunks": len(chunks),
            "time": elapsed,
        })

        print(f"  ✅ {fp}: {len(chunks)} chunks in {elapsed:.1f}s")

    # ── Print summary ─────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n{'═' * 60}")
    print(f"📊  INGESTION COMPLETE — SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Files ingested    : {len(results)}")
    print(f"  Total chunks      : {total_chunks}")
    print(f"  Total time        : {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Avg time per file : {total_time/len(results):.1f}s")
    print()

    for r in results:
        ext = Path(r['file']).suffix.upper()
        print(f"  📄 {r['file']:40s} │ {ext:5s} │ {r['chunks']:>5} chunks │ {r['time']:>6.1f}s")

    print(f"\n  Stored in PGVector : ✅ (incremental — all documents coexist)")
    print(f"  Run 'python query.py' to search across ALL documents.\n")

    # ── Run a quick verification search ───────────────────────────────────
    print("── Verification Search ──")
    query = "What does the document discuss?"
    search_results = vectorstore.similarity_search(query, k=3)
    print(f"  Query: '{query}'")
    print(f"  Top {len(search_results)} results:")
    for j, doc in enumerate(search_results):
        source = doc.metadata.get("source_doc", "unknown")
        fmt = doc.metadata.get("source_format", "?")
        preview = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        print(f"    [{j+1}] [{fmt.upper()}] [{source}] {preview}")

    print(f"\n✅ All done! {total_chunks} chunks from {len(results)} file(s) are searchable.")


if __name__ == "__main__":
    main()
