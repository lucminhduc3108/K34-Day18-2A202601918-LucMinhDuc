from __future__ import annotations

"""
Latency Breakdown Benchmark — Lab 18 (bonus deliverable)
========================================================
Đo thời gian từng bước của pipeline production và ghi `reports/latency_report.json`.

Cố ý KHÔNG chạy RAGAS và chỉ gọi vài API enrichment (mẫu nhỏ) để rẻ & nhanh.
Các bước đo: Chunking → Enrichment (mẫu) → Indexing (BM25+Dense) → Hybrid Search
→ Rerank (CrossEncoder) → Rerank (Flashrank, để so sánh) → End-to-end retrieval.

Run:  python src/latency_benchmark.py
"""

import os, sys, json, time, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker, FlashrankReranker
from src.m4_eval import load_test_set
from src.m5_enrichment import enrich_chunks

N_QUERIES = 8          # số query để đo search/rerank
ENRICH_SAMPLE = 3      # số chunk gọi API enrichment (giữ chi phí thấp)


def _ms(sec: float) -> float:
    return round(sec * 1000, 1)


def _stats(times: list[float]) -> dict:
    return {
        "avg_ms": _ms(statistics.mean(times)),
        "min_ms": _ms(min(times)),
        "max_ms": _ms(max(times)),
    }


def main():
    report: dict = {}
    print("=" * 60)
    print("LATENCY BREAKDOWN BENCHMARK")
    print("=" * 60, flush=True)

    # 1. Chunking (M1) — no API
    docs = load_documents()
    t = time.perf_counter()
    chunks = []
    for d in docs:
        _, children = chunk_hierarchical(d["text"], metadata=d["metadata"])
        for c in children:
            chunks.append({"text": c.text, "metadata": {**c.metadata, "parent_id": c.parent_id}})
    chunk_ms = _ms(time.perf_counter() - t)
    report["chunking"] = {
        "n_docs": len(docs), "n_chunks": len(chunks),
        "total_ms": chunk_ms, "per_doc_ms": round(chunk_ms / max(len(docs), 1), 2),
    }
    print(f"[1] Chunking: {len(chunks)} chunks / {len(docs)} docs in {chunk_ms} ms", flush=True)

    # 2. Enrichment (M5) — API-bound, đo trên mẫu nhỏ rồi ngoại suy
    sample = chunks[:ENRICH_SAMPLE]
    t = time.perf_counter()
    enrich_chunks(sample)
    enrich_ms = _ms(time.perf_counter() - t)
    per_chunk = round(enrich_ms / max(len(sample), 1), 1)
    report["enrichment"] = {
        "sampled_chunks": len(sample), "total_ms": enrich_ms, "per_chunk_ms": per_chunk,
        "projected_full_corpus_ms": round(per_chunk * len(chunks), 1),
        "note": "API-bound, 1 call/chunk (combined mode)",
    }
    print(f"[2] Enrichment: {per_chunk} ms/chunk (API) → ~{report['enrichment']['projected_full_corpus_ms']} ms cho {len(chunks)} chunks", flush=True)

    # 3. Indexing (M2): BM25 + Dense encode
    search = HybridSearch()
    t = time.perf_counter()
    search.index(chunks)
    index_ms = _ms(time.perf_counter() - t)
    report["indexing"] = {
        "n_chunks": len(chunks), "total_ms": index_ms,
        "per_chunk_ms": round(index_ms / max(len(chunks), 1), 1),
    }
    print(f"[3] Indexing (BM25+Dense): {index_ms} ms / {len(chunks)} chunks", flush=True)

    # 4. Hybrid search per query
    queries = [q["question"] for q in load_test_set()[:N_QUERIES]]
    per_query_docs, search_times = [], []
    for q in queries:
        t = time.perf_counter()
        results = search.search(q)
        search_times.append(time.perf_counter() - t)
        per_query_docs.append([{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results])
    report["search_hybrid"] = {"n_queries": len(queries), **_stats(search_times)}
    print(f"[4] Hybrid search: {report['search_hybrid']['avg_ms']} ms avg / {len(queries)} queries", flush=True)

    # 5. Rerank — CrossEncoder (required)
    ce = CrossEncoderReranker()
    ce.rerank(queries[0], per_query_docs[0])  # warmup (load model once)
    ce_times = []
    for q, docs_q in zip(queries, per_query_docs):
        t = time.perf_counter()
        ce.rerank(q, docs_q)
        ce_times.append(time.perf_counter() - t)
    report["rerank_crossencoder"] = {"model": "BAAI/bge-reranker-v2-m3", **_stats(ce_times)}
    print(f"[5] Rerank CrossEncoder: {report['rerank_crossencoder']['avg_ms']} ms avg", flush=True)

    # 6. Rerank — Flashrank (lightweight alternative, optional)
    try:
        fr = FlashrankReranker()
        fr.rerank(queries[0], per_query_docs[0])  # warmup + model download
        fr_times = []
        for q, docs_q in zip(queries, per_query_docs):
            t = time.perf_counter()
            fr.rerank(q, docs_q)
            fr_times.append(time.perf_counter() - t)
        report["rerank_flashrank"] = {"model": "ms-marco-TinyBERT (default)", **_stats(fr_times)}
        print(f"[6] Rerank Flashrank: {report['rerank_flashrank']['avg_ms']} ms avg", flush=True)
    except Exception as e:
        report["rerank_flashrank"] = {"error": str(e)}
        print(f"[6] Rerank Flashrank: skipped ({e})", flush=True)

    # 7. End-to-end retrieval (search + CrossEncoder rerank), no LLM
    e2e = [s + c for s, c in zip(search_times, ce_times)]
    report["end_to_end_retrieval"] = {"note": "hybrid search + cross-encoder rerank (no LLM)", **_stats(e2e)}
    print(f"[7] End-to-end retrieval: {report['end_to_end_retrieval']['avg_ms']} ms avg", flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nSaved reports/latency_report.json")

    # Markdown table (copy vào report)
    print("\n" + "=" * 60)
    print("| Stage | Metric | Value |")
    print("|-------|--------|-------|")
    print(f"| Chunking (M1) | total / {report['chunking']['n_chunks']} chunks | {report['chunking']['total_ms']} ms |")
    print(f"| Enrichment (M5) | per chunk (API) | {report['enrichment']['per_chunk_ms']} ms |")
    print(f"| Indexing (M2) | total / {report['indexing']['n_chunks']} chunks | {report['indexing']['total_ms']} ms |")
    print(f"| Hybrid Search (M2) | avg/query | {report['search_hybrid']['avg_ms']} ms |")
    print(f"| Rerank CrossEncoder (M3) | avg/query | {report['rerank_crossencoder']['avg_ms']} ms |")
    fr = report.get("rerank_flashrank", {})
    print(f"| Rerank Flashrank (M3) | avg/query | {fr.get('avg_ms', 'n/a')} ms |")
    print(f"| **End-to-end retrieval** | avg/query | **{report['end_to_end_retrieval']['avg_ms']} ms** |")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal benchmark time: {time.time() - start:.1f}s")
