# Latency Breakdown — Lab 18: Production RAG (bonus)

**Sinh viên:** Lục Minh Đức · **Ngày:** 2026-08-18
**Nguồn:** `python src/latency_benchmark.py` → `reports/latency_report.json`
**Máy:** CPU-only, Python 3.11.15, Qdrant local · corpus 100 chunks / 26 docs · 8 query mẫu

---

## Bảng thời gian từng bước

| Stage | Loại | Metric | Value |
|-------|------|--------|------:|
| Chunking (M1) | offline, 1 lần | total / 100 chunks | **0.4 ms** |
| Enrichment (M5) | offline, 1 lần | per chunk (API, combined) | **3618.6 ms** |
| Indexing BM25+Dense (M2) | offline, 1 lần | total / 100 chunks | **31659.6 ms** |
| Hybrid Search (M2) | **per-query** | avg | **148.9 ms** |
| Rerank CrossEncoder (M3) | **per-query** | avg | **3688.8 ms** |
| Rerank Flashrank (M3) | **per-query** | avg | **10.1 ms** |
| **End-to-end retrieval** (search + CE rerank, no LLM) | **per-query** | avg | **3837.8 ms** |

> Số liệu min/max đầy đủ nằm trong `reports/latency_report.json`.

---

## Phân tích

**1. Rerank là nút cổ chai của truy vấn (per-query).**
CrossEncoder (`bge-reranker-v2-m3`) chiếm **3688.8 / 3837.8 ≈ 96%** thời gian end-to-end retrieval.
Trên CPU, cross-encoder phải chạy 1 forward pass cho *mỗi cặp (query, doc)* trên ~20 doc → chậm.
Đây là cái giá của việc precision tăng (context_precision 0.925 → 0.958).

**2. Flashrank rẻ hơn ~365×.**
TinyBERT (Flashrank) chỉ **10.1 ms/query** so với 3688.8 ms của cross-encoder — đúng như quảng cáo
"lightweight". Đánh đổi: mô hình nhỏ nên độ chính xác rerank thấp hơn bge-reranker-v2-m3 (đa ngôn ngữ,
mạnh tiếng Việt). → **Chọn theo SLA:** cần độ chính xác cao dùng CrossEncoder; cần latency thấp / QPS cao
dùng Flashrank, hoặc chạy CrossEncoder trên GPU.

**3. Chi phí offline vs online.**
- *Offline (index-time, trả 1 lần):* Chunking ~0.4 ms (không đáng kể), Enrichment ~3.6 s/chunk (API-bound,
  100 chunk ≈ 6 phút), Indexing ~31.7 s (encode bge-m3 trên CPU). Những bước này KHÔNG ảnh hưởng latency
  người dùng — chỉ chạy khi build index.
- *Online (query-time):* Hybrid search **148.9 ms** (nhanh), rerank là phần cần tối ưu.

**4. Search nhanh, không phải chỗ cần lo.**
BM25 (in-memory) + Dense (Qdrant `query_points`) + RRF chỉ tốn ~149 ms/query — chấp nhận được.

---

## Đề xuất tối ưu (theo thứ tự ROI)
1. **Đưa CrossEncoder lên GPU** hoặc giảm số doc vào rerank (top-20 → top-10) → cắt mạnh 96% bottleneck.
2. **Two-stage rerank:** Flashrank lọc 20 → 8 (10 ms), rồi CrossEncoder chấm 8 → 3 → giảm ~60% chi phí CE mà giữ chất lượng.
3. **Batch/cache enrichment & embeddings** ở index-time (đã là offline nên ưu tiên thấp cho latency).
4. **Cache câu hỏi lặp** (semantic cache) để bỏ qua cả search + rerank cho truy vấn trùng.
