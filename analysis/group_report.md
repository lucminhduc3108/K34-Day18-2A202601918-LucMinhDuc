# Group Report — Lab 18: Production RAG

**Sinh viên:** Lục Minh Đức (K34 · Day18 — bài **cá nhân**, tự implement toàn bộ M1→M5)
**Ngày:** 2026-08-18

## Phân công & Trạng thái (self)

| Người | Module | Hoàn thành | Tests pass |
|-------|--------|-----------|-----------|
| Lục Minh Đức | M1: Chunking (semantic / hierarchical / structure) | ✅ | 13/13 |
| Lục Minh Đức | M2: Hybrid Search (BM25 VI + Dense + RRF) | ✅ | 5/5 |
| Lục Minh Đức | M3: Reranking (CrossEncoder bge-reranker-v2-m3) | ✅ | 5/5 |
| Lục Minh Đức | M4: Evaluation (RAGAS + failure analysis) | ✅ | 4/4 |
| Lục Minh Đức | M5: Enrichment (combined 1 call/chunk) | ✅ | 10/10 |

**Tổng: 37/37 tests passed** · `main.py` chạy end-to-end exit 0 (~970s) · 0 TODO bắt buộc còn lại
(chỉ còn `FlashrankReranker` là optional).

## Kết quả RAGAS (20 câu hỏi)

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7125 | **0.7683** | +0.0558 ✓ |
| Answer Relevancy | 0.7258 | 0.7235 | −0.0024 |
| Context Precision | 0.9250 | **0.9583** | +0.0333 ✓ |
| Context Recall | 0.9250 | 0.7667 | −0.1583 ✗ |

## Key Findings

1. **Biggest improvement:** Context Precision (0.925 → 0.958) và Faithfulness (0.712 → 0.768) — nhờ
   **rerank (M3)** đẩy đúng chunk lên top-3 + **enrichment (M5)** contextual-prepend giảm nhiễu ⇒ LLM ít bịa.
2. **Biggest challenge:** **Context Recall giảm −0.16.** `chunk_hierarchical` (child 256) cắt nhỏ tài liệu,
   rerank chỉ giữ top-3 → các bảng ngưỡng/điều kiện (phê duyệt, thang lương) bị vỡ, câu multi-hop & numeric sót chunk.
3. **Surprise finding:** Baseline "ngây thơ" lại **recall cao hơn** production — vì chunk paragraph lớn vô tình
   gói đủ thông tin. Bài học: chunk nhỏ + top-k chặt tối ưu precision nhưng phải bù recall bằng parent-retrieval.

## Presentation Notes (5 phút)

1. **RAGAS naive vs production:** 3/4 metric ≥ 0.75 ở production; precision & faithfulness tăng, recall đánh đổi.
2. **Biggest win:** M3 Reranking + M5 Enrichment → precision/faithfulness.
3. **Case study:** Câu "thiết bị 55 triệu cần ai duyệt" — Error Tree chỉ ra lỗi ở **chunking/retrieval** (context thiếu), không phải generation.
4. **Next optimization (1 giờ):** parent-retrieval thật (retrieve child → return parent) + `chunk_structure_aware`
   cho bảng → hồi phục recall; prompt phủ định để đẩy faithfulness ≥ 0.85.
