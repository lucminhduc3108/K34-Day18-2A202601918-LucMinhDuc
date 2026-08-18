# Individual Reflection — Lab 18: Production RAG

**Tên:** Lục Minh Đức
**Module phụ trách:** Toàn bộ M1 → M5 (bài cá nhân)
**Kết quả:** 37/37 tests pass · `main.py` end-to-end exit 0 · 3/4 RAGAS metric ≥ 0.75

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (từ lần chạy thực tế) |
|-----------------|--------|------------|-----------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2` + cosine giữa câu liền kề; threshold cao → nhiều chunk nhỏ, threshold thấp → gộp nhiều. Test `groups_by_topic` xác nhận semantic ≤ basic + 2. |
| Parent-child retrieval | M1 | `chunk_hierarchical()` | Parent 2048 / child 256. Sinh 100 child từ 26 docs. Child có `parent_id` trỏ về parent → nền tảng cho "retrieve child, return parent" (chưa bật ở pipeline). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Split theo header markdown `#{1,3}` → giữ nguyên section; quan trọng cho bảng ngưỡng phê duyệt (thứ đã vỡ khi dùng hierarchical). |
| BM25 + Vietnamese tokenization | M2 | `segment_vietnamese()` + `BM25Search` | `underthesea` nối từ ghép bằng `_` ("nghỉ_phép"); phải `.replace("_", " ")` nếu không BM25 tách token sai và query 2 chữ không khớp. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | `score = Σ 1/(k+rank+1)`, k=60. RRF gộp 2 bảng xếp hạng khác thang điểm mà không cần chuẩn hoá score — giải quyết việc BM25 và cosine không cùng đơn vị. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder` (KHÔNG dùng FlagEmbedding vì crash với transformers≥5). Top-20 → top-3. Precision tăng rõ (0.925→0.958). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness / Answer Relevancy / Context Precision / Context Recall. Metric thấp nhất là **Context Recall (0.767)** vì chunk nhỏ + top-3 chặt làm sót thông tin. |
| Diagnostic tree (failure) | M4 | `failure_analysis()` | Map worst_metric → (diagnosis, fix): faithfulness→"LLM bịa", context_recall→"thiếu chunk"… Giúp biết fix ở **retrieval hay generation**. |
| Contextual embeddings (Anthropic) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả vị trí chunk trong tài liệu trước khi embed → giảm retrieval failure, góp phần đẩy faithfulness/precision. |
| Cost optimization | M5 | `_enrich_single_call()` | Gộp summary + questions + context + metadata vào **1 API call/chunk** thay vì 4 → giảm 75% chi phí enrichment (đạt bonus combined mode). |

---

## Phần 2: Khó khăn & cách giải quyết

1. **Python version & dependency cũ.**
   - Vấn đề: repo pin **Python 3.11** nhưng máy mặc định 3.14 / 3.13; các package pin cũ (`ragas<0.2`,
     `langchain<0.3`) dễ thiếu wheel trên bản mới.
   - Giải quyết: tìm ra bản **3.11.15** đã cài sẵn qua `uv`, tạo venv bằng đúng bản này → mọi wheel cp311 cài sạch.

2. **`torch` 122 MB tải hỏng liên tục (network/DNS).**
   - Exact error: `NameResolutionError(... 'files.pythonhosted.org' ... getaddrinfo failed)` và `ReadTimeoutError`,
     pip resume rất chậm, kết thúc bằng `OSError WinError 32` (file bị khoá).
   - Giải quyết: kill pip đang treo → tải riêng wheel bằng `curl -L -C - --retry 100 --retry-all-errors`
     (resume ổn định hơn pip) → `pip install <local wheel>` → cài phần còn lại. Rút ra: với mạng chập chờn,
     tách file lớn ra tải bằng công cụ có resume tốt.

3. **`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cc'`.**
   - Nguyên nhân: `main.py` in emoji (📌) nhưng console Windows dùng cp1252.
   - Giải quyết: chạy với `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (không sửa source).

4. **Enrichment: `Expecting value: line 1 column 1 (char 0)`.**
   - Nguyên nhân: `_enrich_single_call` gọi `json.loads` nhưng LLM thỉnh thoảng trả JSON bọc trong ```
     hoặc kèm chữ → parse fail (khoảng 6/100 chunk).
   - Giải quyết hiện tại: try/except đã có → chunk lỗi fallback về text gốc (pipeline không chết).
   - Cải tiến nên làm: thêm `response_format={"type":"json_object"}` hoặc strip fence trước khi `json.loads`.

- **Thời gian debug:** phần lớn thời gian không phải ở logic module (đã pass test nhanh) mà ở **môi trường**
  (network + encoding) — đúng tinh thần "production RAG khó ở vận hành, không chỉ ở thuật toán".

---

## Phần 3: Action Plan cho project cá nhân

```markdown
## Project: RAG hỏi-đáp tài liệu nội bộ (điều chỉnh theo project thực tế của bạn)

### Hiện tại
- RAG pipeline: chunk cố định + dense-only (giống naive baseline của lab).
- Known issues: câu multi-hop và câu phủ định trả lời sai; đôi khi bịa số liệu.

### Plan áp dụng (rút ra từ lab)
1. [ ] Chunking: dùng **structure-aware cho tài liệu có bảng/điều khoản** (giữ nguyên bảng ngưỡng),
       hierarchical cho văn bản dài — tránh vỡ context như lỗi "55 triệu → CEO".
2. [ ] Search: **Hybrid BM25(VI) + Dense + RRF**. Với tiếng Việt bắt buộc `segment_vietnamese` + `replace("_"," ")`.
3. [ ] Reranking: **có** — `bge-reranker-v2-m3` (CrossEncoder), top-20 → top-3. Đây là đòn tăng precision rẻ nhất.
4. [ ] Evaluation: **RAGAS 4 metrics** + `failure_analysis` diagnostic tree chạy định kỳ trên test set cố định.
5. [ ] Enrichment: **contextual-prepend + combined single-call** (1 API call/chunk) để tiết kiệm chi phí.
6. [ ] Fix riêng cho recall: bật **parent-retrieval** (retrieve child → return parent) — việc lab chưa làm.
7. [ ] Prompt phủ định + temperature=0 để chặn hallucination ở câu "có/không".

### Timeline
- Tuần 1: dựng test set ~30 câu (đủ 6 loại), đo RAGAS baseline hiện tại.
- Tuần 2: thêm Hybrid + Rerank; đo lại, so Δ từng metric.
- Tuần 3: enrichment + parent-retrieval; nhắm 4/4 metric ≥ 0.75, faithfulness ≥ 0.85.
- Tuần 4: hardening vận hành (retry tải model, UTF-8, JSON-mode cho LLM) + latency breakdown.
```

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Ghi chú |
|----------|:---:|---|
| Hiểu bài giảng | 5 | Map được đủ 5 module về concept + quan sát số liệu thật. |
| Code quality | 4 | Modules pass 37/37; điểm trừ: `_enrich_single_call` chưa ép JSON-mode. |
| Teamwork | – | Bài cá nhân. |
| Problem solving | 5 | Xử lý gọn network (curl resume), encoding (UTF-8), version (uv 3.11). |
