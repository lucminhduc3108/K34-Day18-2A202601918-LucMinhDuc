# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Lục Minh Đức (bài cá nhân — M1→M5)
**Test set:** 20 câu hỏi tiếng Việt (lookup / version / negation / multi-hop / numeric / ambiguous)
**Ngày chạy:** 2026-08-18 · `main.py` end-to-end (exit 0, ~970s)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7125 | **0.7683** | **+0.0558** ✓ |
| Answer Relevancy | 0.7258 | 0.7235 | −0.0024 |
| Context Precision | 0.9250 | **0.9583** | **+0.0333** ✓ |
| Context Recall | 0.9250 | 0.7667 | **−0.1583** ✗ |

**Đọc kết quả:**
- **Precision ↑ + Faithfulness ↑**: reranking (M3) + enrichment (M5) đẩy đúng chunk lên top-3 và cắt
  nhiễu → LLM bám context tốt hơn, ít bịa hơn. Đây là "win" chính của pipeline production.
- **Context Recall ↓ (−0.16)**: đây là **regression đáng chú ý**. Baseline dùng `chunk_basic` (paragraph,
  57 chunks lớn) → mỗi chunk chứa nhiều thông tin nên "vô tình" recall cao. Production dùng
  `chunk_hierarchical` (child 256 ký tự, 100 chunks nhỏ) + rerank chỉ giữ top-3 → chunk nhỏ hơn,
  giữ ít hơn ⇒ với câu hỏi cần gộp nhiều mảnh (multi-hop, ngưỡng phê duyệt) dễ **sót chunk**.
- Trade-off điển hình: **precision-recall**. Production tối ưu precision/faithfulness, đánh đổi recall.

---

## Bottom-5 Failures (từ `reports/ragas_report.json`)

### #1 — Multi-hop (thấp nhất)
- **Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm **và** lương trong khoảng nào?"
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3) = **18 ngày**; lương Senior (P3–P4) **20–35 triệu/tháng**.
- **Got:** trả lời được một vế (ngày phép hoặc lương) nhưng không ghép đủ 2 vế → lệch câu hỏi.
- **Worst metric:** answer_relevancy = **0.4583**
- **Error Tree:** Output sai → Context đúng một phần (thiếu bảng lương) → Query OK → **fix ở retrieval + prompt**
- **Root cause:** Câu hỏi multi-hop cần 2 nguồn (chính sách phép + thang lương). Top-3 sau rerank chỉ
  lấy được cụm "phép năm", không kéo được cụm "lương Senior" ⇒ answer thiếu vế → relevancy thấp.
- **Suggested fix:** Multi-query / query decomposition (tách thành 2 sub-query), hoặc tăng `RERANK_TOP_K`
  cho câu multi-hop; prompt yêu cầu trả lời đủ mọi vế của câu hỏi.

### #2 — Negation
- **Question:** "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"
- **Expected:** **KHÔNG** — thử việc chưa được hưởng PVI, chỉ có BHXH bắt buộc.
- **Got:** khẳng định sai hướng / mơ hồ (không nắm được mệnh đề phủ định trong context).
- **Worst metric:** faithfulness = **0.5000**
- **Error Tree:** Output sai → Context đúng → Query OK → **fix ở generation (prompt)**
- **Root cause:** Câu phủ định. Context có nhắc "PVI" và "thử việc" ở các chunk khác nhau; LLM bắt được
  từ khóa "PVI 200 triệu" rồi suy diễn "có" → hallucination.
- **Suggested fix:** Prompt nhấn mạnh xử lý phủ định ("nếu context nói KHÔNG/chưa/loại trừ, phải trả lời
  KHÔNG"), temperature=0; contextual-prepend ghi rõ chunk thuộc nhóm "thử việc".

### #3 — Negation
- **Question:** "Nhân viên thử việc có được nghỉ phép năm không?"
- **Expected:** **KHÔNG** — thử việc không được nghỉ phép năm; muốn nghỉ phải xin không lương + trưởng phòng duyệt.
- **Got:** trả lời khẳng định/mơ hồ.
- **Worst metric:** faithfulness = **0.5000**
- **Error Tree:** Output sai → Context đúng → Query OK → **fix ở generation (prompt)**
- **Root cause:** Giống #2. Chunk "phép năm 15 ngày" lấn át chunk "thử việc KHÔNG được nghỉ" → LLM trộn 2 ý.
- **Suggested fix:** như #2; thêm rule "ưu tiên mệnh đề áp dụng cho đối tượng trong câu hỏi (thử việc)".

### #4 — Numeric / ngưỡng phê duyệt
- **Question:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"
- **Expected:** Trên 50 triệu → **Tổng Giám đốc (CEO)** phê duyệt.
- **Got:** trả sai cấp (Director) hoặc thiếu căn cứ.
- **Worst metric:** context_recall = **0.6578**
- **Error Tree:** Output sai → **Context thiếu** (không kéo được dòng ngưỡng >50tr) → Query OK → **fix ở retrieval/chunking**
- **Root cause:** Bảng ngưỡng phê duyệt bị `chunk_hierarchical` cắt nhỏ (child 256) → dòng ">50 triệu = CEO"
  rơi khỏi top-3. Đây chính là nguyên nhân recall giảm.
- **Suggested fix:** `chunk_structure_aware` để giữ nguyên bảng ngưỡng trong 1 chunk; hoặc parent-retrieval
  (retrieve child → trả parent) để không mất ngữ cảnh bảng.

### #5 — Numeric
- **Question:** "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"
- **Expected:** Junior cao nhất 20 triệu → thử việc = 85% × 20 = **17 triệu/tháng**.
- **Got:** đúng số nền nhưng suy luận % chưa chắc / trích dẫn lệch → faithfulness giảm.
- **Worst metric:** faithfulness = **0.7060**
- **Error Tree:** Output gần đúng → Context đúng → Query OK → **fix ở generation (yêu cầu show phép tính)**
- **Root cause:** Cần phép tính 2 bước (tra 20tr → nhân 85%). LLM đưa số nhưng không neo chặt vào context.
- **Suggested fix:** Prompt "chỉ dùng con số có trong context, trình bày phép tính"; hoặc tool/calculator.

---

## Case Study (cho presentation) — Câu #4 "thiết bị 55 triệu"

**Error Tree walkthrough:**
1. Output đúng? → **Sai** (không chỉ ra CEO).
2. Context đúng? → **Sai** — top-3 không chứa dòng ngưỡng ">50 triệu → CEO" (context_recall 0.66).
3. Query rewrite OK? → **OK** ("ai phê duyệt" đủ rõ, BM25 khớp "phê duyệt").
4. Fix ở bước: **Chunking + Retrieval** (không phải generation).

**Vì sao đây là ví dụ tốt:** nó giải thích trực tiếp regression **context_recall −0.16** của toàn pipeline —
chunk hierarchical quá nhỏ làm vỡ các bảng ngưỡng/điều kiện.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Chuyển tài liệu dạng bảng (ngưỡng phê duyệt, thang lương) sang `chunk_structure_aware` để giữ nguyên bảng.
- Bật **parent-retrieval** thực sự (hiện child có `parent_id` nhưng pipeline chỉ index child): retrieve child
  → trả parent để lấy lại ngữ cảnh → kỳ vọng recall hồi phục mà vẫn giữ precision.
- Prompt cứng rắn hơn với câu phủ định (fix #2, #3) → kéo faithfulness ≥ 0.85 (đạt bonus +3).
