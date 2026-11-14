# Báo cáo Nhóm G06 — Track T06: Dataset Semantic Integrity & Provenance Gate

> **Nhóm / Group:** `G06`  
> **Track:** `t06` — *Semantic Integrity & Provenance Gate* (`fixtures/t06_release_gate`)  
> **Người ký duyệt (Owner):** La Thai Hoa (Release Gate Owner / Data Steward)  
> **Cấu hình đóng băng:** [`configs/lab.yaml`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/configs/lab.yaml) · **Seed:** `42` · **Mode:** `--offline` · **Base Git SHA:** `b0ce8bf`

---

## 1. Tóm tắt bài toán & Phạm vi cổng kiểm soát (Problem & Gate Scope)

Trong quy trình phát hành bộ dữ liệu phát hiện đối tượng COCO (`releases/v1.0` gồm `100` ảnh `320x180`, `697` bounding boxes, `5` lớp `car, pedestrian, motorbike, sign, truck`), các bước kiểm tra bề mặt (cú pháp JSON hợp lệ và file ảnh tồn tại trên đĩa) **không phát hiện được lỗi âm thầm (silent corruption)** trên 20 candidate releases (`r01..r20`, gồm `dev = r01..r10` và `eval = r11..r20`).

Nhiệm vụ của Track T06 là xây dựng **Semantic Integrity & Provenance Gate** kiểm tra đa tầng đối với mỗi candidate release so với bản gốc `releases/v1.0` và bảng nhãn chuẩn [`releases/label_maps.json`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/fixtures/t06_release_gate/releases/label_maps.json), đồng thời hỗ trợ vòng lặp **Human-in-the-Loop (HITL)**, kiểm tra **COCO ↔ YOLO lossless round-trip**, và xuất metadata **Croissant 1.1** cùng báo cáo validation suite.

### 5 nhóm lỗi âm thầm (silent corruptions) + lỗi provenance/cú pháp cần chặn
1. **`class_swap`** (`r07`, `r20`): Hoán đổi `category_id` (`{2: 'car', 1: 'pedestrian', ...}` thay vì `{1: 'car', 2: 'pedestrian', ...}`).
2. **`stale_map`** (`r01`, `r09`): Khai báo `label_map_version = "lm-3"` trong `release_meta.json`, nhưng `categories` chỉ có `4` lớp `1..4` (thiếu `5: truck`).
3. **`id_collision`** (`r08`, `r18`): Trùng lặp `annotation.id` (`12` ID trùng trên `r08`, `11` ID trùng trên `r18`).
4. **`dropped_frames`** (`r02`, `r16`): Hụt ảnh/frame so với `v1.0` (`r02` thiếu `9` ảnh -> còn `91`; `r16` thiếu `7` ảnh -> còn `93`, đứt chuỗi `frame_index` trong `sequence_id`).
5. **`bbox_shift`** (`r05`, `r10`): Tọa độ bounding box bị dịch chuyển vượt ngưỡng (`max_shift` ~ `29.75–29.85 px`, `mean_shift` ~ `6.19–6.39 px`), khác biệt rõ so với nhiễu lành tính trên 10 bản sạch (`max_jitter <= 1.00 px`, `mean_shift <= 0.0062 px`).
6. **`corrupt_json` / `provenance_mismatch`**: JSON hỏng/bị cắt cụt hoặc sai lệch `parent_sha256` / `manifest.csv`.

---

## 2. Khai báo Metric Contract & Self-Evaluation Disclosure

Metric contract đã được đóng băng trước khi chạy đánh giá tại [`reports/metric_contract.md`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/reports/metric_contract.md):
- **Primary metric:** `Silent-defect Detection F1` (higher is better; positive = `is_defective == 1` và `canonical_status(decision) != "GO"`).
- **Metric phụ:** `Defect recall` (higher), `False-block rate` (lower), `Root-cause accuracy` (higher).
- **Review budget:** `B = 10` releases trong `review_queue.csv` (sắp xếp theo anomaly `score` giảm dần).
- **Ngưỡng đóng băng trên `dev` (`r01..r10`):** `max_benign_jitter_px = 2.0`, `max_mean_shift_px = 1.5`, `roundtrip_tol_px = 0.5`.
- **Eval split cố định:** `split = eval` (`r11..r20`, N = 10) theo [`configs/tracks/t06.yaml`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/configs/tracks/t06.yaml).

> **Minh bạch về Ground Truth (Self-Eval Disclosure):** Repository sinh viên không chứa `fixtures/t06_release_gate/gt/gt_releases.csv` (file GT chính thức nằm ở gói chấm của giảng viên). Nhóm tạo tập tự kiểm chứng độc lập tại `out/self_eval/gt_releases.csv` suy diễn trực tiếp từ đối chiếu cấu trúc/tọa độ với `releases/v1.0/annotations.json` và `releases/label_maps.json`, **tuyệt đối không** đọc GT trong code pipeline và **không** chỉnh sửa bất kỳ file nào dưới `fixtures/`.

---

## 3. Thiết kế Pipeline 5 bước (`src/pipeline.py` vs `src/baseline.py`)

### Baseline ([`src/baseline.py`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/src/baseline.py))
- Chỉ kiểm tra `annotations.json` đọc được, có đủ khóa COCO cấp cao (`images`, `annotations`, `categories`) và toàn bộ `file_name` tồn tại trên đĩa.
- Hệ quả: Toàn bộ `20/20` releases (kể cả `10` bản lỗi âm thầm) đều nhận `GO` -> `Defect recall = 0.0000`, `Silent-defect Detection F1 = 0.0000`.

### PoC Pipeline ([`src/pipeline.py`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/src/pipeline.py))
1. **`01 load(ctx)`:** Kiểm chứng toàn bộ SHA-256 trong `manifest.csv` (`strict_provenance: true`), đọc `releases.csv`, `releases/label_maps.json`, và lập chỉ mục parent `releases/v1.0` (`sha256_file` thực tế của `annotations.json`, `classes.txt`, `yolo_labels/`).
2. **`02 process(ctx, data)` — Thứ tự cổng kiểm tra bắt buộc:**
   - **Gate 0 (`corrupt_json`):** Parse JSON & kiểm tra schema (`width > 0`, `height > 0`, `bbox` 4 số hữu hạn). Cô lập ngay nếu lỗi mà không làm sập pipeline khi `strict_provenance: false`.
   - **Gate 1 (`provenance_mismatch`):** So khớp `release_meta.json["parent_sha256"]` với SHA-256 thực của `v1.0/annotations.json`.
   - **Gate 2 (`id_collision`) — CHẠY TRƯỚC BBOX/CLASS DIFF:** Kiểm tra trùng `annotation.id` và `image.id`. *(Quan trọng: nếu lập dict `ann_id -> bbox` trước khi check trùng ID, các bản `r08`/`r18` sẽ bị ghi đè khóa trong dict và bị gán nhầm nguyên nhân thành `bbox_shift`/`class_swap` thay vì `id_collision`).*
   - **Gate 3 (`class_swap` vs `stale_map`):** So khớp `categories` với `label_maps[label_map_version]`. Nếu trùng tập tên nhãn nhưng sai hoán vị ID -> `class_swap`; nếu thiếu/lệch bảng nhãn -> `stale_map`.
   - **Gate 4 (`dropped_frames`):** Kiểm tra thiếu ảnh từ `v1.0`, orphan annotations (`image_id` không tồn tại), và đứt gãy `frame_index` theo `sequence_id`.
   - **Gate 5 (`bbox_shift` & bounds):** Kiểm tra bbox ngoài biên hoặc `w <= 0`/`h <= 0`; khi không có trùng ID, đo `L_inf` drift giữa các annotation chung với `v1.0` (`max_shift > 2.0 px` hoặc `mean_shift > 1.5 px`).
   - **Bổ trợ:** Thực thi kiểm tra chuyển đổi **COCO ↔ YOLO round-trip** (`out/poc/yolo_roundtrip_report.json`), xuất **Croissant 1.1 JSON-LD** (`out/poc/croissant_metadata.json`), và **Validation Suite Report** (`out/poc/validation_report.json`).
3. **`03 rank(ctx, scored)`:** Tạo `review_queue.csv` top `B = 10` (bao trùm `10/10` candidate nghi ngờ với `score` `0.92–0.98`, toàn bộ `10` bản sạch có `score <= 0.0001`), và render [`out/poc/contact_sheet.html`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/out/poc/contact_sheet.html) kèm bounding-box overlay so sánh trực quan với `v1.0`.
4. **`04 review(ctx, scored, queue)`:** Đọc `out/poc/review_queue_reviewed.csv` qua `load_review_decisions` (bắt buộc non-empty `reviewer`), ghi toàn bộ quyết định (`review:approve`, `review:edit`, `override:edit`) vào [`out/poc/audit_log.jsonl`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/out/poc/audit_log.jsonl), áp dụng override lên `decision`.
5. **`05 decide(ctx, final)`:** Xuất `decision.json` và `decision_log.csv` có người ký duyệt (`La Thai Hoa`), danh sách bằng chứng và breakdown nguyên nhân gốc.

---

## 4. Kết quả định lượng (Baseline vs PoC)

### Bảng 1 — So sánh trên `eval` split (`r11..r20`, từ [`reports/table1_baseline_vs_poc.md`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/reports/table1_baseline_vs_poc.md))

Baseline: `baseline` (git b0ce8bf, seed 42) · PoC: `poc` (git b0ce8bf, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Silent-defect Detection F1** (F1, unit = release, positive = defective & blocked) | all | 0.0000 | 1.0000 | 1.0000 | baseline=0 | higher | 10 |
| Defect recall (defective releases blocked / defective) | all | 0.0000 | 1.0000 | 1.0000 | baseline=0 | higher | None |
| False-block rate (clean releases blocked / clean) | all | 0.0000 | 0.0000 | 0.0000 | baseline=0 | lower | None |
| Root-cause accuracy (correct root cause / defective blocked) | all | n/a | 1.0000 | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 2.9475 | 4.9560 | | | lower | |

*(Lưu ý: Trên các chỉ số có Baseline = 0, độ cải thiện được ghi rõ là `baseline=0` thay vì % vô nghĩa. Thời gian chạy thực tế cho 20 releases: Baseline `0.5895s`, PoC `0.9912s`.)*

### Khoảng tin cậy Wilson 95% (cho cỡ mẫu N nhỏ) & Slice Breakdown

- **`eval` split (`N = 10`):**
  - `Defect recall = 3/3 = 1.0000` -> Wilson 95% CI: `[0.4385, 1.0000]`
  - `False-block rate = 0/7 = 0.0000` -> Wilson 95% CI: `[0.0000, 0.3543]`
  - Slices (`reports/slices_poc.csv`): `class_swap = 1.0`, `dropped_frames = 1.0`, `id_collision = 1.0`, `clean = 0.0` (Baseline: toàn bộ `0.0`).
- **Toàn bộ 20 releases (`dev + eval`, `split=all`, `N = 20`):**
  - `Silent-defect Detection F1 = 1.0000`, `Defect recall = 10/10 = 1.0000` (Wilson 95% CI: `[0.7225, 1.0000]`), `False-block rate = 0/10 = 0.0000` (Wilson 95% CI: `[0.0000, 0.2775]`), `Root-cause accuracy = 10/10 = 1.0000`.
  - Slices (`out/self_eval/slices_poc_all.csv`): `bbox_shift = 1.0`, `class_swap = 1.0`, `dropped_frames = 1.0`, `id_collision = 1.0`, `stale_map = 1.0`, `clean = 0.0`.

### Kiểm chứng bổ trợ (COCO ↔ YOLO Round-Trip & Croissant 1.1)
- **`yolo_roundtrip_report.json`:** Kiểm tra `100` ảnh và `697` boxes trên `releases/v1.0` ↔ `yolo_labels/*.txt`: `100/100` file khớp hoàn toàn (`max_roundtrip_error_px = 0.0`, `max_yolo_file_norm_diff = 5e-07`, ngưỡng `0.5 px`, `passed = true`).
- **`croissant_metadata.json`:** Xuất chuẩn `http://mlcommons.org/croissant/1.1` ghi nhận SHA-256 lineage, license `CC0-1.0`, distributions và RecordSet.

---

## 5. Vòng lặp Human-in-the-Loop & Audit Trail

1. Chạy PoC lần 1 xuất `out/poc/review_queue.csv` gồm đúng `10` bản nghi ngờ (`r07, r20, r08, r18, r02, r16, r10, r05, r01, r09`) và render `contact_sheet.html`.
2. Người review `La Thai Hoa` kiểm tra trên `review_queue_reviewed.csv`:
   - **`approve`** (`8` bản lỗi nghiêm trọng `r07, r20, r08, r18, r02, r16, r10, r05`): giữ `HOLD`.
   - **`edit` -> `override_value = REWORK`** (`r01, r09` có `root_cause = stale_map`): chuyển trạng thái từ `HOLD` sang `REWORK` để đội labeling tái tạo `categories` theo đúng `lm-3`.
3. Chạy PoC lần 2 áp dụng override, ghi đủ `12` bản ghi audit (`review:approve`, `review:edit`, `override:edit`) vào `out/poc/audit_log.jsonl` và lưu chứng từ tại [`reports/review_queue.csv`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/reports/review_queue.csv) và [`reports/decision_log.csv`](file:///d:/App/K5-Day6-L2A-LaThaiHoa-2A202603069-DataGovernanceAndPrivacy/reports/decision_log.csv).

---

## 6. Phân tích lỗi, Shock Test & Hạn chế

1. **Bẫy thứ tự phát hiện (`id_collision` trước `bbox_shift`):** Thực nghiệm phát hiện nếu xây dựng dictionary `{ann_id: ann}` trước khi đếm ID trùng thì trên `r08`/`r18` các annotation trùng sẽ ghi đè lẫn nhau, gây lệch `max_shift` giả. Đặt Gate `id_collision` lên trước loại bỏ triệt để lỗi phân loại nguyên nhân này.
2. **Shock test (`cut half of releases/candidates/r03/annotations.json`):**
   - Với `strict_provenance: true`, `verify_manifest` chặn ngay lập tức với exit code `2` (không crash traceback) và ghi `provenance_problems` vào `audit_log.jsonl`.
   - Khi quét candidate chưa niêm phong hash (`strict_provenance: false`), Gate 0 cô lập `r03` thành `HOLD (corrupt_json)` trong khi `r04` vẫn `GO` và các release khác không bị ảnh hưởng.
3. **Hạn chế:** Cổng `bbox_shift` dựa trên định danh `annotation.id` đối chiếu với parent release; nếu một release sạch thực sự tái đánh số toàn bộ `annotation.id` từ đầu hoặc chứa re-annotation hợp lệ có độ lệch > `2.0 px`, cổng hiện tại sẽ gắn cờ cần human review (`HOLD`) để tránh lọt lỗi âm thầm.

---

## 7. Hướng dẫn tái lập (Reproducibility Commands)

```bash
# Baseline (syntactic + file existence)
python -m src.main --track t06 --pipeline src.baseline --output out/baseline --run-name baseline --offline --seed 42

# PoC Gate (full semantic + provenance + HITL review + Croissant + YOLO roundtrip)
python -m src.main --config configs/lab.yaml --track t06 --output out/poc --run-name poc --offline --seed 42

# Evaluation & Table 1 comparison
python -m src.evaluate --track t06 --config configs/lab.yaml --gt out/self_eval/gt_releases.csv --pred out/baseline/release_decisions.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t06 --config configs/lab.yaml --gt out/self_eval/gt_releases.csv --pred out/poc/release_decisions.csv --run-name poc --out reports/metrics_poc.json --slices-out reports/slices_poc.csv
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md

# Verification suite
python -m tools.verify_manifests
python -m tools.preflight
python -m pytest -v
```
