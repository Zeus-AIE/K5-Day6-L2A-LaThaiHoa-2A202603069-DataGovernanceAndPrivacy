# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T06 — Dataset Semantic Integrity & Provenance Gate (`t06`) |
| Primary metric (tên đúng như evaluator in ra) | `Silent-defect Detection F1` |
| Tăng tốt hay giảm tốt? | higher (cùng với `Defect recall` & `Root-cause accuracy` higher; `False-block rate` lower) |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | Candidate `release` (mỗi dòng trong `release_decisions.csv` ứng với một `release_id` duy nhất) |
| Công thức + numerator / denominator | `blocked = (canonical_status(decision) != "GO")`; positive GT = (`is_defective == 1`). `TP = sum(is_defective==1 & blocked==1)`, `FP = sum(is_defective==0 & blocked==1)`, `FN = sum(is_defective==1 & blocked==0)`. `Precision = TP / (TP + FP)`, `Defect recall = TP / (TP + FN)`, `Silent-defect Detection F1 = 2 * P * R / (P + R)`, `False-block rate = FP / (TP_clean + FP)` trên các release sạch, `Root-cause accuracy = mean(root_cause == corruption_type)` trên các defective blocked rows. |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Exact join theo `release_id` duy nhất (`gt_releases.csv` ↔ `release_decisions.csv`) qua `eval_t06`. |
| Candidate pool cố định | 20 candidate releases (`r01`..`r20`) trong `fixtures/t06_release_gate/releases.csv`, đối chiếu với parent `releases/v1.0` và `releases/label_maps.json`. |
| Review budget B / K | `B = 10` releases trên `review_queue.csv` (sắp theo anomaly `score` giảm dần, trong `B = 10` bao phủ toàn bộ candidate nghi ngờ trước khi điền decision). |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Đóng băng trên dev split (`r01`..`r10`) dựa vào phân bố clean vs corrupt so với `v1.0`: (1) `parent_sha256` phải khớp SHA-256 của `releases/v1.0/annotations.json`; (2) `categories` map phải khớp chính xác `label_maps[label_map_version]`; (3) `dup_ann_ids == 0` và `dup_img_ids == 0`; (4) không hụt ảnh/frame từ parent; (5) `max_benign_jitter_px = 2.0`, `max_mean_shift_px = 1.5`, không có bbox ngoài biên hoặc invalid (`w<=0` hoặc `h<=0`), và `roundtrip_tol_px = 0.5`. Ngưỡng này giữ cho benign jitter (`<= 1.0 px`) được `PASS` trong khi bắt toàn bộ `bbox_shift` (`~9 px` mean / `~43 px` max). |
| Eval split (không dùng để tune) | `split = eval` (`r11`..`r20`: 10 candidate releases, chỉ dùng để đánh giá `eval_t06` sau khi đóng băng ngưỡng). |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | `Defect recall` (1 - Miss rate trên defective releases) và `False-block rate` (false alarm block nhầm clean release). |
| Slices sẽ báo cáo (≥2) | `corruption_type` (`bbox_shift`, `class_swap`, `dropped_frames`, `id_collision`, `stale_map`, `clean`) và `split` (`dev` vs `eval`). |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson 95% confidence interval cho `Defect recall` và `False-block rate` (N_eval = 10; N_total = 20), vì mẫu đánh giá release nhỏ. |
| Ngày giờ chốt + người chốt | 2026-11-14 — La Thai Hoa (Release Gate Owner / Data Steward, G06) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
