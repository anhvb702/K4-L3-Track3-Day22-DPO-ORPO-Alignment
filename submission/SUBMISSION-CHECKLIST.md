# Trạng thái bài nộp NB0–NB4

## File chính để người chấm mở

- [NB0 đã hoàn thiện, có output CPU thật](<../notebooks/00_dpo_loss_from_scratch.ipynb>): `my_dpo_loss` khớp công thức, loss khởi đầu = log(2), gradient hữu hạn và ví dụ likelihood displacement qua kiểm tra. Hoàn thành sau lượt chạy GPU; không sửa output trong notebook huấn luyện gốc.

- [Notebook huấn luyện gốc](<../Lab22_DPO_T4.ipynb>): output NB1–NB3 và sinh 58 câu NB4; có lỗi OOM ở lần chấm đầu. Giữ nguyên để không sửa lịch sử thực nghiệm.
- [Notebook khôi phục NB4](<../notebook8375f43734.ipynb>): có output judge và tổng hợp thành công sau khi đọc lại JSONL; các lỗi thử khôi phục/bonus vẫn được giữ, không dùng làm bằng chứng bonus.
- [Phản tư](<REFLECTION.md>): đã tổng hợp số liệu và phân tích, người học cần xác nhận thông tin cá nhân và quyết định.
- [Kết quả judge](<../data/eval/judge_summary.json>) và [phiếu chấm từng cặp](<../data/eval/judge_results_rm.json>).
- [Câu trả lời gốc](<../data/eval/side_by_side.jsonl>), [train](<../data/pref/train.parquet>), [held-out](<../data/pref/eval.parquet>) và [thống kê độ dài](<../data/pref/stats.json>).
- [DPO metrics](<../adapters/dpo/dpo_metrics.json>) và [adapter config gốc](<../adapters/dpo/adapter_config.json>).
- 4 ảnh bắt buộc: [SFT loss](<screenshots/02-sft-loss.png>), [thiên vị độ dài](<screenshots/02b-pref-length.png>), [DPO reward](<screenshots/03-dpo-reward-curves.png>), [8 câu so sánh](<screenshots/04-side-by-side-table.png>).

## Đã xác minh, không cần chạy lại GPU chỉ để có các số liệu này

- 800 train / 100 held-out, không trùng prompt sau chuẩn hoá.
- 50 câu chấm held-out là 50 prompt khác nhau và nằm trong held-out đã cung cấp; cộng 8 câu cố định = 58.
- SHA256 của JSONL khớp cả judge results và summary.
- Phiếu chấm giữ nguyên prompt và câu trả lời của JSONL; quyết định hội đồng khớp hai judge thành viên.
- Tính lại số thắng/thua/hoà, win rate, CI bootstrap 95%, độ dài và đồng thuận đều khớp summary.
- 4 ảnh bắt buộc là PNG thật, mở được.
- Held-out win rate 58% (CI 51–65%), 11 DPO thắng / 3 SFT thắng / 36 hoà; sanity của cả hai judge 100%.

Kiểm tra riêng bằng `python scripts/audit_submission.py`. Đây chỉ là kiểm tra file bằng chứng, **không thay thế** `make verify` và không chứng minh pipeline GPU chạy lại thành công từ đầu.

## Kiểm tra mã nguồn

- NB0: bài kiểm tra đã thất bại khi hàm còn `return None`, sau đó qua khi triển khai công thức logsigmoid; notebook bổ sung đã thực thi 9 code cell thật trên CPU.
- Bộ CPU của repo: `python -m pytest -q scripts/` — **54 passed**, không bỏ/skip bài kiểm tra nào.
- Colab bundles đã sinh lại cho khớp nguồn NB0; bản Kaggle riêng ở [Lab22_DPO_Kaggle.ipynb](<../colab/Lab22_DPO_Kaggle.ipynb>). Không thay đổi hai notebook đã chạy do người học cung cấp; không cần chạy lại GPU để có NB0.
- [Kết quả verify nguyên bản](<VERIFY-RESULT.txt>): vẫn exit 1 vì các thiếu sót dưới đây. Bộ kiểm tra và tiêu chí chấm không bị sửa để che thiếu bằng chứng.

## Còn cần người học cung cấp/xác nhận

1. Đã điền Vũ Bá Anh, MSSV 2A202602893, Track 3B. Chi phí thực tế chưa được xác nhận. Đọc và xác nhận [REFLECTION.md](<REFLECTION.md>), nhất là §6 là phân tích hậu nghiệm, không tự nhận các quyết định chưa làm.
2. **Đã bổ sung đủ ba metadata:** [split.json](<../adapters/dpo/split.json>), [config adapter SFT](<../adapters/sft-mini/adapter_config.json>) và [config SFT gộp](<../models/sft-merged/config.json>). Hai hash trong split khớp chính xác dữ liệu train/eval. Đã cho phép commit riêng config SFT gộp trong gitignore, vẫn chặn trọng số; không cần chạy lại GPU để khôi phục metadata.
3. Config DPO gốc trỏ tới `/kaggle/working/lab22/models/sft-merged`. Bộ `scripts/verify.py` so đường dẫn tuyệt đối với repo hiện tại trên Windows, nên báo WRONG REF khi chuyển môi trường. Không sửa config lịch sử cho vừa bộ kiểm tra. Nếu metadata được bổ sung, vẫn cần xác minh trên môi trường phù hợp hoặc trao đổi với trợ giảng về kiểm tra repo chỉ có bằng chứng nhỏ.
4. Hiện chưa chứng minh Colab/Kaggle Run All từ môi trường sạch thành công; notebook ghi nhận phải khôi phục judge sau OOM. Không tuyên bố tái lập end-to-end đã đạt.
5. Chỉ file đã commit mới được chấm. Kiểm tra `git status`, commit các notebook chạy thật, NB0 bổ sung, phản tư, ảnh và bằng chứng ở thư mục chuẩn; không commit `.env`, API key, thư mục phụ thuộc CPU `.verification-deps/`, trọng số hoặc download `.crdownload`.
6. Tạo repo public, giữ public đến khi có điểm và nộp link LMS. Không tự push/commit từ phiên này khi chưa được yêu cầu.

## Bonus

Không cần NB5–NB7 để hoàn thành phần bắt buộc. Có output NB3b và ảnh biến thể nhưng thiếu summary gốc/baseline so độ dài; không nhận bonus đã hoàn chỉnh. Các lỗi bonus trong notebook bổ sung không phải bằng chứng benchmark hoặc GGUF.
