"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import openpyxl
import torch


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    torch.save(result, Path(results_dir) / f"{result['exp_id']}.pt")
    path = Path(results_dir) / f"{result['exp_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": result["cfg"],
                "history": result["history"],
                "summary": result["summary"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    raise NotImplementedError  # TODO


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    row = {}

    # 1. Định danh thí nghiệm
    exp_id = result.get("exp_id", "")
    row["exp_id"] = exp_id

    # 2. Gộp thông tin cấu hình (cfg)
    cfg = result.get("config", {}) or result.get("cfg", {})
    row.update(cfg)

    # 3. Gộp tóm tắt kết quả huấn luyện (summary hoặc các chỉ số tốt nhất)
    if "summary" in result and isinstance(result["summary"], dict):
        row.update(result["summary"])
    else:
        # Nếu result lưu trực tiếp các trường summary ở cấp ngoài
        for key in ["best_epoch", "best_val_loss", "best_val_acc", "best_val_macro_f1"]:
            if key in result:
                row[key] = result[key]

    # 4. Thêm kết quả đánh giá tập kiểm thử (nếu có)
    if eval_scores:
        row["eval_acc"] = eval_scores.get("eval_acc", eval_scores.get("acc", None))
        row["eval_macro_f1"] = eval_scores.get("eval_macro_f1", eval_scores.get("macro_f1", None))
    else:
        row["eval_acc"] = None
        row["eval_macro_f1"] = None

    # 5. Đường dẫn ảnh biểu đồ và ghi chú
    row["figure_file"] = f"figures/{exp_id}.png"
    row["notes"] = notes

    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    # Các cột công thức trong template Excel cần bỏ qua để không ghi đè mất công thức
    FORMULA_COLS = {
        "step0_gap_vs_lnC",
        "gap_val_minus_train",
        "delta_val_f1_vs_base",
        "beyond_noise",
    }

    # 1. Load template (giữ nguyên công thức)
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]

    # 2. Đọc header dòng 1 để map {tên_cột: chỉ_số_cột_1_based}
    header_to_col: dict[str, int] = {}
    for col_idx in range(1, ws.max_column + 1):
        cell_val = ws.cell(row=1, column=col_idx).value
        if cell_val is not None:
            header_to_col[str(cell_val).strip()] = col_idx

    # 3. Ghi dữ liệu từ dòng 2 trở xuống
    current_row = 2
    for row_data in rows:
        for key, val in row_data.items():
            key_clean = str(key).strip()
            # Bỏ qua các cột công thức tính toán tự động
            if key_clean in FORMULA_COLS:
                continue

            if key_clean in header_to_col:
                col_idx = header_to_col[key_clean]
                # Đổi tuple thành string nếu có (ví dụ cấu hình hidden: (256, 128) -> "(256, 128)")
                if isinstance(val, (tuple, list)):
                    val = str(val)
                ws.cell(row=current_row, column=col_idx, value=val)

        current_row += 1

    # Tạo thư mục chứa file đầu ra nếu chưa tồn tại
    parent_dir = os.path.dirname(out_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # 4. Lưu workbook ra đường dẫn đích
    wb.save(out_path)
