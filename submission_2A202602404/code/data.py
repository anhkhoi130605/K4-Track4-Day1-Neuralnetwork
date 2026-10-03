"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)

ROOT = Path(__file__).resolve().parents[2]
def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    training_data=np.load(f"{processed_dir}/train.npz")
    eval_data=np.load(f"{processed_dir}/eval.npz")
    X_train_full = training_data["X"]
    y_train_full = training_data["y"]
    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    eval_row_id = eval_data["row_id"]
    assert X_train_full.shape[1] == 54 and X_eval.shape[1] == 54
    assert y_train_full.dtype == np.int64 and y_eval.dtype == np.int64
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0)
    return mean, std
    #nếu tính trên toàn bộ dữ liệu thì sẽ không có tập dữ liệu để validate, còn nếu tính trên cả tập test thì sẽ bị hiện tương data leakeage


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    raise NotImplementedError  # TODO


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    raise NotImplementedError  # TODO


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    raise NotImplementedError  # TODO
if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    processed_path = ROOT / "data" / "processed"

    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir=processed_path)

    # print("=== 5 dòng đầu của tập train ===")
    # df = pd.DataFrame(X_train_full[:5])
    # df["target"] = y_train_full[:5]
    # print(df)
    #
    # print("\nKích thước X_train_full:", X_train_full.shape)
    # print("Kích thước y_train_full:", y_train_full.shape)

    # classes = np.unique(y_train_full)
    # print("Danh sách các class:", classes)
    # print("Tổng số lượng class:", len(classes))
    #
    # classes, counts = np.unique(y_train_full, return_counts=True)
    # for c, cnt in zip(classes, counts):
    #     print(f"Class {c}: {cnt} mẫu")   7 Class
    # print("1. Kiểu dữ liệu (dtype):", X_train_full.dtype)

    ratio_full = pd.Series(y_train_full).value_counts(normalize=True).sort_index() * 100
    ratio_tr = pd.Series(y_train_full).value_counts(normalize=True).sort_index() * 100
    ratio_val = pd.Series(y_eval).value_counts(normalize=True).sort_index() * 100

    # 2. Tính số lượng mẫu cụ thể ở mỗi tập
    count_tr = pd.Series(y_train_full).value_counts().sort_index()
    count_val = pd.Series(y_eval).value_counts().sort_index()

    # 3. Tạo bảng tổng hợp để xem
    df_compare = pd.DataFrame({
        'Số lượng Train': count_tr,
        'Tỷ lệ Train (%)': ratio_tr.round(2),
        'Số lượng Val': count_val,
        'Tỷ lệ Val (%)': ratio_val.round(2),
        'Tỷ lệ Gốc (%)': ratio_full.round(2)
    })

    print("=== SO SÁNH PHÂN PHỐI CLASS TRƯỚC VÀ SAU KHI STRATIFY ===")
    print(df_compare)
