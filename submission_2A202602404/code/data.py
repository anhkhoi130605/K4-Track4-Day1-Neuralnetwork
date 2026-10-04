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
def load_split(processed_dir: str | Path | None = None):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    if processed_dir is None:
        processed_dir = ROOT / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)
        # Nếu truyền vào là đường dẫn tương đối, ép nó đi từ ROOT
        if not processed_dir.is_absolute():
            processed_dir = ROOT / processed_dir

    train_path = processed_dir / "train.npz"
    eval_path = processed_dir / "eval.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {train_path}. Bạn đã chạy scripts/split_data.py chưa?")

    training_data = np.load(train_path)
    eval_data = np.load(eval_path)

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
    #tạo bản sao
    X_scaled = X.copy().astype(float)
    eps = 1e-8
    X_scaled[:, :10] = (X_scaled[:, :10] - mean) / (std + eps) # công thức Z-score Standardization (chuẩn hóa dữ liệu theo phân phối chuẩn), dùng trong bước tiền xử lý đặc trưng (feature scaling) để đưa dữ liệu về phân phối có trung bình bằng 0 và độ lệch chuẩn bằng 1.
    #công thức handle được từ âm vô cực đến dưong vô cực, tránh chia cho 0
    return X_scaled


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
    processed_path = ROOT / "data" / "processed"
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=0.2, seed=42)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)

    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device)
    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if shuffle:
        perm = torch.randperm(len(X), generator=generator, device=X.device) #tensor 1D hoán vị ngẫu nhiên từ 0 đến len(X)-1
    else :
        perm = torch.arange(len(X), device=X.device) #tensor 1D tuần tự từ 0 đến len(X)-1 trên tập dữ liệu ban đầu
    for i in range(0, len(X), batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
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
    #
    # classes = np.unique(y_train_full)
    # print("Danh sách các class:", classes)
    # print("Tổng số lượng class:", len(classes))
    # Kích thước X_train_full: (464809, 54)
    # Kích thước y_train_full: (464809,)
    # Danh  sách  các  class: [0 1 2 3 4 5 6]
    # Tổng số lượng class: 7

    #
    # classes, counts = np.unique(y_train_full, return_counts=True)
    # for c, cnt in zip(classes, counts):
    #     print(f"Class {c}: {cnt} mẫu")   7 Class
    # print("1. Kiểu dữ liệu (dtype):", X_train_full.dtype)

    # ratio_full = pd.Series(y_train_full).value_counts(normalize=True).sort_index() * 100
    # ratio_tr = pd.Series(y_train_full).value_counts(normalize=True).sort_index() * 100
    # ratio_val = pd.Series(y_eval).value_counts(normalize=True).sort_index() * 100
    #
    # # 2. Tính số lượng mẫu cụ thể ở mỗi tập
    # count_tr = pd.Series(y_train_full).value_counts().sort_index()
    # count_val = pd.Series(y_eval).value_counts().sort_index()
    #
    # # 3. Tạo bảng tổng hợp để xem
    # df_compare = pd.DataFrame({
    #     'Số lượng Train': count_tr,
    #     'Tỷ lệ Train (%)': ratio_tr.round(2),
    #     'Số lượng Val': count_val,
    #     'Tỷ lệ Val (%)': ratio_val.round(2),
    #     'Tỷ lệ Gốc (%)': ratio_full.round(2)
    # })
    #
    # print("=== SO SÁNH PHÂN PHỐI CLASS TRƯỚC VÀ SAU KHI STRATIFY ===")
    # print(df_compare)

#    Số lượng Train  Tỷ lệ Train (%)  Số lượng Val  Tỷ lệ Val (%)  Tỷ lệ Gốc (%)
# 0          169472            36.46         42368          36.46          36.46
# 1          226640            48.76         56661          48.76          48.76
# 2           28603             6.15          7151           6.15           6.15
# 3            2198             0.47           549           0.47           0.47
# 4            7594             1.63          1899           1.63           1.63
# 5           13894             2.99          3473           2.99           2.99
# 6           16408             3.53          4102           3.53           3.53
#
# Process finished with exit code 0


# device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Gọi hàm chuẩn bị dữ liệu
# data = prepare_data(device=device)
# X_tr = data["X_tr"].cpu().numpy()
# y_tr = data["y_tr"].cpu().numpy()

# Cấu hình numpy in số thực gọn gàng (làm tròn 4 chữ số thập phân)
# np.set_printoptions(precision=4, suppress=True, linewidth=120)

# print("=" * 60)
# print("1. XEM TRƯỚC 5 DÒNG ĐẦU CỦA 10 CỘT ĐÃ CHUẨN HOÁ (cột 0 -> 9):")
# print(X_tr[:5, :N_NUMERIC])
# [[ 0.877  -1.3648 -0.148   1.7651  2.4137  1.373  -0.3794 -0.4203  0.1697  0.2057]
#  [ 1.2233 -0.8735 -0.415  -0.671  -0.8487  0.2485  0.5922 -0.2685 -0.5357  0.4341]
#  [ 1.0412 -1.1325  1.1873  1.5719  2.5854  0.8615 -0.2673 -1.9884 -0.9799 -0.0529]
#  [-0.0298 -0.9807 -0.0144 -0.3741 -0.9689 -0.9756  0.4427 -0.7238 -0.6925 -0.3462]
#  [ 1.4303 -1.0968 -0.415  -0.5626  0.2502  1.149   0.2185 -0.3697 -0.2483  1.445 ]]

# print("\n" + "=" * 60)
# print("2. XEM TRƯỚC 5 DÒNG ĐẦU CỦA CÁC CỘT NHỊ PHÂN (cột 10 -> 19 mẫu):")
# print(X_train_full[:5, N_NUMERIC:N_NUMERIC + 10])
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# print("\n" + "=" * 60)
# print("3. KIỂM TRA THỐNG KÊ CỦA 10 CỘT ĐẦU TRÊN TẬP TRAIN:")
# computed_mean = np.mean(X_train_full[:, :N_NUMERIC], axis=0)
# computed_std = np.std(X_train_full[:, :N_NUMERIC], axis=0)
# print("Mean thực tế sau chuẩn hoá (kỳ vọng xấp xỉ 0.0):")
# print(computed_mean)
# print("Std thực tế sau chuẩn hoá  (kỳ vọng xấp xỉ 1.0):")
# print(computed_std)
# [ 0.0001 -0.     -0.     -0.0003 -0.     -0.      0.     -0.      0.     -0.    ]
# Std thực tế sau chuẩn hoá  (kỳ vọng xấp xỉ 1.0):
# [1.     0.9999 1.0008 0.9999 0.9998 1.     0.9997 1.     0.9997 1.    ]

# print("\n" + "=" * 60)
# print("4. KIỂM TRA 44 CỘT NHỊ PHÂN:")
# binary_unique_vals = np.unique(X_train_full[:, N_NUMERIC:])
# print(f"Các giá trị xuất hiện trong 44 cột nhị phân: {binary_unique_vals}")
# print("Kết luận:",
#       "HỢP LỆ (chỉ gồm 0 và 1)" if np.array_equal(binary_unique_vals, [0., 1.]) else "CẢNH BÁO (bị đổi giá trị)")
# Các giá trị xuất hiện trong 44 cột nhị phân: [0. 1.]
# Kết luận: HỢP LỆ (chỉ gồm 0 và 1)








