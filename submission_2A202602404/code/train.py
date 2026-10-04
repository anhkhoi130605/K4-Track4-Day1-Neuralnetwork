"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
Tích hợp:
  - TensorBoard: theo dõi loss, metrics, và lưu ma trận trọng số (histograms).
  - tqdm: xem tiến trình huấn luyện trực quan theo epoch và mini-batch.
"""
from __future__ import annotations

import os
import sys
import random
import time
import shutil
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import iterate_batches, prepare_data
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="adamw",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # TODO: chọn bằng val, không dùng eval (chọn 0.05 làm mặc định)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    pin_memory=True,           # Tối ưu hoá GPU: Pinned memory để nạp lô siêu tốc sang GPU
    seed=1,
)


def get_args():
    parser = ArgumentParser(description="Neural Network training")
    parser.add_argument("--root", "-r", type=str, default="./data", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="Logging tool")
    parser.add_argument("--log_dir", type=str, default="tensorboard", help="TensorBoard log directory")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models", help="Folder to save checkpoints")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (CPU & all GPUs)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Đảm bảo tính tất định cho các thuật toán tích chập / nhân ma trận (cuDNN)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    model_device = next(model.parameters()).device
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        if xb.device != model_device:
            xb = xb.to(model_device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    if len(preds) == 0:
        return torch.empty((0,), dtype=torch.int64, device=model_device)
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    model_device = next(model.parameters()).device
    n = len(X)
    if n == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}

    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        if xb.device != model_device:
            xb = xb.to(model_device, non_blocking=True)
            yb = yb.to(model_device, non_blocking=True)
        logits = model(xb)
        batch_loss = compute_loss(logits, yb, loss_name=loss_name, reduction="sum")
        total_loss += batch_loss.item()
        all_preds.append(logits.argmax(dim=1))

    preds = torch.cat(all_preds, dim=0)
    loss = total_loss / n
    acc = (preds == y).float().mean().item()

    # Dựng ma trận nhầm lẫn 7x7
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y.cpu().numpy(), preds.cpu().numpy()), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
    }


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        num_classes = logits.shape[-1]
        y_one_hot = F.one_hot(y, num_classes=num_classes).float()
        return F.mse_loss(logits, y_one_hot, reduction=reduction)
    else:
        raise ValueError(f"Hàm mất mát không hỗ trợ: '{loss_name}'. Chọn 'ce' hoặc 'mse'.")


def run_experiment(cfg: dict, data: dict, log_dir: str | None = None, writer: SummaryWriter | None = None,
                   save_dir: str = "trained_models") -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    # 0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg.get("lr") is None:
        cfg["lr"] = 0.05

    exp_id = str(cfg.get("exp_id", "exp"))
    seed = int(cfg.get("seed", 1))
    set_seed(seed)

    device = data["X_tr"].device
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # Tối ưu hoá GPU: Kích hoạt Pinned Memory và Tensor Cores (TF32)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        if cfg.get("pin_memory", True):
            for k in ["X_tr", "y_tr", "X_val", "y_val", "X_eval", "y_eval"]:
                if k in data and isinstance(data[k], torch.Tensor):
                    if data[k].device.type == "cpu" and not data[k].is_pinned():
                        data[k] = data[k].pin_memory()

    # Generator cho việc xáo trộn lô
    generator = torch.Generator(device=device).manual_seed(seed)

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)

    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], (
            f"Số tham số {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]} cho kiến trúc {hidden}"
        )

    # chuyển model lên device; tạo optimizer = build_optimizer(...)
    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    # nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
    precision = str(cfg.get("precision", "fp32"))
    use_amp = precision in ("fp16", "bf16") and (device_type == "cuda")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and device_type == "cuda") else None

    # Khởi tạo TensorBoard SummaryWriter
    close_writer_at_end = False
    if writer is None:
        if log_dir is None:
            log_dir = os.path.join("runs", exp_id)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        close_writer_at_end = True

    # 1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    # Ghi log TensorBoard bước 0: metrics và trọng số ban đầu
    if writer is not None:
        writer.add_scalar("Loss/val", step0_loss, 0)
        writer.add_scalar("Metrics/val_acc", step0_res["acc"], 0)
        writer.add_scalar("Metrics/val_macro_f1", step0_res["macro_f1"], 0)
        # Lưu ma trận trọng số (Weights & Biases) vào TensorBoard
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param.data.detach().cpu(), 0)

    # Cấu trúc lưu lịch sử
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_state = None
    diverged = False

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    total_train = len(data["X_tr"])
    num_batches = (total_train + batch_size - 1) // batch_size
    clip_norm = cfg.get("clip_norm")

    # 2. for epoch in 1..epochs:
    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Exp [{exp_id}]", unit="epoch", colour="cyan")

    for epoch in epoch_pbar:
        t0 = time.time()
        model.train()
        batch_grad_norms = []

        batch_pbar = tqdm(
            iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, generator=generator, shuffle=True),
            total=num_batches,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            colour="cyan",
        )
        #forward
        for xb, yb in batch_pbar:
            # Truyền batch sang GPU bất đồng bộ với non_blocking=True (tận dụng DMA từ Pinned Memory)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

            # with torch.autocast(...) nếu precision != "fp32": chỉ bọc forward + loss
            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name=cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name=cfg["loss"])

            # nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            optimizer.zero_grad(set_to_none=True)

            # backward (qua scaler nếu fp16)
            if scaler is not None:
                scaler.scale(loss).backward()
                # nếu fp16 và có clip: scaler.unscale_(optimizer) TRƯỚC khi clip
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                # gn = clip_gradients(model.parameters(), cfg["clip_norm"]) chuẩn TRƯỚC khi cắt; ghi lại
                gn = clip_gradients(model.parameters(), clip_norm)
                # bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # gn = clip_gradients(model.parameters(), cfg["clip_norm"]) chuẩn TRƯỚC khi cắt; ghi lại
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            batch_grad_norms.append(gn)
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "gn": f"{gn:.4f}"})

        if diverged:
            print(f"\n[Cảnh báo] Loss là NaN/Inf ở epoch {epoch}! Huấn luyện phân kỳ.")
            break

        # thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        # cuối epoch (dùng evaluate, chế độ eval):
        # train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
        mean_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        eval_tr_size = min(50000, total_train)
        train_eval = evaluate(model, data["X_tr"][:eval_tr_size], data["y_tr"][:eval_tr_size], loss_name=cfg["loss"])
        train_loss = train_eval["loss"]

        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        val_loss = val_eval["loss"]
        val_acc = val_eval["acc"]
        val_macro_f1 = val_eval["macro_f1"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        # nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_pbar.set_postfix({
            "tr_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "val_f1": f"{val_macro_f1:.4f}",
            "best_f1": f"{best_val_macro_f1:.4f}",
        })

        # Ghi log TensorBoard: Scalars và LƯU TRỌNG SỐ (weights/gradients histogram)
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/val_acc", val_acc, epoch)
            writer.add_scalar("Metrics/val_macro_f1", val_macro_f1, epoch)
            writer.add_scalar("Gradients/norm_mean", mean_gn, epoch)
            writer.add_scalar("Time/epoch_s", epoch_time, epoch)
            writer.add_scalar("Optimizer/lr", optimizer.param_groups[0]["lr"], epoch)

            # LƯU TRỌNG SỐ MÔ HÌNH VÀ GRADIENTS VÀO TENSORBOARD
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data.detach().cpu(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), epoch)

    # 3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    peak_mem_MB = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    time_per_epoch_s = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Lưu checkpoint trọng số tốt nhất ra thư mục save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{exp_id}_best.pt")
        torch.save(best_state, ckpt_path)

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch_s),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    if writer is not None:
        writer.flush()
        if close_writer_at_end:
            writer.close()

    result = {
        "cfg": cfg,
        "config": cfg,
        "exp_id": exp_id,
        "history": history,
        "summary": summary,
        "best_state": best_state,
        # Các trường trực tiếp giúp plots.py và code khác dễ dàng truy cập
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "val_macro_f1": history["val_macro_f1"],
        "grad_norm": history["grad_norm"],
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
    }
    return result


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if isinstance(row_id, torch.Tensor):
        row_id = row_id.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    row_id_arr = np.asarray(row_id, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)

    df = pd.DataFrame({"row_id": row_id_arr, "pred": preds_arr})
    df.to_csv(path, index=False)
    print(f"Đã ghi thành công {len(df)} dòng dự đoán vào: {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    # 1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    # 2. preds = predict(model, data["X_eval"])  # fp32, eval mode
    preds = predict(model, data["X_eval"])

    # 3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)

    # 4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    repo_root = Path(__file__).resolve().parents[2]
    eval_script = repo_root / "scripts" / "evaluate.py"
    data_csv = repo_root / "data" / "covtype.csv.gz"
    meta_csv = repo_root / "data" / "split_metadata.csv"
    pred_path_abs = os.path.abspath(pred_path)

    if eval_script.exists():
        import subprocess
        import sys
        print(f"\n--- Đang thực thi chấm điểm qua {eval_script} ---")
        cmd = [
            sys.executable,
            str(eval_script),
            "--pred", str(pred_path_abs),
            "--data", str(data_csv),
            "--meta", str(meta_csv),
        ]
        eval_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            res = subprocess.run(cmd, cwd=str(repo_root), env=eval_env, capture_output=True, text=True, encoding="utf-8")
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
        except Exception as e:
            print(f"Không thể chạy lệnh chấm điểm tự động: {e}")
    else:
        print(f"Vui lòng chạy lệnh chấm điểm: python scripts/evaluate.py --pred {pred_path}")


if __name__ == "__main__":
    args = get_args()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Khởi chạy huấn luyện trên thiết bị: {device_str}")

    # Chuẩn bị dữ liệu
    print("Đang nạp và chuẩn hóa dữ liệu...")
    data = prepare_data(device=device_str)

    cfg = {
        **DEFAULT_CFG,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "lr": args.lr,
    }

    print(f"\nBắt đầu huấn luyện cấu hình baseline: {cfg['exp_id']}")
    result = run_experiment(cfg, data, log_dir=args.log_dir, save_dir=args.trained_models)

    print("\n=== KẾT QUẢ HUẤN LUYỆN (SUMMARY) ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    # Dự đoán trên tập eval và ghi file
    pred_path = os.path.join(args.trained_models, "predictions_eval.csv")
    print(f"\nThực hiện dự đoán tập eval và lưu tại {pred_path}...")
    final_eval(cfg, result, data, pred_path)
