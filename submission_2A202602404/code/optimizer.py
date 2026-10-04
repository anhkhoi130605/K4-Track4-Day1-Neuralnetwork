"""optimizer.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Các bước:
      1. kiểm tra name nằm trong OPTIMIZERS, nếu không raise ValueError
      2. "sgd"          -> torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
         "sgd_momentum" -> torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
         "adam"         -> torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
         "adamw"        -> torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """
    optimizer =["sgd", "sgd_momentum", "adam", "adamw"]
    if name not in optimizer:
        raise ValueError(f"Optimizer name must be one of {OPTIMIZERS}, got {name}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu bạn dùng scheduler ở một thí nghiệm, hãy ghi vào bảng (cột notes).
    """
    if name is None:
        return None

    name = name.lower()

    if name == "cosine":
        # Giảm LR dần theo đường cong cosine từ lr ban đầu về eta_min (mặc định 0) sau total_steps
        eta_min = kwargs.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )

    elif name == "step":
        # Giảm LR theo cấp số nhân sau mỗi step_size bước
        step_size = kwargs.get("step_size", max(1, total_steps // 3))
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif name == "linear":
        # Giảm LR tuyến tính từ giá trị ban đầu về 0
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )

    else:
        raise ValueError(f"Scheduler không được hỗ trợ: '{name}'. Chọn 'cosine', 'step', 'linear' hoặc None.")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt (ví dụ clip_grad_norm_ với max_norm=inf)
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    Giá trị trả về chính là `grad_norm` bạn phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        return float(total_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    return float(total_norm)
