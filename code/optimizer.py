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
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name}. Expected one of {OPTIMIZERS}")

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu bạn dùng scheduler ở một thí nghiệm, hãy ghi vào bảng (cột notes).
    """
    if name is None:
        return None

    name = name.lower()
    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), **kwargs)
    if name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=max(1, kwargs.get("step_size", 1)), gamma=kwargs.get("gamma", 0.1))
    if name == "linear":
        from torch.optim.lr_scheduler import LambdaLR
        if "warmup_steps" in kwargs:
            warmup_steps = max(0, int(kwargs["warmup_steps"]))
            total = max(1, total_steps)

            def lr_lambda(epoch):
                if epoch < warmup_steps:
                    return (epoch + 1) / max(1, warmup_steps)
                progress = (epoch - warmup_steps) / max(1, total - warmup_steps)
                return max(0.0, 1.0 - progress)

            return LambdaLR(optimizer, lr_lambda)

        return LambdaLR(optimizer, lambda step: 1.0)

    raise ValueError(f"Unsupported scheduler: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    Giá trị trả về chính là `grad_norm` bạn phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0

    if max_norm is None:
        total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), p=2)
        return float(total_norm.item())

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    return float(total_norm)
