"""plots.py — Vẽ biểu đồ thí nghiệm và so sánh theo yêu cầu."""

from __future__ import annotations

import os
from typing import Any
import matplotlib.pyplot as plt


def plot_run(result: dict[str, Any], path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:

    (1) train_loss và val_loss theo epoch (cùng một trục)
    (2) val_acc (và val_macro_f1) theo epoch
    (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    # Đảm bảo thư mục lưu ảnh tồn tại
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Lấy thông tin cấu hình và metrics
    exp_id = result.get("exp_id", "Unknown")
    cfg = result.get("config", {})
    cfg_text = (
        f"opt: {cfg.get('optimizer', 'N/A')} | lr: {cfg.get('lr', 'N/A')} | "
        f"batch: {cfg.get('batch_size', 'N/A')} | init: {cfg.get('init', 'N/A')}"
    )
    fig.suptitle(f"Experiment: {exp_id}\n({cfg_text})", fontsize=13, y=1.03)

    train_loss = result.get("train_loss", [])
    val_loss = result.get("val_loss", [])
    val_acc = result.get("val_acc", [])
    val_macro_f1 = result.get("val_macro_f1", [])
    grad_norm = result.get("grad_norm", [])
    best_epoch = result.get("best_epoch", None)

    epochs = range(1, len(train_loss) + 1)

    # (1) train_loss & val_loss
    ax1 = axes[0]
    if train_loss:
        ax1.plot(epochs, train_loss, label="Train Loss", color="royalblue")
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="crimson")
    if best_epoch is not None:
        ax1.axvline(
            x=best_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best Epoch ({best_epoch})",
        )
    ax1.set_title("Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # (2) val_acc & val_macro_f1
    ax2 = axes[1]
    if val_acc:
        ax2.plot(epochs, val_acc, label="Val Acc", color="forestgreen")
    if val_macro_f1:
        ax2.plot(epochs, val_macro_f1, label="Val Macro F1", color="darkorange")
    if best_epoch is not None:
        ax2.axvline(
            x=best_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best Epoch ({best_epoch})",
        )
    ax2.set_title("Validation Metrics vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # (3) grad_norm (trước khi clip)
    ax3 = axes[2]
    if grad_norm:
        # Nếu grad_norm được ghi nhận theo epoch
        x_grad = range(1, len(grad_norm) + 1)
        ax3.plot(x_grad, grad_norm, label="Grad Norm (L2)", color="purple")
        if best_epoch is not None and len(grad_norm) == len(train_loss):
            ax3.axvline(
                x=best_epoch,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label=f"Best Epoch ({best_epoch})",
            )
    ax3.set_title("Gradient Norm vs. Epoch")
    ax3.set_xlabel("Epoch / Step")
    ax3.set_ylabel("Grad Norm")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compare(
    results: list[dict[str, Any]], metric: str, path: str, title: str = ""
) -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục.

    Mỗi thí nghiệm là một đường kèm nhãn chú thích exp_id.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        exp_id = res.get("exp_id", "Unknown")
        values = res.get(metric, [])
        if values:
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker="o", markersize=3, label=exp_id)

    plot_title = title if title else f"Comparison of {metric}"
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)