"""Plot HF schedulers: Linear vs Cosine."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import get_scheduler

LR = 1e-4
MAX_STEPS = 53_220
WARMUP_STEPS = 0


def get_lrs(scheduler_type: str) -> list[float]:
    dummy_param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([dummy_param], lr=LR)
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )
    lrs = []
    for _ in range(MAX_STEPS):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


steps = np.arange(1, MAX_STEPS + 1)
linear_lrs = get_lrs("linear")
cosine_lrs = get_lrs("cosine")
inv_sqrt_lrs = get_lrs("inverse_sqrt")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, linear_lrs, label="Linear", linewidth=1.5)
ax.plot(steps, cosine_lrs, label="Cosine", linewidth=1.5)
ax.plot(steps, inv_sqrt_lrs, label="Inverse Sqrt", linewidth=1.5)

# Mark LR values at 50k steps
mark_step = 50_000
mark_idx = mark_step - 1
linear_lr_50k = linear_lrs[mark_idx]
cosine_lr_50k = cosine_lrs[mark_idx]

inv_sqrt_lr_50k = inv_sqrt_lrs[mark_idx]

ax.axvline(x=mark_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.plot(mark_step, linear_lr_50k, "o", color="C0", markersize=6)
ax.plot(mark_step, cosine_lr_50k, "o", color="C1", markersize=6)
ax.plot(mark_step, inv_sqrt_lr_50k, "o", color="C2", markersize=6)
ax.annotate(
    f"Linear @50k: {linear_lr_50k:.2e}",
    xy=(mark_step, linear_lr_50k),
    xytext=(mark_step + 3000, linear_lr_50k + 3e-5),
    arrowprops=dict(arrowstyle="->", color="C0"),
    fontsize=9,
    color="C0",
)
ax.annotate(
    f"Cosine @50k: {cosine_lr_50k:.2e}",
    xy=(mark_step, cosine_lr_50k),
    xytext=(mark_step + 3000, cosine_lr_50k + 3e-5),
    arrowprops=dict(arrowstyle="->", color="C1"),
    fontsize=9,
    color="C1",
)
ax.annotate(
    f"Inverse Sqrt @50k: {inv_sqrt_lr_50k:.2e}",
    xy=(mark_step, inv_sqrt_lr_50k),
    xytext=(mark_step + 3000, inv_sqrt_lr_50k + 3e-5),
    arrowprops=dict(arrowstyle="->", color="C2"),
    fontsize=9,
    color="C2",
)

ax.set_xlabel("Step")
ax.set_ylabel("Learning Rate")
ax.set_title(f"LR Schedulers (initial_lr={LR}, max_steps={MAX_STEPS:,})")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("tests/scheduler_plot.png", dpi=150)
print("Saved tests/scheduler_plot.png")
