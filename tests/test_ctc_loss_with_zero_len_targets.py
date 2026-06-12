import torch
import torch.nn.functional as F

# ------------------ global device parameter ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(42)

# Maximum time steps, batch size, number of classes (blank=0)
T = 6  # seq_len
N = 3  # batch_size
C = 4  # number of labels

# Force high probability on blank token (index 0)
logits = torch.ones(T, N, C).to(DEVICE)
logits[:, :, 0] = 100.0  # blank dominates
log_probs = logits.log_softmax(dim=-1)  # shape: (T, N, C)

# Input lengths: each sample uses all T time steps
input_lengths = torch.tensor([T, T, T], dtype=torch.long).to(DEVICE)

# Targets for non-empty sequences (concatenated)
targets = torch.tensor([1, 2, 3], dtype=torch.long).to(DEVICE)

# Target lengths: sample index 1 has length 0 → empty target
target_lengths = torch.tensor([2, 0, 1], dtype=torch.long).to(DEVICE)

# Compute CTC loss per sample
loss = F.ctc_loss(
    log_probs=log_probs,
    targets=targets,
    input_lengths=input_lengths,
    target_lengths=target_lengths,
    blank=0,
    reduction="none",  # per‑sample loss
    zero_infinity=False,
)

print("Loss per sample:", loss)

# ------ Check that the sample with empty target (index 1) has zero loss ------
empty_target_idx = 1
loss_empty = loss[empty_target_idx].item()

# Because we forced the model to output blank with near‑certainty,
# the all‑blank path probability is essentially 1 → loss ≈ 0.
assert torch.allclose(
    loss[empty_target_idx],
    torch.tensor(0.0, device=DEVICE),
    atol=1e-6,
), f"Expected zero loss for empty target, but got {loss_empty}"

print(
    f"\n✓ Loss for empty‑target sample (index {empty_target_idx}) = {loss_empty:.6f} (≈ 0, as expected)"
)
