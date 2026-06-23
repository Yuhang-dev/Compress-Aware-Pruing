from __future__ import annotations

import torch
import torch.nn.functional as F


def refusal_nll(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Causal LM NLL for refusal targets.

    `logits[:, t]` predicts `labels[:, t + 1]`, so this function performs the
    autoregressive shift internally.
    """

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    return F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=ignore_index)


def utility_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str = "student_teacher",
) -> torch.Tensor:
    """Utility-preserving KL.

    Default matches the experiment plan: KL(pi_theta || pi_ref). Use
    `direction="teacher_student"` for standard KD-style KL(pi_ref || pi_theta).
    """

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    if direction == "student_teacher":
        student_probs = F.softmax(student_logits, dim=-1)
        return F.kl_div(teacher_log_probs, student_probs, reduction="batchmean", log_target=False)
    if direction != "teacher_student":
        raise ValueError(f"Unknown KL direction: {direction}")
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def entangle_a1(write_weight: torch.Tensor, refusal_dir: torch.Tensor, input_norm: torch.Tensor) -> torch.Tensor:
    a = write_weight.transpose(0, 1).matmul(refusal_dir)
    denom = a.square().sum().clamp_min(1e-12)
    return -(input_norm.detach() * a.square()).sum() / denom


def entangle_a2(
    write_weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    input_norm: torch.Tensor,
    keep_basis: torch.Tensor,
) -> torch.Tensor:
    a = write_weight.transpose(0, 1).matmul(refusal_dir)
    weighted_a = input_norm.detach() * a
    projected = keep_basis.matmul(keep_basis.transpose(0, 1).matmul(weighted_a))
    residual = weighted_a - projected
    return residual.square().sum() / weighted_a.square().sum().clamp_min(1e-12)
