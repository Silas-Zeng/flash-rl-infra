"""Tiny MTP/speculative primitives used by the reference ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Independent future-token heads for a small MTP training experiment."""

    def __init__(self, hidden_size: int, vocab_size: int, future_tokens: int = 2) -> None:
        super().__init__()
        if hidden_size < 1 or vocab_size < 2 or future_tokens < 1:
            raise ValueError("MTP dimensions must be positive")
        self.future_tokens = future_tokens
        self.heads = nn.ModuleList(nn.Linear(hidden_size, vocab_size) for _ in range(future_tokens))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=-2)

    def loss(self, hidden_states: torch.Tensor, future_targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy across future positions.

        ``future_targets`` has shape ``[..., future_tokens]`` and is aligned
        with the final dimension returned by ``forward``.
        """

        logits = self(hidden_states)
        if logits.shape[:-1] != future_targets.shape:
            raise ValueError(f"future target shape mismatch: expected {logits.shape[:-1]}, got {future_targets.shape}")
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), future_targets.reshape(-1))


@dataclass(frozen=True)
class VerificationTrace:
    proposed: int
    accepted: int
    rejected_at: int | None


def verify_candidates(
    target_logprobs: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    min_probability: float = 0.0,
) -> VerificationTrace:
    """Accept a draft prefix while target probability passes a threshold."""

    if target_logprobs.numel() != candidate_ids.numel():
        raise ValueError("target_logprobs and candidate_ids must have the same length")
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be in [0, 1]")
    accepted = 0
    rejected_at: int | None = None
    for index, logprob in enumerate(target_logprobs.flatten()):
        if float(logprob.detach().exp()) < min_probability:
            rejected_at = index
            break
        accepted += 1
    return VerificationTrace(int(candidate_ids.numel()), accepted, rejected_at)


def run_mtp_smoke(*, hidden_size: int = 16, vocab_size: int = 32, future_tokens: int = 2) -> dict[str, float | int]:
    """Run one deterministic MTP loss/verification pass for ablation reports."""

    torch.manual_seed(20260923)
    head = MTPHead(hidden_size, vocab_size, future_tokens)
    hidden = torch.randn(8, hidden_size)
    targets = torch.randint(0, vocab_size, (8, future_tokens))
    loss = head.loss(hidden, targets)
    logits = head(hidden)[0]
    token_ids = logits.argmax(dim=-1)
    logprobs = logits.log_softmax(dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    trace = verify_candidates(logprobs, token_ids, min_probability=0.01)
    return {
        "future_tokens": future_tokens,
        "training_loss": float(loss.item()),
        "proposed_tokens": trace.proposed,
        "accepted_tokens": trace.accepted,
    }

