"""Small MTP and speculative-decoding reference primitives.

The verifier in this module operates on probability rows produced by a draft
and a target model.  It implements the rejection-sampling step used by
speculative decoding: a proposed token is accepted with ``min(1, p / q)``;
after the first rejection a token is drawn from the residual distribution
``max(p - q, 0)``.  The implementation is deliberately model-agnostic: an
SGLang adapter can provide the rows without importing any runtime internals.
"""

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
    emitted: tuple[int, ...] = ()
    correction_token: int | None = None
    bonus_token: int | None = None
    acceptance_probs: tuple[float, ...] = ()


def _normalise_rows(probabilities: torch.Tensor, name: str) -> torch.Tensor:
    """Return finite, row-normalised probabilities with a clear error."""

    if probabilities.ndim != 2:
        raise ValueError(f"{name} must have shape [draft_tokens, vocab]")
    if not torch.is_floating_point(probabilities):
        probabilities = probabilities.float()
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    totals = probabilities.sum(dim=-1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError(f"{name} rows must have positive mass")
    return probabilities / totals


def _draw(probabilities: torch.Tensor, generator: torch.Generator | None) -> int:
    """Sample one categorical token, keeping generator/device handling local."""

    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def verify_speculative_tokens(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    target_next_probs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> VerificationTrace:
    """Verify a draft block with standard speculative sampling.

    ``target_probs[i]`` and ``draft_probs[i]`` are the target and draft
    distributions for candidate ``candidate_ids[i]``.  On rejection, the
    emitted correction is sampled from ``normalize(max(target - draft, 0))``;
    this is the residual distribution from speculative sampling.  If all
    candidates are accepted and ``target_next_probs`` is supplied, one bonus
    token is sampled from the target distribution at the next position.

    The returned ``emitted`` tuple contains accepted candidates followed by a
    correction or bonus token.  Inputs are copied only through normalisation;
    no model forward pass is performed here.
    """

    target = _normalise_rows(target_probs, "target_probs")
    draft = _normalise_rows(draft_probs, "draft_probs")
    if draft.device != target.device:
        draft = draft.to(device=target.device)
    if target.shape != draft.shape:
        raise ValueError("target_probs and draft_probs must have the same shape")
    if candidate_ids.ndim != 1 or candidate_ids.numel() != target.shape[0]:
        raise ValueError("candidate_ids must have one entry per probability row")
    if candidate_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("candidate_ids must be an integer tensor")
    candidate_ids = candidate_ids.to(device=target.device, dtype=torch.long)
    if (candidate_ids < 0).any() or (candidate_ids >= target.shape[1]).any():
        raise ValueError("candidate_ids contain an out-of-range vocabulary index")
    if target_next_probs is not None:
        if target_next_probs.ndim != 1 or target_next_probs.numel() != target.shape[1]:
            raise ValueError("target_next_probs must have shape [vocab]")
        target_next = _normalise_rows(
            target_next_probs.to(device=target.device).unsqueeze(0), "target_next_probs"
        )[0]
    else:
        target_next = None

    emitted: list[int] = []
    acceptance_probs: list[float] = []
    correction_token: int | None = None
    rejected_at: int | None = None
    for index in range(target.shape[0]):
        token = int(candidate_ids[index].item())
        p = target[index, token]
        q = draft[index, token]
        # q == 0 cannot produce this candidate in a valid draft sample.  Treat
        # it as accepted if p has mass, which avoids a NaN and is the limiting
        # form of min(1, p/q).
        ratio = torch.where(q > 0, p / q, torch.ones_like(p))
        acceptance = float(torch.clamp(ratio, max=1.0).item())
        acceptance_probs.append(acceptance)
        draw = torch.rand((), device=target.device, generator=generator)
        # Use a strict comparison so an acceptance probability of zero can
        # never accept the candidate (torch.rand may return exactly zero).
        if float(draw.item()) < acceptance:
            emitted.append(token)
            continue

        rejected_at = index
        residual = torch.clamp(target[index] - draft[index], min=0)
        if float(residual.sum().item()) <= 1e-12:
            residual = target[index]
        residual = residual / residual.sum()
        correction_token = _draw(residual, generator)
        emitted.append(correction_token)
        break

    bonus_token: int | None = None
    if rejected_at is None and target_next is not None:
        bonus_token = _draw(target_next, generator)
        emitted.append(bonus_token)
    return VerificationTrace(
        proposed=int(candidate_ids.numel()),
        accepted=int(rejected_at if rejected_at is not None else candidate_ids.numel()),
        rejected_at=rejected_at,
        emitted=tuple(emitted),
        correction_token=correction_token,
        bonus_token=bonus_token,
        acceptance_probs=tuple(acceptance_probs),
    )


def speculative_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    target_next_probs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, VerificationTrace]:
    """Sample a draft block from ``q`` and run target-model verification.

    This convenience wrapper covers the complete algorithm-level operation:
    candidate IDs are sampled independently from each draft row, then passed
    to :func:`verify_speculative_tokens`.  A serving adapter remains
    responsible for producing the probability rows with the appropriate KV
    cache and target-model forward passes.
    """

    draft = _normalise_rows(draft_probs, "draft_probs")
    if target_probs.device != draft.device:
        draft = draft.to(device=target_probs.device)
    if target_probs.ndim != 2 or target_probs.shape != draft.shape:
        raise ValueError("target_probs and draft_probs must have the same shape")
    candidates = torch.multinomial(draft, 1, generator=generator).squeeze(-1)
    trace = verify_speculative_tokens(
        target_probs,
        draft,
        candidates,
        target_next_probs=target_next_probs,
        generator=generator,
    )
    return candidates, trace


def verify_candidates(
    target_logprobs: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    min_probability: float = 0.0,
    draft_logprobs: torch.Tensor | None = None,
    target_next_logprobs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> VerificationTrace:
    """Verify candidates, with compatibility for the original threshold API.

    Supplying ``draft_logprobs`` selects standard speculative sampling.  The
    old one-dimensional threshold behavior remains available when it is not
    supplied so existing ablation callers continue to work.
    """

    if draft_logprobs is not None:
        target_probs = target_logprobs.exp() if target_logprobs.ndim == 2 else target_logprobs
        draft_probs = draft_logprobs.exp() if draft_logprobs.ndim == 2 else draft_logprobs
        target_next_probs = (
            target_next_logprobs.exp() if target_next_logprobs is not None else None
        )
        return verify_speculative_tokens(
            target_probs,
            draft_probs,
            candidate_ids,
            target_next_probs=target_next_probs,
            generator=generator,
        )

    if target_logprobs.ndim != 1:
        raise ValueError("threshold verification expects one log-probability per candidate")

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


def run_mtp_smoke(*, hidden_size: int = 16, vocab_size: int = 32, future_tokens: int = 2) -> dict[str, float | int | bool]:
    """Run one deterministic MTP loss, update, and verification pass."""

    torch.manual_seed(20260923)
    head = MTPHead(hidden_size, vocab_size, future_tokens)
    hidden = torch.randn(8, hidden_size)
    targets = torch.randint(0, vocab_size, (8, future_tokens))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    loss_before = head.loss(hidden, targets)
    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    optimizer.step()
    loss_after = head.loss(hidden, targets)
    logits = head(hidden)[0]
    token_ids = logits.argmax(dim=-1)
    logprobs = logits.log_softmax(dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    trace = verify_candidates(logprobs, token_ids, min_probability=0.01)
    return {
        "future_tokens": future_tokens,
        "training_loss": float(loss_after.item()),
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "optimized": bool(loss_after.item() < loss_before.item()),
        "optimization_steps": 1,
        "proposed_tokens": trace.proposed,
        "accepted_tokens": trace.accepted,
    }

