"""Portable reference optimizers from the public Flash training description.

The implementations favor inspectability over fused-kernel performance.  They
are useful for checking parameter grouping and update invariants on a toy
model; they are not a replacement for a production Muon/Sinkhorn kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def _newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor for a matrix or a batch of matrices."""

    if matrix.ndim < 2:
        return matrix
    transposed = matrix if matrix.shape[-2] <= matrix.shape[-1] else matrix.transpose(-1, -2)
    norm = transposed.flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    x = transposed / norm
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        x = 1.5 * x - 0.5 * (gram @ x)
    if matrix.shape[-2] > matrix.shape[-1]:
        x = x.transpose(-1, -2)
    return x


class HeadWiseMuon(Optimizer):
    """Muon-style momentum update with optional leading head dimensions."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, momentum: float = 0.95, ns_steps: int = 5):
        if lr <= 0 or not 0 <= momentum < 1 or ns_steps < 1:
            raise ValueError("invalid Muon hyperparameters")
        super().__init__(params, {"lr": lr, "momentum": momentum, "ns_steps": ns_steps})

    @torch.no_grad()
    def step(self, closure: Any | None = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                momentum.mul_(beta).add_(gradient, alpha=1 - beta)
                update = momentum
                if parameter.ndim >= 2:
                    update = _newton_schulz(momentum.float(), group["ns_steps"]).to(parameter.dtype)
                    update = update * (parameter.shape[-1] ** 0.5)
                parameter.add_(update, alpha=-group["lr"])
        return loss


class SinkhornMomentum(Optimizer):
    """Momentum plus alternating row/column RMS normalization."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        gamma: float = 0.18,
        iterations: int = 3,
        threshold: float = 0.01,
    ):
        if lr <= 0 or not 0 <= momentum < 1 or gamma <= 0 or iterations < 1 or threshold < 0:
            raise ValueError("invalid Sinkhorn hyperparameters")
        super().__init__(params, {"lr": lr, "momentum": momentum, "gamma": gamma, "iterations": iterations, "threshold": threshold})

    @torch.no_grad()
    def step(self, closure: Any | None = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                momentum.mul_(beta).add_(gradient, alpha=1 - beta)
                if parameter.ndim < 2:
                    parameter.add_(momentum, alpha=-group["lr"] * group["gamma"])
                    continue
                update = momentum.float()
                flat = update.reshape(-1, update.shape[-1])
                row_norm = flat.norm(dim=-1)
                mean_norm = row_norm.mean().clamp_min(1e-8)
                flat = flat.clone()
                flat[row_norm <= group["threshold"] * mean_norm] = 0
                for iteration in range(group["iterations"]):
                    if iteration % 2 == 0:
                        denom = flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    else:
                        denom = flat.norm(dim=0, keepdim=True).clamp_min(1e-8)
                    flat = flat / denom
                update = flat.reshape_as(update)
                update = update * (parameter.shape[-1] ** 0.5)
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"] * group["gamma"])
        return loss


def build_flash_optimizers(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 1e-3,
) -> dict[str, Optimizer]:
    """Partition a model into Muon, Sinkhorn and AdamW reference groups."""

    muon: list[torch.nn.Parameter] = []
    sinkhorn: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if any(tag in lowered for tag in ("engram", "embedding", "prediction_head")) and parameter.ndim >= 2:
            sinkhorn.append(parameter)
        elif any(tag in lowered for tag in ("q_proj", "k_proj", "query", "key")) and parameter.ndim >= 2:
            muon.append(parameter)
        else:
            adamw.append(parameter)
    optimizers: dict[str, Optimizer] = {}
    if muon:
        optimizers["head_wise_muon"] = HeadWiseMuon(muon, lr=lr)
    if sinkhorn:
        optimizers["sinkhorn_momentum"] = SinkhornMomentum(sinkhorn, lr=lr)
    if adamw:
        optimizers["adamw"] = torch.optim.AdamW(adamw, lr=lr)
    return optimizers


def run_optimizer_smoke() -> dict[str, Any]:
    """Exercise parameter grouping and one finite update on a toy model.

    This is deliberately small: it validates the reference update contracts
    without implying that a production fused optimizer or the full Flash
    pre-training stack is available.
    """

    torch.manual_seed(20260923)
    parameters = [
        ("q_proj.weight", torch.nn.Parameter(torch.randn(4, 4))),
        ("embedding.weight", torch.nn.Parameter(torch.randn(8, 4))),
        ("prediction_head.weight", torch.nn.Parameter(torch.randn(8, 4))),
        ("norm.weight", torch.nn.Parameter(torch.ones(4))),
    ]
    optimizers = build_flash_optimizers(parameters, lr=1e-2)
    before = sum(float(parameter.detach().square().mean()) for _, parameter in parameters)
    loss = sum(parameter.square().mean() for _, parameter in parameters)
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    after = sum(float(parameter.detach().square().mean()) for _, parameter in parameters)
    if not torch.isfinite(torch.tensor(after)):
        raise FloatingPointError("reference optimizer produced a non-finite parameter")
    return {
        "optimizers": sorted(optimizers),
        "loss_before": before,
        "loss_after": after,
        "updated": before != after,
    }

