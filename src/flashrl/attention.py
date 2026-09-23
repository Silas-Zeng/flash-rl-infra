"""Small reference versions of CED/CSA2 control flow.

The implementation is intentionally dense-tensor PyTorch code.  It exposes
the state and mode transitions that a production sparse kernel must preserve:
Full creates global KV and indices, Reindex reuses KV but refreshes indices,
and Reuse shares both.  It is suitable for correctness tests, not a speed
claim for the DeepSeek kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


class CSA2Mode(str, Enum):
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass
class CSA2State:
    global_k: torch.Tensor | None = None
    global_v: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None


class HierarchicalSparseIndexer:
    """Block-first then token-level top-k selection."""

    def __init__(self, block_size: int = 4, top_k: int = 8) -> None:
        if block_size < 1 or top_k < 1:
            raise ValueError("block_size and top_k must be positive")
        self.block_size = block_size
        self.top_k = top_k

    def select(self, query: torch.Tensor, indexer_k: torch.Tensor) -> torch.Tensor:
        if query.shape[-1] != indexer_k.shape[-1]:
            raise ValueError("query and indexer_k dimensions must match")
        original_tokens = indexer_k.shape[0]
        blocks = math.ceil(original_tokens / self.block_size)
        padded = blocks * self.block_size
        if padded != indexer_k.shape[0]:
            indexer_k = torch.nn.functional.pad(indexer_k, (0, 0, 0, padded - indexer_k.shape[0]))
        block_keys = indexer_k.reshape(blocks, self.block_size, -1).mean(dim=1)
        block_scores = query @ block_keys.transpose(0, 1)
        candidate_blocks = min(blocks, math.ceil(self.top_k / self.block_size) + 1)
        block_ids = block_scores.topk(candidate_blocks, dim=-1).indices
        candidates = (block_ids.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=query.device)).reshape(query.shape[0], -1)
        candidates = candidates.clamp_max(original_tokens - 1)
        token_scores = torch.einsum("qd,qkd->qk", query, indexer_k[candidates])
        keep = min(self.top_k, candidates.shape[-1])
        selected = token_scores.topk(keep, dim=-1).indices
        return candidates.gather(-1, selected)


class CSA2Reference:
    """Dense reference for Full/Reindex/Reuse cache and index semantics."""

    def __init__(self, modes: list[CSA2Mode] | None = None, *, top_k: int = 8, block_size: int = 4) -> None:
        self.modes = modes or [CSA2Mode.FULL, CSA2Mode.REINDEX, CSA2Mode.REUSE]
        self.indexer = HierarchicalSparseIndexer(block_size=block_size, top_k=top_k)
        self.state = CSA2State()

    def run_layer(
        self,
        layer: int,
        query: torch.Tensor,
        global_k: torch.Tensor,
        global_v: torch.Tensor,
        *,
        indexer_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CSA2Mode]:
        mode = self.modes[layer % len(self.modes)]
        if mode is CSA2Mode.FULL:
            self.state.global_k = global_k
            self.state.global_v = global_v
            self.state.indexer_k = indexer_k if indexer_k is not None else global_k
            self.state.topk_indices = self.indexer.select(query, self.state.indexer_k)
        elif mode is CSA2Mode.REINDEX:
            if self.state.global_k is None or self.state.global_v is None:
                raise RuntimeError("Reindex requires a preceding Full layer")
            if indexer_k is not None:
                self.state.indexer_k = indexer_k
            self.state.topk_indices = self.indexer.select(query, self.state.indexer_k)
        elif mode is CSA2Mode.REUSE:
            if self.state.global_k is None or self.state.global_v is None or self.state.topk_indices is None:
                raise RuntimeError("Reuse requires a preceding Full/Reindex layer")
        selected_k = self.state.global_k[self.state.topk_indices]
        selected_v = self.state.global_v[self.state.topk_indices]
        scores = torch.einsum("qd,qkd->qk", query, selected_k) / math.sqrt(query.shape[-1])
        weights = scores.softmax(dim=-1)
        output = torch.einsum("qk,qkd->qd", weights, selected_v)
        return output, mode


@dataclass(frozen=True)
class CEDPlan:
    encoder_layers: int
    decoder_layers: int
    swa_window: int

    @property
    def total_layers(self) -> int:
        return self.encoder_layers + self.decoder_layers

    @property
    def ideal_prefill_layer_fraction(self) -> float:
        return self.encoder_layers / self.total_layers


class CEDReference:
    """Reference planner for encoder-produced decoder global KV."""

    def __init__(self, plan: CEDPlan) -> None:
        if plan.encoder_layers < 1 or plan.decoder_layers < 1 or plan.swa_window < 1:
            raise ValueError("CED plan dimensions must be positive")
        self.plan = plan

    def project_decoder_global_kv(
        self,
        encoder_hidden: torch.Tensor,
        key_projections: list[torch.Tensor],
        value_projections: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(key_projections) != self.plan.decoder_layers or len(value_projections) != self.plan.decoder_layers:
            raise ValueError("one key/value projection is required per decoder layer")
        return (
            [encoder_hidden @ projection for projection in key_projections],
            [encoder_hidden @ projection for projection in value_projections],
        )

    def bounded_replay_tokens(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        return prompt_tokens[..., -self.plan.swa_window :]


def run_attention_smoke() -> dict[str, int | float | list[str]]:
    """Exercise the mode state machine for the combined ablation report."""

    torch.manual_seed(20260923)
    query = torch.randn(3, 8)
    global_k = torch.randn(12, 8)
    global_v = torch.randn(12, 8)
    reference = CSA2Reference(top_k=4, block_size=4)
    modes: list[str] = []
    for layer in range(3):
        output, mode = reference.run_layer(layer, query, global_k, global_v)
        if output.shape != query.shape:
            raise RuntimeError("CSA2 reference returned an invalid attention shape")
        modes.append(mode.value)
    plan = CEDPlan(encoder_layers=2, decoder_layers=2, swa_window=4)
    replay = CEDReference(plan).bounded_replay_tokens(torch.arange(16))
    return {
        "modes": modes,
        "encoder_layers": plan.encoder_layers,
        "decoder_layers": plan.decoder_layers,
        "prefill_layer_fraction": plan.ideal_prefill_layer_fraction,
        "bounded_replay_tokens": int(replay.numel()),
    }

