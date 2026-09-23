"""Runnable KV/communication compression primitives for the Flash prototype.

These are reference PyTorch implementations, not vendor kernels.  They make
the memory/data contracts observable on a tiny tensor: symmetric INT8,
packed FP4-style codes, cross-layer KV references, tiered persistent/temporary
cache, sharded Engram metadata, and bounded approximate replay.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import pickle
import random
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


# A software FP4 E2M1-like codebook.  Hardware FP4 encodings and kernels may
# use a different codebook; the packed four-bit representation is faithful to
# the storage trade-off while keeping this implementation portable.
FP4_CODEBOOK = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


@dataclass
class PackedTensor:
    payload: torch.Tensor
    scale: torch.Tensor | None
    shape: tuple[int, ...]
    bits: int
    group_size: int
    kind: str
    original_dtype: torch.dtype
    padded_last_dim: int

    @property
    def compressed_nbytes(self) -> int:
        payload_bytes = self.payload.numel() * self.payload.element_size()
        scale_bytes = 0 if self.scale is None else self.scale.numel() * self.scale.element_size()
        return payload_bytes + scale_bytes


def _group_view(tensor: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    if group_size < 1:
        raise ValueError("group_size must be positive")
    last = tensor.shape[-1]
    padded = math.ceil(last / group_size) * group_size
    flat = tensor.float().reshape(-1, last)
    if padded != last:
        flat = torch.nn.functional.pad(flat, (0, padded - last))
    return flat.reshape(-1, group_size), padded


def compress_tensor(tensor: torch.Tensor, kind: str = "fp4", group_size: int = 64) -> PackedTensor:
    """Compress a tensor along its last dimension with portable PyTorch ops."""

    if tensor.numel() == 0:
        raise ValueError("cannot compress an empty tensor")
    kind = kind.lower()
    if kind in {"fp16", "float16"}:
        return PackedTensor(
            tensor.to(torch.float16).contiguous(), None, tuple(tensor.shape), 16,
            group_size, "fp16", tensor.dtype, tensor.shape[-1]
        )
    groups, padded = _group_view(tensor, group_size)
    max_abs = groups.abs().amax(dim=1).clamp_min(1e-8)
    if kind in {"int8", "i8"}:
        scale = (max_abs / 127.0).to(torch.float16)
        quantized = torch.round(groups / scale.float().unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        return PackedTensor(
            quantized, scale, tuple(tensor.shape), 8, group_size, "int8", tensor.dtype, padded
        )
    if kind not in {"fp4", "fp4_e2m1"}:
        raise ValueError(f"unsupported compression kind: {kind}")
    scale = (max_abs / 8.0).to(torch.float16)
    codebook = torch.tensor(FP4_CODEBOOK, dtype=torch.float32, device=tensor.device)
    normalized = groups / scale.float().unsqueeze(1)
    codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)
    if codes.shape[1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return PackedTensor(
        packed.contiguous(), scale, tuple(tensor.shape), 4, group_size, "fp4_e2m1", tensor.dtype, padded
    )


def decompress_tensor(packed: PackedTensor) -> torch.Tensor:
    """Restore a compressed tensor to its original shape and dtype."""

    if packed.kind == "fp16":
        return packed.payload.reshape(packed.shape).to(packed.original_dtype)
    if packed.scale is None:
        raise ValueError("quantized tensor is missing scales")
    if packed.kind == "int8":
        values = packed.payload.float() * packed.scale.float().unsqueeze(1)
    elif packed.kind == "fp4_e2m1":
        low = packed.payload & 0x0F
        high = (packed.payload >> 4) & 0x0F
        codes = torch.stack((low, high), dim=-1).reshape(packed.payload.shape[0], -1)
        codebook = torch.tensor(FP4_CODEBOOK, dtype=torch.float32, device=packed.payload.device)
        values = codebook[codes.long()] * packed.scale.float().unsqueeze(1)
    else:
        raise ValueError(f"unsupported packed kind: {packed.kind}")
    flat = values.reshape(-1, packed.padded_last_dim)[..., : packed.shape[-1]]
    return flat.reshape(packed.shape).to(packed.original_dtype)


@dataclass
class _KVEntry:
    key: PackedTensor
    value: PackedTensor
    share_key: str | None
    canonical: bool


def _fingerprint(tensor: torch.Tensor) -> str:
    """Hash raw tensor bytes for share validation without retaining the tensor."""

    contiguous = tensor.detach().cpu().contiguous()
    raw = bytes(contiguous.view(torch.uint8).reshape(-1).tolist())
    descriptor = f"{tuple(contiguous.shape)}:{contiguous.dtype}".encode("utf-8")
    return hashlib.blake2b(descriptor + raw, digest_size=16).hexdigest()


class CrossLayerKVCache:
    """Compressed KV cache with optional cross-layer references."""

    def __init__(self, *, kind: str = "fp4", group_size: int = 64) -> None:
        self.kind = kind
        self.group_size = group_size
        self._entries: dict[int, _KVEntry] = {}
        self._canonical: dict[str, tuple[PackedTensor, PackedTensor]] = {}
        self._canonical_fingerprints: dict[str, tuple[str, str]] = {}

    def put(self, layer: int, key: torch.Tensor, value: torch.Tensor, share_key: str | None = None) -> None:
        if share_key is not None and share_key in self._canonical:
            canonical_key, canonical_value = self._canonical[share_key]
            raw_key, raw_value = self._canonical_fingerprints[share_key]
            if _fingerprint(key) != raw_key:
                raise ValueError(f"layer {layer} does not match shared KV group {share_key}")
            if _fingerprint(value) != raw_value:
                raise ValueError(f"layer {layer} value does not match shared KV group {share_key}")
            self._entries[layer] = _KVEntry(canonical_key, canonical_value, share_key, False)
            return
        packed_key = compress_tensor(key, self.kind, self.group_size)
        packed_value = compress_tensor(value, self.kind, self.group_size)
        self._entries[layer] = _KVEntry(packed_key, packed_value, share_key, True)
        if share_key is not None:
            self._canonical[share_key] = (packed_key, packed_value)
            self._canonical_fingerprints[share_key] = (_fingerprint(key), _fingerprint(value))

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self._entries[layer]
        return decompress_tensor(entry.key), decompress_tensor(entry.value)

    @property
    def layers(self) -> int:
        return len(self._entries)

    @property
    def canonical_layers(self) -> int:
        return sum(entry.canonical for entry in self._entries.values())

    @property
    def compressed_nbytes(self) -> int:
        unique = sum(entry.key.compressed_nbytes + entry.value.compressed_nbytes for entry in self._entries.values() if entry.canonical)
        references = sum(8 for entry in self._entries.values() if not entry.canonical)
        return unique + references


class TieredKVCache:
    """Separate persistent global KV from temporary encoder/SWA KV."""

    def __init__(self, *, kind: str = "fp4", group_size: int = 64) -> None:
        self.persistent = CrossLayerKVCache(kind=kind, group_size=group_size)
        self.temporary = CrossLayerKVCache(kind=kind, group_size=group_size)

    def put(self, layer: int, key: torch.Tensor, value: torch.Tensor, *, persistent: bool, share_key: str | None = None) -> None:
        (self.persistent if persistent else self.temporary).put(layer, key, value, share_key)

    def clear_temporary(self) -> None:
        self.temporary = CrossLayerKVCache(kind=self.persistent.kind, group_size=self.persistent.group_size)

    @property
    def compressed_nbytes(self) -> int:
        return self.persistent.compressed_nbytes + self.temporary.compressed_nbytes


class EngramShardTable:
    """Small sharded content-addressed table for Engram-style embeddings."""

    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("invalid rank/world_size")
        self.rank = rank
        self.world_size = world_size
        self._values: dict[str, torch.Tensor] = {}

    def owner(self, key: str) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.world_size

    def put(self, key: str, value: torch.Tensor) -> int:
        owner = self.owner(key)
        if owner == self.rank:
            self._values[key] = value.detach().cpu().contiguous()
        return owner

    def get_local(self, key: str) -> torch.Tensor | None:
        return self._values.get(key)

    @property
    def entries(self) -> int:
        return len(self._values)

    @property
    def storage_nbytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self._values.values())


class Int8GradientCompressor:
    """Reference gradient compression payload used before an all-reduce."""

    def compress(self, gradient: torch.Tensor) -> PackedTensor:
        return compress_tensor(gradient, kind="int8", group_size=256)

    def decompress(self, payload: PackedTensor) -> torch.Tensor:
        return decompress_tensor(payload)


class BoundedKVReplay:
    """Bounded approximate replay metadata for interrupted KV states."""

    _PAYLOAD_VERSION = 1

    def __init__(
        self,
        max_tokens: int = 1024,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")
        self.max_tokens = max_tokens
        self._states: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self.evictions = 0
        self._rng = rng if rng is not None else random.Random(seed)

    @property
    def rng(self) -> random.Random:
        """The replay RNG, exposed for deterministic scheduler integration."""

        return self._rng

    def append(
        self,
        sample_id: str,
        token_index: int,
        *,
        policy_version: int,
        kv_bytes: int,
        payload: Any = None,
    ) -> None:
        """Append one token state, optionally retaining its replay payload.

        Payloads are deliberately opaque to the replay container.  They may be
        JSON-compatible metadata, bytes, or tensors.  Persistence encodes the
        payload without changing the in-memory object, and restoring creates a
        detached copy for tensor payloads.
        """

        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if token_index < 0:
            raise ValueError("token_index must be non-negative")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if kv_bytes < 0:
            raise ValueError("kv_bytes must be non-negative")
        key = (sample_id, token_index)
        self._states[key] = {
            "policy_version": int(policy_version),
            "kv_bytes": int(kv_bytes),
            "payload": payload,
        }
        self._states.move_to_end(key)
        while len(self._states) > self.max_tokens:
            self._states.popitem(last=False)
            self.evictions += 1

    def get(self, sample_id: str, token_index: int) -> dict[str, Any] | None:
        """Return a replay state without exposing the internal mapping."""

        state = self._states.get((sample_id, token_index))
        return None if state is None else dict(state)

    def items(self) -> list[tuple[tuple[str, int], dict[str, Any]]]:
        """Return states in eviction order as a detached list."""

        return [(key, dict(state)) for key, state in self._states.items()]

    @property
    def tokens(self) -> int:
        return len(self._states)

    @property
    def retained_bytes(self) -> int:
        return sum(int(state["kv_bytes"]) for state in self._states.values())

    @staticmethod
    def _encode_value(value: Any) -> Any:
        """Encode common payload values into a JSON-safe representation."""

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return {
                "__flashrl_type__": "bytes",
                "data": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, bytearray):
            return {
                "__flashrl_type__": "bytes",
                "data": base64.b64encode(bytes(value)).decode("ascii"),
            }
        if isinstance(value, tuple):
            return {
                "__flashrl_type__": "tuple",
                "items": [BoundedKVReplay._encode_value(item) for item in value],
            }
        if isinstance(value, list):
            return [BoundedKVReplay._encode_value(item) for item in value]
        if isinstance(value, dict):
            if all(isinstance(key, str) for key in value):
                return {key: BoundedKVReplay._encode_value(item) for key, item in value.items()}
        if torch.is_tensor(value):
            buffer = io.BytesIO()
            torch.save(value.detach().cpu(), buffer)
            return {
                "__flashrl_type__": "torch_tensor",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }

        # A replay payload may be an application-specific object.  Keep the
        # on-disk format JSON while retaining a useful round-trip for local
        # checkpoints.  Restore is intentionally for checkpoints created by
        # this process, rather than an untrusted input format.
        encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "__flashrl_type__": "pickle",
            "data": base64.b64encode(encoded).decode("ascii"),
        }

    @staticmethod
    def _decode_value(value: Any) -> Any:
        if isinstance(value, list):
            return [BoundedKVReplay._decode_value(item) for item in value]
        if not isinstance(value, dict):
            return value
        value_type = value.get("__flashrl_type__")
        if value_type == "bytes":
            return base64.b64decode(value["data"])
        if value_type == "tuple":
            return tuple(BoundedKVReplay._decode_value(item) for item in value["items"])
        if value_type == "torch_tensor":
            buffer = io.BytesIO(base64.b64decode(value["data"]))
            try:
                return torch.load(buffer, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older torch fallback
                buffer.seek(0)
                return torch.load(buffer, map_location="cpu")
        if value_type == "pickle":
            return pickle.loads(base64.b64decode(value["data"]))
        return {key: BoundedKVReplay._decode_value(item) for key, item in value.items()}

    @staticmethod
    def _encode_rng_state(state: object) -> str:
        return base64.b64encode(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

    @staticmethod
    def _decode_rng_state(value: str) -> object:
        return pickle.loads(base64.b64decode(value))

    def to_payload(self) -> dict[str, Any]:
        """Return a versioned, JSON-safe snapshot of the replay."""

        states = []
        for (sample_id, token_index), state in self._states.items():
            states.append(
                {
                    "sample_id": sample_id,
                    "token_index": token_index,
                    "policy_version": int(state["policy_version"]),
                    "kv_bytes": int(state["kv_bytes"]),
                    "payload": self._encode_value(state.get("payload")),
                }
            )
        return {
            "format": "flashrl.bounded_kv_replay",
            "version": self._PAYLOAD_VERSION,
            "max_tokens": self.max_tokens,
            "evictions": self.evictions,
            "states": states,
            "rng_state": self._encode_rng_state(self._rng.getstate()),
        }

    @property
    def payload(self) -> dict[str, Any]:
        """Alias for :meth:`to_payload` useful to checkpoint writers."""

        return self.to_payload()

    state_dict = to_payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BoundedKVReplay":
        """Restore a replay from :meth:`to_payload` output."""

        if not isinstance(payload, dict):
            raise TypeError("replay payload must be a mapping")
        if payload.get("format") != "flashrl.bounded_kv_replay":
            raise ValueError("invalid bounded KV replay format")
        if payload.get("version") != cls._PAYLOAD_VERSION:
            raise ValueError(f"unsupported bounded KV replay version: {payload.get('version')!r}")
        replay = cls(int(payload["max_tokens"]))
        replay.evictions = int(payload.get("evictions", 0))
        if replay.evictions < 0:
            raise ValueError("replay evictions must be non-negative")
        states = payload.get("states", [])
        if not isinstance(states, list) or len(states) > replay.max_tokens:
            raise ValueError("replay states exceed max_tokens")
        for state in states:
            if not isinstance(state, dict):
                raise ValueError("replay state must be a mapping")
            replay.append(
                str(state["sample_id"]),
                int(state["token_index"]),
                policy_version=int(state["policy_version"]),
                kv_bytes=int(state["kv_bytes"]),
                payload=cls._decode_value(state.get("payload")),
            )
        # Appending cannot evict because the snapshot is bounded.  Restore the
        # original counter after rebuilding to preserve observability.
        replay.evictions = int(payload.get("evictions", 0))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            replay._rng.setstate(cls._decode_rng_state(rng_state))
        return replay

    from_state_dict = from_payload

    def persist(self, path: str | Path) -> Path:
        """Atomically persist a replay snapshot as UTF-8 JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_payload(), handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target

    @classmethod
    def restore(cls, path: str | Path) -> "BoundedKVReplay":
        """Restore a replay snapshot written by :meth:`persist`."""

        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_payload(payload)

    # Explicit aliases make checkpoint integration readable without coupling
    # callers to the JSON implementation details.
    snapshot = to_payload
    load = restore


def run_compression_benchmark(
    *,
    layers: int = 4,
    tokens: int = 16,
    kv_heads: int = 2,
    head_dim: int = 16,
    group_size: int = 8,
    share_every: int = 2,
) -> dict[str, Any]:
    """Compare FP16/INT8/FP4 and cross-layer KV on a deterministic toy cache."""

    if min(layers, tokens, kv_heads, head_dim, group_size, share_every) < 1:
        raise ValueError("benchmark dimensions must be positive")
    torch.manual_seed(20260923)
    shape = (tokens, kv_heads, head_dim)
    keys = [torch.randn(shape, dtype=torch.float32) for _ in range(layers)]
    values = [torch.randn(shape, dtype=torch.float32) for _ in range(layers)]
    # Cross-layer sharing is only valid when the trained model emits matching
    # KV. The fixture makes this condition explicit for paired layers.
    for layer in range(1, layers, share_every):
        source = layer - (layer % share_every)
        keys[layer] = keys[source].clone()
        values[layer] = values[source].clone()
    fp16_bytes = sum(t.numel() * 2 for t in keys + values)
    fp16_restored = [t.to(torch.float16).to(torch.float32) for t in keys + values]
    fp16_mse = sum(float((a - b).pow(2).mean()) for a, b in zip(keys + values, fp16_restored)) / len(fp16_restored)
    formats: dict[str, Any] = {"fp16": {"bytes": fp16_bytes, "mse": fp16_mse}}
    for kind in ("int8", "fp4"):
        packed = [compress_tensor(t, kind=kind, group_size=group_size) for t in keys + values]
        restored = [decompress_tensor(item) for item in packed]
        mse = sum(float((a - b).pow(2).mean()) for a, b in zip(keys + values, restored)) / len(packed)
        bytes_used = sum(item.compressed_nbytes for item in packed)
        formats[kind] = {"bytes": bytes_used, "mse": mse, "compression_ratio": fp16_bytes / bytes_used}

    cache = CrossLayerKVCache(kind="fp4", group_size=group_size)
    for layer, (key, value) in enumerate(zip(keys, values)):
        cache.put(layer, key, value, share_key=f"pair-{layer // share_every}")
    recovered_mse = sum(float((keys[layer] - cache.get(layer)[0]).pow(2).mean()) for layer in range(layers)) / layers
    tiers = TieredKVCache(kind="fp4", group_size=group_size)
    for layer, (key, value) in enumerate(zip(keys, values)):
        tiers.put(layer, key, value, persistent=layer < max(1, layers // 2), share_key=f"pair-{layer // share_every}")
    tier_before_clear = tiers.compressed_nbytes
    persistent_bytes = tiers.persistent.compressed_nbytes
    tiers.clear_temporary()
    engram = EngramShardTable(rank=0, world_size=2)
    for index in range(8):
        engram.put(f"token-{index}", torch.ones(8, dtype=torch.float16))
    gradient = torch.randn(1024)
    gradient_payload = Int8GradientCompressor().compress(gradient)
    replay = BoundedKVReplay(max_tokens=8)
    for token_index in range(16):
        replay.append("sample-0", token_index, policy_version=token_index // 4, kv_bytes=cache.compressed_nbytes // 4)
    return {
        "shape_per_layer": list(shape),
        "layers": layers,
        "tokens": tokens,
        "fp16_bytes": fp16_bytes,
        "formats": formats,
        "cross_layer_fp4": {
            "share_every": share_every,
            "canonical_layers": cache.canonical_layers,
            "referenced_layers": cache.layers - cache.canonical_layers,
            "bytes": cache.compressed_nbytes,
            "bytes_per_token": cache.compressed_nbytes / tokens,
            "mse": recovered_mse,
        },
        "tiered_kv": {
            "persistent_bytes": persistent_bytes,
            "temporary_bytes_before_clear": tier_before_clear - persistent_bytes,
            "total_bytes_before_clear": tier_before_clear,
            "total_bytes_after_clear": tiers.compressed_nbytes,
        },
        "engram_shard": {"world_size": 2, "local_entries_rank0": engram.entries, "local_bytes_rank0": engram.storage_nbytes},
        "gradient_int8": {
            "original_bytes": gradient.numel() * gradient.element_size(),
            "compressed_bytes": gradient_payload.compressed_nbytes,
            "compression_ratio": (gradient.numel() * gradient.element_size()) / gradient_payload.compressed_nbytes,
            "mse": float((gradient - Int8GradientCompressor().decompress(gradient_payload)).pow(2).mean()),
        },
        "bounded_kv_replay": {
            "max_tokens": replay.max_tokens,
            "retained_tokens": replay.tokens,
            "evictions": replay.evictions,
            "retained_bytes": replay.retained_bytes,
        },
    }

