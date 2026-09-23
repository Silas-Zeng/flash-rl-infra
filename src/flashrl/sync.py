"""Optional SGLang policy update client."""

from __future__ import annotations

from pathlib import Path


def update_sglang_from_disk(
    base_url: str,
    checkpoint_path: str | Path,
    token_step: int,
    *,
    weight_version: str | None = None,
    abort_all_requests: bool = True,
) -> dict:
    """Publish a checkpoint through SGLang's control-plane endpoint."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install flash-rl-infra[http] to sync SGLang weights") from exc
    payload = {
        "model_path": str(checkpoint_path),
        "token_step": token_step,
        "flush_cache": True,
        "abort_all_requests": abort_all_requests,
    }
    if weight_version is not None:
        payload["weight_version"] = weight_version
    response = httpx.post(
        f"{base_url.rstrip('/')}/update_weights_from_disk",
        json=payload,
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()

