"""Optional SGLang policy update client."""

from __future__ import annotations

from pathlib import Path


def update_sglang_from_disk(base_url: str, checkpoint_path: str | Path, token_step: int) -> dict:
    """Publish a checkpoint through SGLang's control-plane endpoint."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install flash-rl-infra[http] to sync SGLang weights") from exc
    response = httpx.post(
        f"{base_url.rstrip('/')}/update_weights_from_disk",
        json={"model_path": str(checkpoint_path), "token_step": token_step, "flush_cache": True},
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()

