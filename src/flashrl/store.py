"""Append-only JSONL records with SQLite state and idempotent writes."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .schemas import encode


class EventStore:
    """Durable local store used by the MVP.

    The data payload is append-only JSONL; SQLite only tracks the latest stage
    for each idempotency key. This keeps derived experiences rebuildable.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite3")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS stages "
            "(key TEXT PRIMARY KEY, stage TEXT NOT NULL, payload_path TEXT NOT NULL)"
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def put(self, key: str, stage: str, record: Any) -> bool:
        """Write once; return False for an existing idempotency key."""
        if self.db.execute("SELECT 1 FROM stages WHERE key = ?", (key,)).fetchone():
            return False
        payload = encode(record)
        path = self.root / f"{stage}.jsonl"
        fd, tmp_name = tempfile.mkstemp(prefix=f"{stage}-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with path.open("a", encoding="utf-8") as target, open(tmp_name, encoding="utf-8") as source:
                target.write(source.read())
                target.flush()
                os.fsync(target.fileno())
            self.db.execute(
                "INSERT INTO stages(key, stage, payload_path) VALUES (?, ?, ?)",
                (key, stage, str(path)),
            )
            self.db.commit()
            return True
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def count(self, stage: str | None = None) -> int:
        if stage is None:
            return int(self.db.execute("SELECT COUNT(*) FROM stages").fetchone()[0])
        return int(self.db.execute("SELECT COUNT(*) FROM stages WHERE stage = ?", (stage,)).fetchone()[0])

    def read_stage(self, stage: str) -> Iterable[dict[str, Any]]:
        path = self.root / f"{stage}.jsonl"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def inspect(self, key_prefix: str = "") -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT key, stage, payload_path FROM stages WHERE key LIKE ? ORDER BY rowid",
            (f"{key_prefix}%",),
        ).fetchall()
        return [{"key": k, "stage": s, "payload_path": p} for k, s, p in rows]

