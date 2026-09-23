"""Dataset manifest support for experimental integrity."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_dataset_manifest(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    source: str,
    raw_filename: str,
    raw_path: str | Path,
    row_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the identity of a raw dataset used in an experiment."""

    raw_file = Path(raw_path)
    manifest = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "source": source,
        "raw_filename": raw_filename,
        "row_count": int(row_count),
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": compute_sha256(raw_file),
    }
    if metadata is not None:
        manifest.update(metadata)
    return manifest


__all__ = ["compute_sha256", "create_dataset_manifest"]
