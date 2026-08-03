#!/usr/bin/env python3
"""Download and verify the Spider 1.0 benchmark, including its SQLite databases.

The widely used ``xlangai/spider`` HuggingFace parquet contains only
question/SQL pairs. It ships **no databases**, which makes execution accuracy
impossible to compute from it. This script instead pulls ``HAL-9001/spider-databases``,
a re-host of the canonical Yale distribution that includes the SQLite files, and
verifies it against a pinned SHA256 so the benchmark cannot silently change
underneath a set of results.

    python scripts/download_spider.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

HF_REPO = "HAL-9001/spider-databases"
HF_FILENAME = "spider_data.zip"
EXPECTED_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACT_ROOT = DATA_DIR / "spider_data"

# Files the rest of the pipeline depends on existing after extraction.
REQUIRED_MEMBERS = ["tables.json", "dev.json", "train_spider.json", "database"]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required. Run: uv pip install -e '.[dev]'")

    print(f"downloading {HF_REPO}/{HF_FILENAME} (206 MB)...")
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")
    return Path(cached)


def verify(archive: Path) -> None:
    print("verifying checksum...")
    actual = sha256_of(archive)
    if actual != EXPECTED_SHA256:
        sys.exit(
            "SHA256 mismatch, refusing to use this archive.\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  actual   {actual}\n"
            "The upstream re-host may have changed. Do not proceed with results "
            "from an unverified benchmark."
        )
    print("  checksum ok")


def extract(archive: Path) -> None:
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"extracting into {EXTRACT_ROOT}...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)

    # The archive nests everything under spider_data/; tolerate either shape.
    if not EXTRACT_ROOT.exists():
        sys.exit(f"expected {EXTRACT_ROOT} after extraction, not found")

    missing = [m for m in REQUIRED_MEMBERS if not (EXTRACT_ROOT / m).exists()]
    if missing:
        sys.exit(f"extraction incomplete, missing: {', '.join(missing)}")

    db_count = len(list((EXTRACT_ROOT / "database").glob("*/*.sqlite")))
    print(f"  found {db_count} SQLite databases")


def main() -> None:
    if (EXTRACT_ROOT / "dev.json").exists():
        print(f"Spider already present at {EXTRACT_ROOT}. Delete it to re-download.")
        return
    archive = fetch()
    verify(archive)
    extract(archive)
    print("\nSpider ready. Next: python scripts/run_baseline.py")


if __name__ == "__main__":
    main()
