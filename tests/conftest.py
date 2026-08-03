from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DB_PATH = REPO_ROOT / "data" / "sample.db"


@pytest.fixture(scope="session")
def sample_db() -> Path:
    """The offline fixture, built on demand so tests never require a make target."""
    if not DB_PATH.exists():
        from make_sample_db import main as build

        build()
    return DB_PATH


@pytest.fixture(scope="session")
def sample_questions() -> list[dict]:
    from make_sample_db import QUESTIONS

    return QUESTIONS
