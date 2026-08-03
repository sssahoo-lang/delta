"""Content-addressed disk cache for model responses.

This is the single most important piece of infrastructure in the project, and it
exists from the first phase rather than being retrofitted. Three reasons:

1. **Cost.** The optimizer re-evaluates overlapping question sets constantly. A
   cache hit costs nothing and consumes no rate-limit quota.
2. **Reproducibility.** Re-running a completed evaluation returns byte-identical
   answers, so a reported number can be regenerated exactly.
3. **Iteration speed.** A cached evaluation pass finishes in under a second
   instead of the forty minutes that Groq's 30 requests/minute limit imposes.

The key covers everything that could change the response: model, system prompt,
user prompt, and sampling parameters. Nothing else is included, so an unrelated
code change does not invalidate the cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from delta.config import CACHE_DIR

_lock = threading.Lock()


def cache_key(model_id: str, system_prompt: str, user_prompt: str, params: dict) -> str:
    payload = json.dumps(
        {
            "model": model_id,
            "system": system_prompt,
            "user": user_prompt,
            "params": params,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, root: Path | None = None, enabled: bool = True) -> None:
        self.root = Path(root) if root else CACHE_DIR / "responses"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path_for(self, key: str) -> Path:
        # Two-level fan-out keeps directory listings small once there are tens of
        # thousands of cached responses.
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A truncated file from an interrupted write is a miss, not a crash.
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with _lock:
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(path)  # atomic, so a crash cannot leave a partial entry

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4)}
