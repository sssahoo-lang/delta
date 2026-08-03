"""Stratified sampling helpers.

Spider's ``dev.json`` is grouped by database. Taking the first N examples therefore
samples one or two schemas rather than a cross-section. Stratify by difficulty
(and secondarily by ``db_id``) so smoke tests and real-path checks stay honest.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from delta.evalh.dataset import Example


def stratified_sample(
    examples: Sequence[Example],
    n: int,
    seed: int = 0,
    by: str = "difficulty",
) -> list[Example]:
    """Draw ``n`` examples, balancing the ``by`` attribute as evenly as possible."""
    if n <= 0:
        return []
    if n >= len(examples):
        return list(examples)

    buckets: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        key = getattr(ex, by, "unknown")
        buckets[str(key)].append(ex)

    rng = random.Random(seed)
    for key in buckets:
        rng.shuffle(buckets[key])

    keys = sorted(buckets)
    # Round-robin across buckets so each difficulty/db gets a fair share.
    out: list[Example] = []
    idxs = {k: 0 for k in keys}
    while len(out) < n:
        progressed = False
        for key in keys:
            if idxs[key] < len(buckets[key]):
                out.append(buckets[key][idxs[key]])
                idxs[key] += 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out
