# Deterministic seeding and worker-count helpers shared across Step 3/4/5.
# Main functions: derive_seed and resolve_worker_count.
# The seed derivation intentionally depends only on semantic keys, not loop order.

from __future__ import annotations

import hashlib
from os import cpu_count
from typing import Any


def derive_seed(base_seed: int, *keys: Any) -> int:
    """Derive a deterministic 32-bit seed from a base seed and semantic keys.

    This helper ensures that random seeds are stable across:
    - serial versus parallel execution,
    - different worker counts,
    - different task scheduling orders.

    Args:
        base_seed: User-provided base seed.
        *keys: Additional semantic identifiers (for example run id, participant id,
            model name, block id, and task label).

    Returns:
        Seed value in the range [0, 2_147_483_647].
    """
    payload = "|".join([str(int(base_seed)), *(str(key) for key in keys)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # Keep seed in the common signed 32-bit positive range accepted by NumPy APIs.
    return int.from_bytes(digest, byteorder="big", signed=False) % 2_147_483_648


def resolve_worker_count(requested_workers: int, n_tasks: int) -> int:
    """Validate and clamp requested worker count to a safe executable value.

    Args:
        requested_workers: Requested number of processes.
        n_tasks: Number of independent tasks available.

    Returns:
        Effective worker count in [1, min(cpu_count, n_tasks)].

    Raises:
        ValueError: If `requested_workers` is less than 1.
    """
    workers = int(requested_workers)
    if workers < 1:
        raise ValueError("workers must be >= 1.")
    tasks = max(int(n_tasks), 1)
    max_cpus = max(int(cpu_count() or 1), 1)
    return max(1, min(workers, max_cpus, tasks))
