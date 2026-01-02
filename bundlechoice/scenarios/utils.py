"""Utilities shared by scenario factory helpers."""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from typing import Callable, Optional, TypeVar

import numpy as np
from mpi4py import MPI

T = TypeVar("T")


@contextmanager
def maybe_timeout(seconds: Optional[int]) -> None:
    """POSIX alarm-based timeout (no-op when ``seconds`` is falsy)."""

    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def mpi_call_with_timeout(
    comm: MPI.Comm,
    func: Callable[[], T],
    timeout_seconds: Optional[int],
    label: str,
) -> T:
    """
    Execute ``func`` collectively with an optional timeout.

    Every rank participates; on timeout we raise ``TimeoutError`` to prevent
    silent hangs under MPI.
    """

    comm.Barrier()
    start = time.perf_counter()

    try:
        with maybe_timeout(timeout_seconds):
            result = func()
    finally:
        comm.Barrier()

    elapsed = time.perf_counter() - start
    root = comm.Get_rank() == 0
    if root:
        print(f"[{label}] completed in {elapsed:.2f}s", flush=True)
    return result


def rng(seed: Optional[int] = None) -> np.random.Generator:
    """Deterministic RNG convenience wrapper."""

    return np.random.default_rng(seed)


def root_dict(comm: MPI.Comm, payload: Optional[dict]) -> dict:
    """
    Return ``payload`` on root and ``None`` elsewhere, ensuring the shape of
    the dictionary is consistent across ranks.
    """

    return payload if comm.Get_rank() == 0 else None


__all__ = [
    "maybe_timeout",
    "mpi_call_with_timeout",
    "rng",
    "root_dict",
]


