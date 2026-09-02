"""Process-level runtime fixes that must run *before* torch is imported.

Anaconda ships its own copy of the Intel OpenMP runtime (``libiomp5md.dll``) and
so does the PyTorch wheel. When both get loaded into one process the Intel
runtime aborts with ``OMP: Error #15``. The supported fix is to install a single
OpenMP runtime; on this machine that would mean rebuilding the Anaconda
environment, so we use the documented escape hatch instead and additionally pin
the thread count so the two runtimes cannot oversubscribe the CPU.

Import this module (``import app.core.runtime``) at the top of any entry point
before importing torch/numpy.
"""
from __future__ import annotations

import os

_DEFAULTS = {
    # Allow the duplicated Intel OpenMP runtime (Anaconda MKL + torch).
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    # Keep the two OpenMP runtimes from each spawning a full thread pool.
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    # Reduce fragmentation on the 4 GB GTX 1650.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)


def summary() -> dict[str, str]:
    """Return the runtime environment variables this module manages."""
    return {key: os.environ.get(key, "") for key in _DEFAULTS}
