# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "numpy",
#     "joblib",
#     "diskcache",
#     "marimo>=0.23.8",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Torch-tensor caching through marimo's real `LazyLoader` with the
`.pt` codec (`LAZY_STUB_LOOKUP["torch.Tensor"]`), versus mandala-style
`joblib.hash` + `joblib.dump`.

marimo's key path content-addresses torch tensors through the buffer
protocol (`data_to_buffer`); the `.pt` lazy codec routes the blob
through `torch.save` / `torch.load`. Both marimo stages run the real
disk-backed loader paths (`LazyLoader.save_cache` / `load_cache`,
including manifest encode/decode and the blob thread pool), not
primitive calls — run this against a marimo checkout that carries the
codec patch, e.g.:

    PYTHONPATH=../marimo uv run lib/bench_torch.py 50 500
"""
from __future__ import annotations

import functools
import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np


def median_ms(fn, runs: int = 5) -> float:
    fn()  # warm
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def _mo_tensor_key(t) -> bytes:
    """marimo's content key for a data primitive: buffer view + sha256."""
    from marimo._save import hash as mohash

    return hashlib.sha256(mohash.data_to_buffer(t)).digest()


def _bench_cache(payload):
    """Single-def Pure cache envelope, as in `lib/compute.py`."""
    from marimo._save.cache import Cache

    return Cache(
        defs={"x": payload},
        hash="bench",
        cache_type="Pure",
        stateful_refs=set(),
        hit=True,
        meta={},
    )


def bench(size_mb: float, runs: int = 5) -> dict[str, float]:
    import torch

    from marimo._save.hash import HashKey
    from marimo._save.loaders import LazyLoader
    from marimo._save.loaders.lazy import LazyStore, maybe_update_lazy_stub

    n = int(size_mb * 1024**2 // 4)  # float32
    t = torch.randn(n)
    codec = maybe_update_lazy_stub(t)
    assert codec == "pt", (
        f"torch codec not registered (got {codec!r}); "
        "run against a marimo checkout with the .pt patch"
    )

    out: dict[str, float] = {}
    out["key: mo buffer+sha256"] = median_ms(
        functools.partial(_mo_tensor_key, t), runs)
    out["key: joblib.hash"] = median_ms(
        functools.partial(joblib.hash, t), runs)

    key = HashKey("bench", "Pure")

    # Save timing excludes the temp-dir churn: a fresh directory per
    # run keeps every save a true first write, but mkdtemp/rmtree are
    # not part of the path being measured.
    save_samples = []
    for _ in range(runs + 1):  # first iteration is the warmup
        d = tempfile.mkdtemp()
        try:
            loader = LazyLoader("bench", store=LazyStore(save_path=d))
            t0 = time.perf_counter()
            assert loader.save_cache(_bench_cache(t))
            loader.flush()  # charge background writers to the save
            save_samples.append((time.perf_counter() - t0) * 1000)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    out["save: mo LazyLoader (.pt)"] = float(np.median(save_samples[1:]))

    d = tempfile.mkdtemp()
    try:
        loader = LazyLoader("bench", store=LazyStore(save_path=d))
        assert loader.save_cache(_bench_cache(t))
        loader.flush()
        out["load: mo LazyLoader (.pt)"] = median_ms(
            functools.partial(loader.load_cache, key), runs)

        jb = Path(d) / "blob.joblib"
        out["save: joblib.dump"] = median_ms(
            functools.partial(joblib.dump, t, str(jb)), runs)

        def load_joblib():
            with open(jb, "rb") as fh:
                return joblib.load(fh)

        out["load: joblib.load"] = median_ms(load_joblib, runs)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def report(sizes_mb=(50, 500)) -> dict[str, dict[str, float]]:
    import platform

    results = {}
    print(f"host: {platform.platform()} ({platform.machine()})")
    for s in sizes_mb:
        r = bench(s)
        results[str(s)] = r
        mo_hit = r["key: mo buffer+sha256"] + r["load: mo LazyLoader (.pt)"]
        md_hit = r["key: joblib.hash"] + r["load: joblib.load"]
        mo_miss = r["key: mo buffer+sha256"] + r["save: mo LazyLoader (.pt)"]
        md_miss = r["key: joblib.hash"] + r["save: joblib.dump"]
        print(f"--- {s} MB float32 tensor")
        for k, v in r.items():
            print(f"  {k:24s} {v:9.1f} ms")
        print(f"  hit  (key+load): mo+.pt {mo_hit:7.1f} ms | "
              f"mandala-style {md_hit:7.1f} ms | ratio {md_hit / mo_hit:.2f}x")
        print(f"  miss (key+save): mo+.pt {mo_miss:7.1f} ms | "
              f"mandala-style {md_miss:7.1f} ms | ratio {md_miss / mo_miss:.2f}x")
    return results


if __name__ == "__main__":
    sizes = tuple(float(a) for a in sys.argv[1:]) or (50, 500)
    report(sizes)
