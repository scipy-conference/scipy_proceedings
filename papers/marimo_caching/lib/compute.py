"""Benchmark primitives and plotting helpers for figures in main.md.

Two responsibilities, kept in one module so the figure cells in
main.md need a single import:

- **Bench primitives** (``run_hash_microbench``, ``run_load_microbench``,
  ``sweep_payload_sizes``, ``sweep_methods`` and their helpers)
  produce the numbers shown in §5. Mirrors the standalone bench at
  https://molab.marimo.io/notebooks/nb_LYSFWVz4sxJTPou3yBa8d9 ; the
  paper reproduces it in-cell rather than transcribing results.

- **Plot helpers** (``plot_hash_throughput`` etc.) absorb the
  matplotlib axis configuration so the figure cells stay short.

Marimo-only features (``mo.cache``, ``mo.persistent_cache``)
require a notebook runtime and so cannot be set up inside this
module; the end-to-end benchmark accepts the already-decorated
methods from its calling cell.
"""
from __future__ import annotations

import functools
import hashlib
import io
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import diskcache
import joblib
import numpy as np

from marimo._save import hash as mohash
from marimo._save.cache import Cache
from marimo._save.hash import HashKey
from marimo._save.loaders import LazyLoader, PickleLoader
from marimo._save.loaders import lazy as molazy
from marimo._save.loaders.lazy import LazyStore
from marimo._save.stores.file import FileStore


# ---- Constants --------------------------------------------------------------

# Fixed run count for every timed measurement (hit sweep, decomposition,
# and miss). One warmup call precedes these in `samples_ms`.
BENCH_RUNS = 10

# Lift diskcache's 1 GB default so only `memoize`'s SQLite blob
# ceiling causes a method to fall out of the comparison.
DC_SIZE_LIMIT = 64 * 1024 ** 3

# Shared cross-figure palette.
PALETTE: dict[str, str] = {
    "diskcache (fixed key)":      "#ff7f0e",
    "mo.cache":                   "#d62728",
    "mo.persistent_cache":        "#1f77b4",
    "mo.persistent_cache (lazy)": "#7fbbe6",
    "mandala":                    "#2ca02c",
    "diskcache.memoize":          "#9467bd",
}


# ---- Tiny utilities ---------------------------------------------------------

ASSERTION_FAILURE = (
    "Claim assertion failed — please let the authors know (dylan@marimo.io)!"
)


def claim(label: str, **values) -> str:
    """Format an assertion-failure message for a figure-cell claim.

    Intended as the second argument to a plain `assert`:

        assert (rel_err < 0.25).all(), compute.claim(
            "throughput within 25% of plateau", rel_err=rel_err, ...
        )

    Python evaluates the assertion's failure message lazily, so the
    formatting cost is only paid on a failing claim.
    """
    body = "\n".join(f"  {k} = {v!r}" for k, v in values.items())
    return f"{label}\n{body}\n\n{ASSERTION_FAILURE}"


def identity(value):
    """Module-level identity. Used as the call target for caches that need
    to wrap *something* — e.g. ``mo.persistent_cache()(identity)`` or
    ``cache.memoize()(identity)``."""
    return value


# Default settle window between cache write (warmup) and the measurement
# pass. `mo.persistent_cache(method="lazy")` spawns background writer
# threads; reading the cache before they flush emits "Failed to restore
# lazy cache: Incomplete cache: missing blobs". 50 ms is well below the
# hot-path noise floor and large enough to clear the writers in practice.
LAZY_SETTLE_S = 0.05


def samples_ms(fn: Callable[[], Any], runs: int,
               settle_s: float = 0.0) -> np.ndarray:
    """Raw wall-clock samples (ms) of ``fn()`` over ``runs``, after one
    warmup and an optional ``settle_s`` sleep (e.g. to let async writers
    flush before the cache is read)."""
    fn()
    if settle_s:
        time.sleep(settle_s)
    samples = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000
    return samples


def median_ms(fn: Callable[[], Any], runs: int,
              settle_s: float = 0.0) -> float:
    """Median wall-clock (ms) of ``fn()`` over ``runs``."""
    return float(np.median(samples_ms(fn, runs, settle_s)))


def random_payload(size_mb: float) -> np.ndarray:
    """Random float64 numpy array sized to approximately `size_mb` MB."""
    n = max(1, int(size_mb * 1024 ** 2 / 8))
    return np.random.rand(n)


# ---- Cache-hit primitive measurement (fig:cache-eval panel b) ---------------

def _mo_keyhash(payload) -> bytes:
    """Cache key marimo's `persistent_cache` derives for a primitive ref."""
    try:
        buf = mohash.data_to_buffer(payload)
    except Exception:
        buf = pickle.dumps(payload, protocol=5)
    return hashlib.sha256(buf).digest()


def _bench_cache(payload) -> Cache:
    """Cache envelope used to populate the bench loaders.

    A `Pure` cache with a single def `x = payload` and no stateful
    refs is the simplest shape that exercises every step `load_cache`
    has to perform on the real path: `store.get(blob)`, format-specific
    decode, type/stub walk via `_restore_from_stub_if_needed`, and
    `Cache(...)` reconstruction.
    """
    return Cache(
        defs={"x": payload},
        hash="bench",
        cache_type="Pure",
        stateful_refs=set(),
        hit=True,
        meta={},
    )


_BENCH_KEY = HashKey("bench", "Pure")


def _measure_marimo_pickle(payload, runs: int) -> tuple[float, float]:
    """Key derivation + real disk-backed load through `PickleLoader`.

    Key cost is `_mo_keyhash` (`data_to_buffer` + sha256). Load cost
    is the real `PickleLoader.load_cache(key)` codepath, which reads
    the blob from a `FileStore`-backed directory, runs `pickle.loads`,
    type-checks the envelope, and returns the reconstructed `Cache`.
    This includes the disk read that earlier benches skipped by
    measuring `pickle.loads(blob)` on an in-memory bytes object.
    """
    key_ms = median_ms(functools.partial(_mo_keyhash, payload), runs)
    tmp = tempfile.mkdtemp()
    try:
        loader = PickleLoader("bench-pickle", store=FileStore(save_path=tmp))
        if not loader.save_cache(_bench_cache(payload)):
            return key_ms, float("nan")
        load_ms = median_ms(
            functools.partial(loader.load_cache, _BENCH_KEY), runs,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return key_ms, load_ms


def _measure_marimo_lazy_load(payload, fallback_ms: float, runs: int) -> float:
    """Real disk-backed load through `LazyLoader.load_cache`.

    Exercises the full lazy path: `store.get` of the JSONL manifest,
    `msgspec.json.decode` into a `CacheSchema`, a thread pool that
    loads each blob in parallel through `BLOB_DESERIALIZERS`, the
    `queue.Queue` join, and the final `Cache` reconstruction. Falls
    back to the pickle measurement when the payload type has no lazy
    codec registered (the runtime would route the same way), since
    nothing useful is measured by exercising the registry miss path.
    """
    fmt = molazy.maybe_update_lazy_stub(payload)
    if fmt not in ("npy", "arrow"):
        return fallback_ms
    tmp = tempfile.mkdtemp()
    try:
        loader = LazyLoader("bench-lazy", store=LazyStore(FileStore(save_path=tmp)))
        if not loader.save_cache(_bench_cache(payload)):
            return float("nan")
        loader.flush()  # let background writer threads finish before reads
        return median_ms(
            functools.partial(loader.load_cache, _BENCH_KEY), runs,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _measure_mandala(payload, runs: int) -> tuple[float, float]:
    """Mandala-style key + load, disk-backed for parity with marimo.

    Mandala addresses values by `joblib.hash` and persists them with
    `joblib.dump`. The earlier primitive measurement loaded from
    `BytesIO`, which skipped the disk read marimo's `load_cache` does
    pay. Writing the blob to a temp file and timing `joblib.load`
    against a fresh file handle on each iteration brings mandala's
    load measurement onto the same footing.
    """
    key_ms = median_ms(functools.partial(joblib.hash, payload), runs)
    tmp = tempfile.mkdtemp()
    try:
        path = Path(tmp) / "blob.joblib"
        joblib.dump(payload, str(path))

        def load_from_disk():
            with open(path, "rb") as fh:
                return joblib.load(fh)

        load_ms = median_ms(load_from_disk, runs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return key_ms, load_ms


def _measure_diskcache_memoize_total(payload, runs: int) -> float:
    """End-to-end memoize hit (ms), or NaN if the SQLite blob ceiling fails."""
    with tempfile.TemporaryDirectory() as d:
        try:
            cache = diskcache.Cache(d, disk=diskcache.Disk, size_limit=DC_SIZE_LIMIT)
        except Exception:
            return float("nan")
        try:
            memoized = cache.memoize()(identity)
            memoized(payload)  # populate
            return median_ms(functools.partial(memoized, payload), runs)
        except Exception:
            return float("nan")
        finally:
            cache.close()


def _measure_diskcache_fixed_load(payload, runs: int) -> float:
    """Byte-keyed lookup (ms), or NaN if `set` rejects the value."""
    with tempfile.TemporaryDirectory() as d:
        cache = diskcache.Cache(d, disk=diskcache.Disk, size_limit=DC_SIZE_LIMIT)
        try:
            if not cache.set("k", payload) or cache.get("k") is None:
                return float("nan")
            return median_ms(functools.partial(cache.get, "k"), runs)
        finally:
            cache.close()


def measure_hit(payload, runs: int = BENCH_RUNS) -> list[dict]:
    """Decompose cache-hit latency per method into key + load (ms).

    Returns a list of ``{method, stage, ms}`` rows ready to pivot.
    """
    nan = float("nan")

    mo_key, mo_load = _measure_marimo_pickle(payload, runs)
    lazy_load = _measure_marimo_lazy_load(payload, fallback_ms=mo_load, runs=runs)
    md_key, md_load = _measure_mandala(payload, runs)

    memo_total = _measure_diskcache_memoize_total(payload, runs)
    fixed_load = _measure_diskcache_fixed_load(payload, runs)

    # If memoize failed, both stages drop out; otherwise subtract the
    # byte-keyed load to recover the memoize-specific keying cost.
    if memo_total != memo_total:  # NaN check
        memo_key, memo_load = nan, nan
    else:
        memo_load = fixed_load
        memo_key = max(memo_total - fixed_load, 0.0)
    fixed_key = 0.0 if fixed_load == fixed_load else nan

    return [
        {"method": "mo.persistent_cache",        "stage": "key",  "ms": mo_key},
        {"method": "mo.persistent_cache",        "stage": "load", "ms": mo_load},
        {"method": "mo.persistent_cache (lazy)", "stage": "key",  "ms": mo_key},
        {"method": "mo.persistent_cache (lazy)", "stage": "load", "ms": lazy_load},
        {"method": "mandala",                    "stage": "key",  "ms": md_key},
        {"method": "mandala",                    "stage": "load", "ms": md_load},
        {"method": "diskcache.memoize",          "stage": "key",  "ms": memo_key},
        {"method": "diskcache.memoize",          "stage": "load", "ms": memo_load},
        {"method": "diskcache (fixed key)",      "stage": "key",  "ms": fixed_key},
        {"method": "diskcache (fixed key)",      "stage": "load", "ms": fixed_load},
    ]


# ---- Cache-miss primitive measurement (write path) --------------------------
#
# A miss costs the cell body (paid with or without a cache) plus the
# cache's overhead: key derivation and value save. These functions
# measure that overhead so the paper can state a break-even body cost
# rather than assert that "caching is a conditional win".

def _median_save_ms(
    make_loader_and_save: Callable[[str], None], runs: int,
) -> float:
    """Median wall-clock (ms) of one save into a fresh temp dir per run.

    A fresh directory per iteration keeps each save a true first
    write — re-saving over an existing blob would measure the
    overwrite path instead.
    """
    times = np.empty(runs)
    for i in range(runs):
        d = tempfile.mkdtemp()
        try:
            t0 = time.perf_counter()
            make_loader_and_save(d)
            times[i] = (time.perf_counter() - t0) * 1000
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return float(np.median(times))


def measure_miss(payload: Any, runs: int = BENCH_RUNS) -> list[dict]:
    """Decompose cache-miss overhead per method into key + save (ms).

    Returns `{method, stage, ms}` rows shaped exactly like
    `measure_hit` so `stage_decomposition_at` can pivot either.
    The cell body's own cost is excluded by construction: only the
    overhead the cache *adds* to a miss is timed.
    """
    nan = float("nan")
    mo_key = median_ms(functools.partial(_mo_keyhash, payload), runs)
    md_key = median_ms(functools.partial(joblib.hash, payload), runs)

    def save_pickle(d: str) -> None:
        loader = PickleLoader("bench", store=FileStore(save_path=d))
        assert loader.save_cache(_bench_cache(payload))

    def save_lazy(d: str) -> None:
        loader = LazyLoader("bench", store=LazyStore(FileStore(save_path=d)))
        assert loader.save_cache(_bench_cache(payload))
        loader.flush()  # charge the background writers to the save

    def save_joblib(d: str) -> None:
        joblib.dump(payload, str(Path(d) / "blob.joblib"))

    def save_diskcache(d: str) -> None:
        cache = diskcache.Cache(d, disk=diskcache.Disk,
                                size_limit=DC_SIZE_LIMIT)
        try:
            assert cache.set("k", payload)
        finally:
            cache.close()

    rows = [
        {"method": "mo.persistent_cache", "stage": "key", "ms": mo_key},
        {"method": "mo.persistent_cache", "stage": "save",
         "ms": _median_save_ms(save_pickle, runs)},
        {"method": "mo.persistent_cache (lazy)", "stage": "key", "ms": mo_key},
        {"method": "mo.persistent_cache (lazy)", "stage": "save",
         "ms": _median_save_ms(save_lazy, runs)},
        {"method": "mandala", "stage": "key", "ms": md_key},
        {"method": "mandala", "stage": "save",
         "ms": _median_save_ms(save_joblib, runs)},
        {"method": "diskcache (fixed key)", "stage": "key", "ms": 0.0},
        {"method": "diskcache (fixed key)", "stage": "save",
         "ms": _median_save_ms(save_diskcache, runs)},
    ]
    for r in rows:  # propagate failures identically to measure_hit
        if r["ms"] != r["ms"]:
            r["ms"] = nan
    return rows


def sweep_write_overhead(
    sizes_mb: tuple[float, ...],
) -> list[dict]:
    """Total miss overhead (key + save, ms) per (size, method).

    Rows: `[{size_mb, method, ms}]` — `sweep_pivot` renders it
    directly. Used for the break-even curve in fig:cache-eval(a).
    """
    rows: list[dict] = []
    for s in sizes_mb:
        payload = random_payload(s)
        for r in measure_miss(payload):
            rows.append({"size_mb": s, "method": r["method"],
                         "stage": r["stage"], "ms": r["ms"]})
        del payload
    return rows


def sweep_pivot(
    rows: list[dict],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tidy rows -> ``(sizes_mb, {method: total_ms_array})``.

    A NaN in any stage of a (method, size_mb) cell propagates to that
    cell's total — so a method that fails at large payloads simply
    disappears from the chart instead of being silently zeroed.
    """
    sizes_mb = sorted({r["size_mb"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    size_idx = {s: i for i, s in enumerate(sizes_mb)}

    totals = {m: np.zeros(len(sizes_mb)) for m in methods}
    saw_nan = {m: np.zeros(len(sizes_mb), dtype=bool) for m in methods}

    for r in rows:
        i = size_idx[r["size_mb"]]
        ms = r["ms"]
        if ms != ms:  # NaN
            saw_nan[r["method"]][i] = True
        else:
            totals[r["method"]][i] += ms

    for m in methods:
        totals[m][saw_nan[m]] = np.nan

    return np.array(sizes_mb), totals


def stage_decomposition_at(
    rows: list[dict],
    size_mb: float,
) -> dict[str, tuple[float, float]]:
    """Extract ``{method: (key_ms, load_ms)}`` for a single size.

    Methods that NaN'd at this size are dropped from the result so
    callers can render the surviving columns directly.
    """
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        if r["size_mb"] != size_mb:
            continue
        key, load = out.get(r["method"], (0.0, 0.0))
        if r["stage"] == "key":
            key = r["ms"]
        else:
            load = r["ms"]
        out[r["method"]] = (key, load)
    return {m: kv for m, kv in out.items() if kv[0] == kv[0] and kv[1] == kv[1]}


# ---- End-to-end benchmark (fig8) -------------------------------------------
#
# The figure cell builds a `setup_methods()` factory over its marimo /
# mandala / diskcache fixtures. The factory returns either a plain
# `{name: bind}` dict or a `(methods, cleanup_fn)` tuple. Each
# `bind(payload)` does per-payload preparation (warm the cache, set a
# byte-key, prime a memoize wrapper) and returns the 0-arg callable to
# time — optionally wrapped in `_Bound` to carry per-binding cleanup
# (close diskcache handles, rm tempdirs). The sweep loop owns *all*
# timing and explicitly calls every cleanup, so reruns do not leak
# SQLite connections or temp directories.

class _Bound:
    """Pairs a 0-arg cache-hit callable with its per-binding cleanup.

    Used by the diskcache bind factories so the sweep can close
    `Cache` handles and remove temp dirs deterministically after each
    (size, method) measurement. `_Bound` instances are themselves
    callable; sweeps that don't recognize the type still get the
    timing behaviour, but resources are then never freed.
    """

    __slots__ = ("_call", "_cleanup")

    def __init__(self, call: Callable[[], Any],
                 cleanup: Callable[[], None] | None = None):
        self._call = call
        self._cleanup = cleanup

    def __call__(self):
        return self._call()

    def close(self) -> None:
        cleanup, self._cleanup = self._cleanup, None
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass


def sweep_methods_samples(
    sizes_mb: tuple[float, ...],
    setup_methods: Callable[[], Any],
    *,
    settle_s: float = LAZY_SETTLE_S,
) -> list[dict]:
    """End-to-end sweep returning raw samples per (size, method).

    ``setup_methods()`` returns either:

    - ``{name: bind}`` — bind factories only; nothing to clean up at
      session level
    - ``(methods, cleanup)`` — a 2-tuple where ``cleanup()`` runs once
      after the sweep, used e.g. to exit a long-lived mandala
      ``Storage`` context

    ``bind(payload)`` returns either a 0-arg cache-hit callable (timed
    by the sweep), a ``_Bound`` instance wrapping that callable with a
    per-binding cleanup, or ``None`` to signal "this method cannot
    handle this payload" (e.g. ``diskcache.memoize`` past the SQLite
    blob ceiling). NaN samples propagate so downstream pivots drop the
    (size, method) cell from the plot.

    Rows: ``[{size_mb, method, samples: np.ndarray}]``.
    """
    setup = setup_methods()
    if isinstance(setup, tuple):
        methods, session_cleanup = setup
    else:
        methods, session_cleanup = setup, None

    rows: list[dict] = []
    try:
        for s in sizes_mb:
            payload = random_payload(s)
            runs = BENCH_RUNS
            for name, bind in methods.items():
                try:
                    hit = bind(payload)
                except Exception:
                    hit = None
                try:
                    if hit is None:
                        samples = np.full(runs, np.nan)
                    else:
                        samples = samples_ms(hit, runs, settle_s=settle_s)
                finally:
                    if hit is not None and hasattr(hit, "close"):
                        hit.close()
                rows.append({"size_mb": s, "method": name, "samples": samples})
            del payload
    finally:
        if session_cleanup is not None:
            try:
                session_cleanup()
            except Exception:
                pass
    return rows


def medians_from_samples_rows(rows: list[dict]) -> list[dict]:
    """Reduce samples rows to ``{size_mb, method, ms}`` (one median per
    cell) so ``sweep_pivot`` can render a line plot."""
    return [
        {"size_mb": r["size_mb"], "method": r["method"],
         "ms": float(np.median(r["samples"]))}
        for r in rows
    ]


def samples_at_size(rows: list[dict], size_mb: float) -> dict[str, np.ndarray]:
    """Pluck ``{method: samples_array}`` for one size from samples rows."""
    return {r["method"]: r["samples"] for r in rows if r["size_mb"] == size_mb}


# ---- diskcache bind factories (used by the e2e cell's setup_methods) ----

def _close_cache_and_rmtree(cache, directory: str) -> None:
    try:
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def bind_diskcache_memoize(payload: Any) -> _Bound | None:
    """Per-payload bind for ``diskcache.Cache.memoize()``. Allocates a
    fresh temp dir, primes the cache with one call, and returns a
    ``_Bound`` holding the hit callable plus the cleanup that closes
    the cache and removes the dir. Returns ``None`` if the SQLite
    blob ceiling rejects the warm-up (so the sweep records NaN samples)."""
    d = tempfile.mkdtemp()
    try:
        cache = diskcache.Cache(d, disk=diskcache.Disk, size_limit=DC_SIZE_LIMIT)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        return None
    try:
        memoized = cache.memoize()(identity)
        memoized(payload)
    except Exception:
        _close_cache_and_rmtree(cache, d)
        return None
    return _Bound(
        functools.partial(memoized, payload),
        cleanup=functools.partial(_close_cache_and_rmtree, cache, d),
    )


def bind_diskcache_fixed(payload: Any) -> _Bound | None:
    """Per-payload bind for a byte-keyed ``diskcache.Cache.get('k')``.
    Stores ``payload`` once under a fixed key and returns a ``_Bound``
    holding the get callable + cache/dir cleanup. Returns ``None`` if
    ``set`` rejects the value."""
    d = tempfile.mkdtemp()
    try:
        cache = diskcache.Cache(d, disk=diskcache.Disk, size_limit=DC_SIZE_LIMIT)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        return None
    if not cache.set("k", payload) or cache.get("k") is None:
        _close_cache_and_rmtree(cache, d)
        return None
    return _Bound(
        functools.partial(cache.get, "k"),
        cleanup=functools.partial(_close_cache_and_rmtree, cache, d),
    )


def bind_warm(fn: Callable[[Any], Any]) -> Callable[[Any], Callable[[], Any]]:
    """Trivial bind factory for caches whose API is just ``fn(payload)``:
    one warm call to populate the cache, then a 0-arg hit closure.
    Used for ``mo.cache`` / ``mo.persistent_cache`` and (with a wrapping
    ``with storage:``) mandala."""
    def bind(payload):
        fn(payload)
        return functools.partial(fn, payload)
    return bind


# ---- Plot helpers -----------------------------------------------------------

def plot_stage_decomposition(
    ax,
    stage_ms: dict[str, tuple[float, float]],
    *,
    miss_ms: dict[str, tuple[float, float]] | None = None,
    title: str = "(b) Stage decomposition",
) -> None:
    """fig7b: stacked horizontal bars — key + load on a hit, and
    (when `miss_ms` is given) key + save overhead on a miss, paired
    per method."""
    names = list(stage_ms.keys())
    keys = np.array([stage_ms[n][0] for n in names]) / 1000.0
    loads = np.array([stage_ms[n][1] for n in names]) / 1000.0
    y = np.arange(len(names))
    height = 0.38 if miss_ms else 0.7
    off = height / 2 if miss_ms else 0.0
    ax.barh(y - off, keys, height=height, color="#1f77b4",
            label="key derivation")
    ax.barh(y - off, loads, height=height, left=keys, color="#aec7e8",
            label="value load (hit)")
    if miss_ms:
        mkeys = np.array(
            [miss_ms.get(n, (np.nan, np.nan))[0] for n in names]) / 1000.0
        msaves = np.array(
            [miss_ms.get(n, (np.nan, np.nan))[1] for n in names]) / 1000.0
        ax.barh(y + off, mkeys, height=height, color="#1f77b4")
        ax.barh(y + off, msaves, height=height, left=mkeys,
                color="#ffbb78", label="value save (miss)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=6.5)


# Method ordering used by every multi-method panel — keeps the legend
# (and any side-by-side panel) consistent. Mirrors the molab notebook's
# `_e2e_order`.
E2E_METHOD_ORDER: tuple[str, ...] = (
    "diskcache.memoize",
    "mandala",
    "mo.persistent_cache",
    "mo.persistent_cache (lazy)",
    "mo.cache",
    "diskcache (fixed key)",
)


def plot_e2e(
    ax,
    e2e: dict[str, np.ndarray],
    sizes_mb: np.ndarray,
    *,
    palette: dict[str, str] | None = None,
    threshold_ms: float = 100.0,
    host_label: str | None = None,
) -> None:
    """fig:cache-eval panel (a): end-to-end cell-rerun cost on a cache hit, log-log."""
    p = palette or PALETTE
    for name in E2E_METHOD_ORDER:
        if name not in e2e:
            continue
        vals = e2e[name]
        finite = ~np.isnan(vals)
        if not finite.any():
            continue
        ax.loglog(
            sizes_mb[finite], vals[finite],
            "o-", label=name, color=p.get(name, "gray"),
            markersize=4, lw=1.2,
        )
    ax.axhline(threshold_ms, ls="--", color="gray", lw=0.8, alpha=0.6)
    # Sit the label in the clear gap just above the line — centered on the
    # geometric-mid payload, right of the upper-left legend and left of where
    # the curves cross the threshold on the right, so it never overlaps either.
    ax.text(
        float(np.sqrt(sizes_mb.min() * sizes_mb.max())), threshold_ms * 1.25,
        f"{threshold_ms:.0f} ms interactive threshold",
        fontsize=7, color="gray", style="italic", ha="center", va="bottom",
    )
    ax.set_xlabel("Payload size (MB)")
    ax.set_ylabel("Cache-hit latency (ms)")
    ax.set_title("(a) End-to-end cache-hit latency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=6.5)
    if host_label:
        ax.text(
            0.99, 0.02, host_label, transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=6, color="gray", style="italic",
        )


def plot_e2e_box(
    ax,
    samples_by_method: dict[str, np.ndarray],
    *,
    palette: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """fig8b: per-method box-plot of cache-hit samples at one payload size.

    NaN-filled series (failure modes) are dropped from the panel
    automatically — they show up as a gap in panel (a) anyway.
    """
    p = palette or PALETTE
    names, data, colors = [], [], []
    for name in E2E_METHOD_ORDER:
        if name not in samples_by_method:
            continue
        s = samples_by_method[name]
        finite = s[~np.isnan(s)]
        if finite.size == 0:
            continue
        names.append(name)
        data.append(finite)
        colors.append(p.get(name, "gray"))

    bp = ax.boxplot(
        data, vert=True, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", lw=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Cache-hit latency (ms)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


# ---- Walked example (fig2) --------------------------------------------------
#
# A four-cell DAG used in §3.2 to illustrate the dispatch in action.
# Every hash in the figure is produced by marimo's real `BlockHasher`
# (`marimo._save.hash`) over a real compiled cell graph — nothing is
# simulated. The model class stands in for any opaque object a cell
# can produce (a torch `nn.Module`, a database handle, ...).
#
#   a: seed = «v»            — leaf constant (Pure key)
#   b: x = rng(seed)...      — ContentAddressed (ndarray buffer)
#   c: model = TinyNet()     — module-pinned refs only; its *product*
#                              is unhashable
#   d: y = model(x)          — ExecutionPath on `model`, content on `x`
#
# Edges encode dataflow refs: a→b (seeds the array), b→d (model input),
# c→d (the model itself). Cell `c` is independent of the seed chain;
# editing the seed invalidates a, b, d but not c.

WALKED_SOURCES: dict[str, str] = {
    "imports": "import numpy as np",
    "a": "seed = 7",  # rewritten per panel by compute_walked_state
    "b": "x = np.random.default_rng(seed).standard_normal(64)",
    "c": (
        "class TinyNet:\n"
        "    def __init__(self):\n"
        "        self.w = np.ones(64) / 64.0\n"
        "    def __call__(self, x):\n"
        "        return float(self.w @ x)\n"
        "model = TinyNet()"
    ),
    "d": "y = model(x)",
}

WALKED_EDGES: tuple[tuple[str, str], ...] = (
    ("a", "b"), ("b", "d"), ("c", "d"),
)
WALKED_POSITIONS: dict[str, tuple[float, float]] = {
    "a": (0.16, 0.82),
    "b": (0.16, 0.46),
    "c": (0.68, 0.82),
    "d": (0.46, 0.16),
}
WALKED_LABELS: dict[str, str] = {
    "a": "a · seed",
    "b": "b · x = rng(seed)",
    "c": "c · model = TinyNet()",
    "d": "d · y = model(x)",
}


def _walked_hashes(
    sources: dict[str, str],
    order: tuple[str, ...] = ("imports", "a", "b", "c", "d"),
    ids: dict[str, str] | None = None,
    scope_mut: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Compile `sources` into a real marimo graph and key every cell.

    Uses the same `BlockHasher` that backs `mo.persistent_cache`.
    `external=True` keeps the synthetic graph out of the live runtime's
    registries. `ids` remaps cell ids (to demonstrate id-invariance)
    and `scope_mut` mutates the executed scope before hashing (to
    demonstrate mutation visibility).
    """
    from marimo._ast import compiler
    from marimo._runtime.dataflow import DirectedGraph
    from marimo._save.hash import BlockHasher

    ids = ids or {k: k for k in sources}
    graph = DirectedGraph()
    cells = {}
    for cid in order:
        cell = compiler.compile_cell(sources[cid], cell_id=ids[cid])
        cells[cid] = cell
        graph.register_cell(ids[cid], cell)
    scope: dict[str, Any] = {}
    for cid in order:
        exec(sources[cid], scope)  # noqa: S102 — synthetic 5-line cells
    scope.pop("__builtins__", None)
    if scope_mut is not None:
        scope_mut(scope)
    state = {}
    for cid in ("a", "b", "c", "d"):
        hasher = BlockHasher(
            module=cells[cid].mod, graph=graph, cell_id=ids[cid],
            scope=dict(scope), external=True,
        )
        state[cid] = {"h": hasher.hash[:6], "kind": hasher.cache_type}
    return state


def compute_walked_state(seed_value: int) -> dict[str, dict]:
    """Key the four-cell DAG at one seed with marimo's real hasher.

    Returns `{cell: {"h": <6-char key prefix>, "kind": <dispatch branch>}}`.
    """
    sources = dict(WALKED_SOURCES)
    sources["a"] = f"seed = {seed_value}"
    return _walked_hashes(sources)


def hash_invariance_report() -> dict[str, bool]:
    """Measured invariance properties of the real cache key derivation.

    Each entry is a claim quoted in the paper, evaluated against
    marimo's `BlockHasher` on the walked-example graph. `True` means
    the property held on this host at build time.
    """
    base = _walked_hashes(WALKED_SOURCES)

    def variant(**overrides: str) -> dict[str, dict]:
        return _walked_hashes({**WALKED_SOURCES, **overrides})

    comment = variant(b="# a cosmetic comment\n" + WALKED_SOURCES["b"])
    ws = variant(b="x = np.random.default_rng( seed ).standard_normal(64)")
    priv1 = variant(b="_rng = np.random.default_rng(seed)\nx = _rng.standard_normal(64)")
    priv2 = variant(b="_gen = np.random.default_rng(seed)\nx = _gen.standard_normal(64)")
    rename = variant(
        b="x2 = np.random.default_rng(seed).standard_normal(64)",
        d="y = model(x2)",
    )
    reordered = _walked_hashes(
        WALKED_SOURCES, order=("imports", "c", "a", "b", "d"))
    new_ids = _walked_hashes(
        WALKED_SOURCES,
        ids={"imports": "Zq1", "a": "Yx2", "b": "Wv3", "c": "Ut4", "d": "Sr5"})

    def mut_x(scope: dict) -> None:
        scope["x"][0] += 100.0

    def mut_model(scope: dict) -> None:
        scope["model"].w = scope["model"].w * 0.0

    mutated_x = _walked_hashes(WALKED_SOURCES, scope_mut=mut_x)
    mutated_model = _walked_hashes(WALKED_SOURCES, scope_mut=mut_model)

    return {
        # robustness to superficial edits (bytecode-level hashing)
        "comment_insensitive": comment == base,
        "whitespace_insensitive": ws == base,
        "reorder_insensitive": reordered == base,
        "cell_id_insensitive": new_ids == base,
        # correct invalidation
        "idempotent": _walked_hashes(WALKED_SOURCES) == base,
        "public_rename_invalidates": rename["d"]["h"] != base["d"]["h"],
        "consumer_sees_inplace_mutation": mutated_x["d"]["h"] != base["d"]["h"],
        # measured boundaries (False would be the surprise; these are
        # the known limitations quoted in the paper)
        "private_rename_invalidates": priv1 != priv2,
        "opaque_attr_mutation_invisible": (
            mutated_model["d"]["h"] == base["d"]["h"]
        ),
    }


def plot_walked_example(states, titles, *, fig=None) -> Any:
    """Render the three-panel PyTorch walked example.

    Box fill colour matches the dispatch palette used in
    {ref}`fig:dispatch` — Pure is blue, ContentAddressed green,
    ExecutionPath orange. State (recomputed this panel vs reused
    from cache) is encoded in the box border: thick solid for
    recompute, thin dashed for cached. Edges (parent → child) are
    coloured by what propagated: red when the parent's hash changed,
    green when the parent is stable, gray for the initial panel.
    """
    import matplotlib.pyplot as plt
    # Dispatch-branch palette (matches figs/fig1_dispatch.dot).
    KIND_FILL = {
        "Pure":             "#e3f2fd",
        "ContentAddressed": "#e8f5e9",
        "ExecutionPath":    "#fff3e0",
    }
    KIND_LABEL = {
        "Pure":             "#1565c0",
        "ContentAddressed": "#2e7d32",
        "ExecutionPath":    "#ef6c00",
    }
    # Edge colours encode propagation state, NOT category.
    edge_initial  = "#888888"
    edge_changed  = "#d32f2f"
    edge_stable   = "#388e3c"

    if fig is None:
        fig, axes = plt.subplots(1, len(states), figsize=(8.6, 3.0))
    else:
        axes = fig.subplots(1, len(states))
    if len(states) == 1:
        axes = [axes]

    # Per-cell box half-extents; the arrow endpoints back off by these
    # so arrowheads land at the box border, not inside the text.
    box_dx, box_dy = 0.13, 0.075

    def _edge_endpoints(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return (x1, y1), (x2, y2)
        # Find where the unit vector from p1 toward p2 exits each box.
        def _shrink(x, y, sx, sy):
            tx = box_dx / abs(sx) if sx else float("inf")
            ty = box_dy / abs(sy) if sy else float("inf")
            t = min(tx, ty)
            return x + sx * t, y + sy * t
        norm = (dx ** 2 + dy ** 2) ** 0.5
        ux, uy = dx / norm, dy / norm
        start = _shrink(x1, y1, ux, uy)
        end   = _shrink(x2, y2, -ux, -uy)
        return start, end

    for i, (ax, state, title) in enumerate(zip(axes, states, titles)):
        prior = states[i - 1] if i > 0 else None
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)

        for parent, child in WALKED_EDGES:
            start, end = _edge_endpoints(
                WALKED_POSITIONS[parent], WALKED_POSITIONS[child],
            )
            if prior is None:
                color = edge_initial
            elif prior[parent]["h"] != state[parent]["h"]:
                color = edge_changed
            else:
                color = edge_stable
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
            )

        for name, info in state.items():
            x, y = WALKED_POSITIONS[name]
            changed = prior is None or prior[name]["h"] != info["h"]
            kind = info["kind"]
            status = "recompute" if changed else "cached"
            label = WALKED_LABELS[name]
            ax.text(
                x, y, f"{label}\nH={info['h']}\n[{status}]",
                ha="center", va="center", fontsize=7.5,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=KIND_FILL[kind],
                    edgecolor="#222" if changed else "#888",
                    lw=1.4 if changed else 0.5,
                    linestyle="solid" if changed else (0, (3, 2)),
                ),
            )
            ax.text(
                x, y - box_dy - 0.04, kind,
                ha="center", va="top",
                fontsize=6.5, color=KIND_LABEL[kind], style="italic",
                fontweight="bold",
            )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    edge_handles = [
        Line2D([0], [0], color=edge_changed, lw=2, label="parent hash changed"),
        Line2D([0], [0], color=edge_stable,  lw=2, label="parent hash stable"),
        Line2D([0], [0], color=edge_initial, lw=2, label="initial (no prior)"),
    ]
    kind_handles = [
        Patch(facecolor=KIND_FILL["Pure"],
              edgecolor=KIND_LABEL["Pure"], label="Pure"),
        Patch(facecolor=KIND_FILL["ContentAddressed"],
              edgecolor=KIND_LABEL["ContentAddressed"], label="ContentAddressed"),
        Patch(facecolor=KIND_FILL["ExecutionPath"],
              edgecolor=KIND_LABEL["ExecutionPath"], label="ExecutionPath"),
    ]
    fig.legend(
        handles=kind_handles + edge_handles, loc="lower center",
        ncol=6, frameon=False, fontsize=7.0,
        bbox_to_anchor=(0.5, -0.04),
    )
    return fig
