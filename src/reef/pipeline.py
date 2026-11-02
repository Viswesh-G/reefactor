"""Bounded concurrent fetches, a checked FSM, and one atomic partition writer."""
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from functools import partial

import xarray as xr
from numcodecs import Blosc

from .harmonize import validate_partition
from .observability import Journal, exclusive_run, peak_rss_bytes
from .planning import fingerprint, iter_jobs, make_plan, source_names
from .sources import fetch_job, preflight
from .cache import ByteCache

TRANSITIONS = {
    "INIT": {"PREFLIGHT", "ERROR"}, "PREFLIGHT": {"RUNNING", "ERROR"},
    "RUNNING": {"READY", "ERROR"}, "READY": set(), "ERROR": set(),
    "PENDING": {"FETCHING_ALIGNING", "REUSED", "FAILED"},
    "FETCHING_ALIGNING": {"VALIDATING", "FAILED"}, "VALIDATING": {"STAGING", "FAILED"},
    "STAGING": {"COMMITTED", "FAILED"}, "COMMITTED": set(), "REUSED": set(), "FAILED": set(),
}


class StateMachine:
    def __init__(self, initial, emit):
        self.state, self.emit = initial, emit

    def move(self, state, **details):
        if state not in TRANSITIONS[self.state]:
            raise RuntimeError(f"Illegal state transition: {self.state} -> {state}")
        self.state = state
        self.emit(state, **details)


def atomic_replace(source, target):
    """Windows file scanners can briefly hold a just-written directory open."""
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1*2**attempt)


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path):
    return {file.relative_to(path).as_posix(): {"bytes": file.stat().st_size, "sha256": file_hash(file)}
            for file in sorted(path.rglob("*")) if file.is_file() and file.name != "_SUCCESS.json"}


def verify_partition(path, identity):
    marker = path / "_SUCCESS.json"
    if not marker.exists():
        raise RuntimeError(f"Uncommitted partition at {path}; move it aside before retry")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata["fingerprint"] != identity or inventory(path) != metadata["files"]:
        raise RuntimeError(f"Partition integrity/identity mismatch at {path}; refusing silent reuse")
    return metadata


def store_partition(ds, root, job, identity, provenance, metrics, remaining_bytes):
    target = root / "tiles" / f"{job.key}.zarr"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{job.key}_", suffix=".partial", dir=target.parent) as temporary:
        stage = Path(temporary) / "tile.zarr"
        ds = ds.chunk({dim: min(ds.sizes[dim], 16 if dim == "time" else 64) for dim in ds.dims})
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        enc = {name: {"compressor": compressor} for name in ds.data_vars}
        ds.attrs = {"fingerprint": identity, "job": json.dumps(asdict(job)),
                    "provenance": json.dumps(provenance), "grid_semantics": "cell_centers; UTC; no temporal gap fill"}
        ds.to_zarr(stage, mode="w", consolidated=True, encoding=enc, zarr_format=2)
        files = inventory(stage)
        size = sum(v["bytes"] for v in files.values())
        if size > remaining_bytes:
            raise RuntimeError("Partition would exceed max_output_mb")
        with xr.open_zarr(stage, consolidated=True) as check:
            if dict(check.sizes) != dict(ds.sizes) or set(check.data_vars) != set(ds.data_vars):
                raise RuntimeError("Zarr read-back schema mismatch")
        metadata = {"fingerprint": identity, "job": asdict(job), "files": files, "bytes": size,
                    "provenance": provenance, "metrics": metrics}
        write_json_atomic(stage / "_SUCCESS.json", metadata)
        if target.exists():
            raise RuntimeError("Refusing to overwrite an existing immutable partition")
        atomic_replace(stage, target)
    return target, metadata


def run_pipeline(cfg, fetcher=fetch_job, preflight_check=preflight):
    started = time.monotonic()
    plan = make_plan(cfg)
    identity, names = plan["fingerprint"], source_names(cfg)
    root = Path(cfg["storage"]["root"]).resolve() / identity[:16]
    cache = ByteCache(int(cfg["execution"].get("remote_cache_mb", 0)*1024**2))
    if fetcher is fetch_job:
        fetcher = partial(fetch_job, cache=cache)
    with exclusive_run(root):
        journal = Journal(root)
        run = StateMachine("INIT", lambda state, **kw: journal.event("RUN", state, **kw))
        scratch = root / "scratch"
        scratch.mkdir(exist_ok=True)
        # Only remove this run's scratch/staging leftovers after acquiring its OS lock.
        # Verify resolved paths before any recursive removal (also on Windows).
        leftovers = list(scratch.iterdir()) + list((root / "tiles").glob("*.partial"))
        for path in leftovers:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Unsafe scratch path; refusing cleanup")
            if path.is_dir():
                shutil.rmtree(path)
        manifest = {"fingerprint": identity, "plan": plan, "status": "RUNNING", "partitions": []}
        futures, machines = {}, {}
        used_bytes = 0
        try:
            run.move("PREFLIGHT")
            preflight_check(cfg, names)
            write_json_atomic(root / "plan.json", plan)
            run.move("RUNNING")
            jobs = iter(iter_jobs(cfg))
            with ThreadPoolExecutor(max_workers=cfg["execution"]["workers"], thread_name_prefix="reef-fetch") as pool:
                def submit_next():
                    nonlocal used_bytes
                    for job in jobs:
                        machine = StateMachine("PENDING", lambda state, _key=job.key, **kw: journal.event(_key, state, **kw))
                        machines[job.key] = machine
                        target = root / "tiles" / f"{job.key}.zarr"
                        if target.exists():
                            metadata = verify_partition(target, identity)
                            used_bytes += metadata["bytes"]
                            journal.commit(job.key, target, metadata["bytes"], metadata)
                            machine.move("REUSED")
                            manifest["partitions"].append(str(target.relative_to(root)))
                            continue
                        machine.move("FETCHING_ALIGNING")
                        futures[pool.submit(fetcher, job, cfg, names, scratch)] = job
                        return True
                    return False

                for _ in range(cfg["execution"]["workers"]):
                    submit_next()
                try:
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = futures.pop(future)
                            machine = machines[job.key]
                            ds, provenance = future.result()
                            try:
                                machine.move("VALIDATING")
                                ds, metrics = validate_partition(ds, names, cfg["execution"]["max_missing_fraction"])
                                machine.move("STAGING")
                                target, metadata = store_partition(ds, root, job, identity, provenance, metrics,
                                    cfg["execution"]["max_output_mb"]*1024**2-used_bytes)
                                used_bytes += metadata["bytes"]
                                journal.commit(job.key, target, metadata["bytes"], metadata)
                                machine.move("COMMITTED", bytes=metadata["bytes"], metrics=metrics)
                                manifest["partitions"].append(str(target.relative_to(root)))
                                # Amortize metadata work; filesystem markers remain the resume authority.
                                if len(manifest["partitions"]) % 64 == 0:
                                    write_json_atomic(root / "manifest.json", manifest)
                                    journal.snapshot()
                            finally:
                                ds.close()
                            submit_next()
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
            manifest["partitions"].sort()
            manifest["status"] = "READY"
            write_json_atomic(root / "manifest.json", manifest)
            run.move("READY", partitions=len(manifest["partitions"]), bytes=used_bytes,
                     ram_cache_hits=cache.hits, ram_cache_misses=cache.misses, ram_cache_peak_bytes=cache.peak_bytes,
                     elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss_bytes())
        except BaseException as exc:
            for machine in machines.values():
                if "FAILED" in TRANSITIONS[machine.state]:
                    machine.move("FAILED", error_type=type(exc).__name__)
            manifest["status"] = "ERROR"
            write_json_atomic(root / "manifest.json", manifest)
            run.move("ERROR", error_type=type(exc).__name__)
            raise
        finally:
            journal.close()
    return root


def open_cube(root, allow_partial=False):
    """Lazy virtual cube; one physical copy of each partition, no monolithic rewrite."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "READY" and not allow_partial:
        raise ValueError("Run is incomplete; partial data access must be explicit")
    paths = [root / p for p in manifest["partitions"]]
    if not paths:
        raise ValueError("No committed partitions")
    return xr.open_mfdataset(paths, engine="zarr", chunks={}, combine="by_coords",
                             join="exact", compat="no_conflicts", combine_attrs="drop_conflicts")
