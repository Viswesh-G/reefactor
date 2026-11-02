"""Single-writer DuckDB journal; Polars scans small Parquet metadata snapshots."""
import json
import os
import threading
import time
import sys
from contextlib import contextmanager
from pathlib import Path

import duckdb
import polars as pl


def peak_rss_bytes():
    """Process high-water memory; includes NumPy/native libraries, unlike tracemalloc."""
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value*1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not read(process(), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize


@contextmanager
def exclusive_run(root):
    """OS releases this lock even if the process dies. Never delete a live lock."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
        except PermissionError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        try:
            yield
        finally:
            unlock()


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.owner = threading.get_ident()
        self.db = duckdb.connect(str(self.root / "observability.duckdb"))
        self.db.execute("CREATE TABLE IF NOT EXISTS events (timestamp_utc DOUBLE, job VARCHAR, state VARCHAR, details VARCHAR)")
        self.db.execute("CREATE TABLE IF NOT EXISTS partitions (job VARCHAR PRIMARY KEY, path VARCHAR, bytes BIGINT, metadata VARCHAR)")

    def event(self, job, state, **details):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the coordinator can write the journal")
        self.db.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                        [time.time(), job, state, json.dumps(details, sort_keys=True)])

    def commit(self, job, path, size, metadata):
        self.db.execute("INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?)",
                        [job, str(path), size, json.dumps(metadata, sort_keys=True)])

    def snapshot(self):
        # Snapshot readers need no DuckDB write lock and never scan the scientific arrays.
        for table in ("events", "partitions"):
            frame = self.db.execute(f"SELECT * FROM {table}").pl()
            temporary = self.root / f"{table}.tmp.parquet"
            frame.write_parquet(temporary)
            os.replace(temporary, self.root / f"{table}.parquet")

    def close(self):
        self.snapshot()
        self.db.close()


def summarize(root):
    root = Path(root)
    events = pl.scan_parquet(root / "events.parquet")
    partitions = pl.scan_parquet(root / "partitions.parquet")
    return {"states": events.group_by("state").len().collect().to_dicts(),
            "storage": partitions.select(pl.len().alias("partitions"), pl.col("bytes").sum()).collect().to_dicts()}
