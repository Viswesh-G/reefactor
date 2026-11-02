"""A bounded in-memory LRU with single-flight downloads across fetch workers."""
import threading
from collections import OrderedDict
from concurrent.futures import Future


class ByteCache:
    """No lock is held during network I/O or while waiting for another loader.

    Loaders must not call this cache recursively: one loader performs one HTTP
    request, preventing dependency cycles between in-flight keys.
    """
    def __init__(self, maximum_bytes):
        if maximum_bytes < 0:
            raise ValueError("Cache budget must be nonnegative")
        self.maximum = maximum_bytes
        self.items, self.inflight = OrderedDict(), {}
        self.bytes, self.peak_bytes, self.hits, self.misses = 0, 0, 0, 0
        self.lock = threading.Lock()

    def get(self, key, loader):
        if not self.maximum:
            return loader(), False
        with self.lock:
            if key in self.items:
                self.hits += 1
                self.items.move_to_end(key)
                return self.items[key], True
            future = self.inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self.inflight[key] = future
                self.misses += 1
            else:
                self.hits += 1
        if not owner:
            return future.result(), True
        try:
            value = loader()
            with self.lock:
                if len(value) <= self.maximum:
                    while self.items and self.bytes+len(value) > self.maximum:
                        _, previous = self.items.popitem(last=False)
                        self.bytes -= len(previous)
                    self.items[key] = value
                    self.bytes += len(value)
                    self.peak_bytes = max(self.peak_bytes, self.bytes)
                self.inflight.pop(key)
            future.set_result(value)
            return value, False
        except BaseException as exc:
            with self.lock:
                self.inflight.pop(key, None)
            future.set_exception(exc)
            raise
