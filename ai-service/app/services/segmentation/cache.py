"""Small thread-safe LRU caches for segmentation results and query embeddings.

Keys always include the image's SHA-256 and the configuration that influences the result, so a
changed setting or model version can never return a stale entry.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, maxsize: int = 64):
        self.maxsize = max(0, int(maxsize))
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        if self.maxsize == 0:
            return
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        return {"size": len(self._d), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses}
