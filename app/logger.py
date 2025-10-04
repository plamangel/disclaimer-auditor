import time
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        dt = int((time.time() - t0) * 1000)
        print(f"[timed] {label}: {dt} ms")