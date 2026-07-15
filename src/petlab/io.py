"""Generic I/O helpers: JSON round-tripping, parallel subprocess execution,
memoization, and time-series resampling.

Ported from ``src/utils/utilities.py`` without behavior changes -- this
module only holds what's genuinely generic; the deck-templating and
distribution-sampling functions that used to live alongside these moved to
``petlab.deck``.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import deque
from functools import lru_cache, wraps
from subprocess import DEVNULL

import numpy as np
import scipy.interpolate as interp
from tqdm import tqdm


def read_json(jsonfilename: str) -> dict:
    with open(jsonfilename, "r") as j:
        return json.loads(j.read())


def save_to_json(filename: str, d: dict) -> None:
    with open(filename, "w") as f:
        json.dump(d, f, indent=4)


def resample(base_date, custom_date, data):
    f = interp.interp1d(custom_date, data, fill_value="extrapolate")
    return f(base_date)


def run_bash_commands_in_parallel(commands, max_tries: int, n_parallel: int) -> list[bool]:
    """Run a list of subprocess commands with a bounded pool of workers.

    Returns a list of booleans (one per command) reporting whether each
    process exited with code 0.
    """
    waiting = deque([(command, 1, i) for i, command in enumerate(commands)])
    running = deque()
    is_success = [False] * len(waiting)

    pbar = tqdm(total=len(waiting), disable=False)
    while len(waiting) > 0 or len(running) > 0:
        while len(waiting) > 0 and len(running) < n_parallel:
            command, tries, index = waiting.popleft()
            try:
                running.append((subprocess.Popen(command, stdout=DEVNULL), command, tries, index))
                pbar.set_description(f"Running: {len(running)}, Waiting: {len(waiting)}")
            except OSError:
                print(f"Failed to start command {command}")

        for _ in range(len(running)):
            process, command, tries, index = running.popleft()
            ret = process.poll()
            if ret is None:
                running.append((process, command, tries, index))
            elif ret != 0:
                if tries < max_tries:
                    waiting.append((command, tries + 1, index))
                else:
                    print(f"Command: {command} errored after {max_tries} tries")
                    pbar.update(1)
            else:
                is_success[index] = True
                pbar.update(1)

        time.sleep(0.5)

    pbar.set_description("All simulations done")
    pbar.close()
    return is_success


def hashable_lru(func):
    """LRU-cache a function whose args/kwargs may contain lists/dicts, by
    JSON-serializing them for the cache key."""

    cache = lru_cache(maxsize=1024)

    def deserialise(value):
        try:
            return json.loads(value)
        except Exception:
            return value

    def func_with_serialized_params(*args, **kwargs):
        _args = tuple(deserialise(arg) for arg in args)
        _kwargs = {k: deserialise(v) for k, v in kwargs.items()}
        return func(*_args, **_kwargs)

    cached_function = cache(func_with_serialized_params)

    @wraps(func)
    def decorator(*args, **kwargs):
        _args = tuple(json.dumps(a, sort_keys=True) if type(a) in (list, dict) else a for a in args)
        _kwargs = {k: json.dumps(v, sort_keys=True) if type(v) in (list, dict) else v for k, v in kwargs.items()}
        return cached_function(*_args, **_kwargs)

    decorator.cache_info = cached_function.cache_info
    decorator.cache_clear = cached_function.cache_clear
    return decorator


def np_cache(function):
    """LRU-cache a function whose args/kwargs may contain numpy arrays."""

    @lru_cache
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}
        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear
    return wrapper
