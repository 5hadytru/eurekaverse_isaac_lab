
import subprocess
import json
import os
import logging
import time
import threading
import contextlib
import re

import time
import threading

try:
    import pynvml  # pip install nvidia-ml-py3
except ImportError as exc:           # fail hard and early
    raise RuntimeError(
        "pynvml (package nvidia‑ml‑py3) is required for GPU queries: "
        "pip install nvidia-ml-py3"
    ) from exc

# ---------- one‑time NVML initialisation ----------
pynvml.nvmlInit()
_NVML_INITIALISED = True

# ---------- concurrency + throttling state ----------
_gpustat_lock         = threading.Lock()
_gpustat_next_allowed = 0.0          # epoch time

def _device_count() -> int:
    """Return the number of visible NVIDIA GPUs."""
    return pynvml.nvmlDeviceGetCount()

def _memory_used(index: int) -> int:
    """Bytes of memory currently allocated on GPU `index`."""
    handle   = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used                         # integer bytes


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_freest_gpu(gpustat_delay: float = 10.0) -> str:
    """
    Return a device string ('cuda:<idx>') for the GPU with the
    least memory currently allocated.  Calls are rate‑limited so that
    NVML is hit at most once every `gpustat_delay` seconds across
    all threads, mirroring the original gpustat-based behaviour.
    """
    global _gpustat_next_allowed

    with _gpustat_lock:
        now = time.time()
        if now < _gpustat_next_allowed:          # honour throttle window
            time.sleep(_gpustat_next_allowed - now)

        n = _device_count()
        if n == 0:
            raise RuntimeError("No NVIDIA GPUs detected by NVML.")

        # O(n) scan for least‑used card
        min_used, best_idx = None, 0
        for idx in range(n):
            used = _memory_used(idx)
            if min_used is None or used < min_used:
                min_used, best_idx = used, idx

        _gpustat_next_allowed = time.time() + gpustat_delay

    return f"cuda:{best_idx}"


def get_num_gpus() -> int:
    """Return the total number of NVIDIA GPUs visible to NVML."""
    return _device_count()

def run_subprocess(command, log_file):
    if log_file == None:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ.copy(), "TQDM_DISABLE": "1"})
    else:
        with open(log_file, "a") as f:
            f.write("\n" + "="*100 + "\n" + f"Running command: {command}\n" + "="*100 + "\n")
            process = subprocess.Popen(command.split(), stdout=f, stderr=f, env={**os.environ.copy(), "TQDM_DISABLE": "1"})
    return process

def wait_subprocess(process, log_file, success_log, failure_log, timeout=60):
    timeout = time.time() + timeout
    while True:
        if log_file is None:
            output = process.stdout.readline()
        else: 
            with open(log_file, 'r') as file:
                output = file.read()
        if output:
            if success_log in output:
                return True, False
            if failure_log in output:
                time.sleep(1)  # Wait for the process to finish writing to the log file
                return False, False

        retcode = process.poll()
        if retcode is not None:
            logging.warning(f"Process terminated while waiting with code {retcode}")
            return False, False
        if time.time() > timeout:
            return False, True

        time.sleep(1)

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield

@contextlib.contextmanager
def seeded():
    import random
    import numpy as np
    import torch

    state = {}
    state['random'] = random.getstate()
    state['np_random'] = np.random.get_state()
    state['torch_rng_cpu'] = torch.get_rng_state()
    state['torch_rng_gpu'] = torch.cuda.get_rng_state_all()
    state['torch_rng_deterministic'] = torch.backends.cudnn.deterministic
    state['os_hash_seed'] = os.environ.get('PYTHONHASHSEED', None)
    
    try:
        yield
    finally:
        random.setstate(state['random'])
        np.random.set_state(state['np_random'])
        torch.set_rng_state(state['torch_rng_cpu'])
        for i, state_gpu in enumerate(state['torch_rng_gpu']):
            torch.cuda.set_rng_state(state_gpu, i)
        if state['os_hash_seed'] is None:
            del os.environ['PYTHONHASHSEED']
        else:
            os.environ['PYTHONHASHSEED'] = state['os_hash_seed']
        torch.backends.cudnn.deterministic = state['torch_rng_deterministic']

def alphanum_key(s):
    # Use this with sorted() to sort a list of strings alphanumerically
    return [int(text) if text.isdigit() else text for text in re.split('(\d+)', s)]