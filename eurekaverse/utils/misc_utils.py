
import subprocess
import json
import os
import logging
import time
import threading
import contextlib
import re
from pathlib import Path
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

# ────────────────────────────────────────────────────────────────
# Launch a command and return Popen handle
# ────────────────────────────────────────────────────────────────
def run_subprocess(command: str, log_file: Path | str | None):
    """
    Execute *command* inside Bash so that shell built‑ins (`source`, `&&`, etc.)
    work. Output handling depends on log_file parameter.
    """
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"

    if log_file is None:
        # No logging - capture in PIPE for caller to handle
        proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
    else:
        # Log to file - write header then redirect output
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(log_path, "a") as f:
            f.write("\n" + "=" * 100 + "\n"
                    f"Running command: {command}\n"
                    + "=" * 100 + "\n")
        
        # Open file again for subprocess to write to
        log_file_handle = open(log_path, "a")
        proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            stdout=log_file_handle,
            stderr=log_file_handle,
            text=True,
            env=env,
        )
        # Store file handle so caller can close it after process completes
        proc._log_file_handle = log_file_handle

    return proc

# ────────────────────────────────────────────────────────────────
# Wait for success/failure strings, premature exit, or timeout
# ────────────────────────────────────────────────────────────────
def wait_subprocess(process, log_file, success_log, failure_log, timeout=60):
    """
    Monitor subprocess completion by checking for success/failure patterns.
    
    For PIPE mode: reads from process.stdout
    For file mode: efficiently tails the log file
    
    Returns (success_found, timed_out).
    """
    deadline = time.time() + timeout
    file_position = 0  # Track where we've read up to in the log file
    
    while True:
        if log_file is None:
            # PIPE mode - read from process stdout
            output = process.stdout.readline() if process.stdout else ""
            if output:
                if success_log in output:
                    return True, False
                if failure_log in output:
                    return False, False
        else:
            # File mode - efficiently tail the log file
            try:
                with open(log_file, 'r') as f:
                    f.seek(file_position)  # Start reading from where we left off
                    new_content = f.read()
                    file_position = f.tell()  # Update position for next read
                    
                    if new_content:
                        if success_log in new_content:
                            return True, False
                        if failure_log in new_content:
                            return False, False
            except FileNotFoundError:
                # Log file doesn't exist yet, continue waiting
                pass
            except IOError:
                # File might be locked by subprocess, continue waiting
                pass

        # Check if process has terminated
        retcode = process.poll()
        if retcode is not None:
            if retcode != 0:
                logging.warning(f"Process terminated with code {retcode}")
            return False, False
        
        # Check timeout
        if time.time() > deadline:
            logging.warning("wait_subprocess(): timeout")
            return False, True

        time.sleep(0.2)  # More responsive than 1 second

# ────────────────────────────────────────────────────────────────
# Helper: grab what’s left in the pipe after exit / timeout
# ────────────────────────────────────────────────────────────────
def _drain_and_dump(proc: subprocess.Popen, mirror):
    try:
        remaining, _ = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        remaining = ""
    if remaining:
        if mirror:
            mirror.write("\n── residual output ──\n")
            mirror.write(remaining)
        else:
            print("\n── residual output ──")
            print(remaining, end="")

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