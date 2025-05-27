
import logging
import os
import concurrent.futures
import httpx
from openai import OpenAI, RateLimitError
import tenacity, logging, time, os, re
from typing import List, Tuple

import time
import re
from pathlib import Path
import threading

file_dir = os.path.dirname(os.path.abspath(__file__))  # Location of this file
with open(Path(f"{file_dir}/../gpt/system_prompt.txt")) as f:
    system_prompt = f.read()
with open(Path(f"{file_dir}/../gpt/evolution_prompt.txt")) as f:
    evolution_prompt = f.read()
with open(Path(f"{file_dir}/../gpt/initial_example_prompt.txt")) as f:
    initial_example_prompt = f.read()
with open(Path(f"{file_dir}/../gpt/evolution_example_prompt.txt")) as f:
    evolution_example_prompt = f.read()
with open(Path(f"{file_dir}/../gpt/terrain_example_initial.py")) as f:
    initial_terrain_example = f.read()
with open(Path(f"{file_dir}/../gpt/terrain_example_evolution.py")) as f:
    evolution_terrain_example = f.read()

# single global client with a sane read/connect timeout
client = OpenAI(
    http_client=httpx.Client(timeout=httpx.Timeout(1200, connect=10))
)
replay_run = ""  # Set to a log directory (e.g., "outputs/.../gpt_queries") to replay a specific run's LLM responses
replay_idx = 0   # Used to keep track of which response to load from a run (if replay_run is set)
replay_idx_lock = threading.Lock()
replay_initial_only = False  # Set to True to only replay initial queries and generate evolution queries from scratch

gpt_pricing = {
     "gpt-4o-2024-11-20": (2.5e-6, 10e-6),
     "gpt-4.1-2025-04-14": (2.0e-6, 8.0e-6),
     'gpt-4o-mini-2024-07-18': (0.15e-6, 0.6e-6)
}

def prepare_prompts(cfg):
    global system_prompt, initial_example_message, evolution_example_message
    
    initial_example_message = initial_example_prompt.replace("<INSERT EXAMPLE HERE>", initial_terrain_example)
    evolution_example_message = evolution_example_prompt.replace("<INSERT INITIAL EXAMPLE HERE>", initial_terrain_example)
    evolution_example_message = evolution_example_message.replace("<INSERT EVOLUTION EXAMPLE HERE>", evolution_terrain_example)

def query_gpt_initial(cfg, num_samples=1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_example_message}
    ]
    return query_gpt(cfg, messages, num_samples)

def query_gpt_evolution(cfg, prev_terrain_code, eval_statistics, terrain_stats, all_best_terrain_descriptions, num_samples=1):
    """
    cfg, 
    prev_terrain_code # terrain to be evolved
    eval_statistics # policy statistics before and after training for terrain to be evolved
    terrain_stats # height max and differences for each terrain
    all_best_terrain_descriptions # all terrain descriptions in this parallel run's lineage
    """
    global replay_run
    if replay_initial_only:
        replay_run = ""

    all_best_terrain_descriptions = "\n".join(["- " + desc for desc in all_best_terrain_descriptions])

    evolution_message = evolution_prompt
    evolution_message = evolution_message.replace("<INSERT POLICY STATISTICS HERE>", eval_statistics)
    evolution_message = evolution_message.replace("<INSERT TERRAIN STATISTICS HERE>", terrain_stats)
    evolution_message = evolution_message.replace("<INSERT TERRAIN DESCRIPTIONS HERE>", all_best_terrain_descriptions)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "```python\n" + prev_terrain_code + "\n```"},
        {"role": "user", "content": evolution_message},
        {"role": "user", "content": evolution_example_message}
    ]
    return query_gpt(cfg, messages, num_samples)

# ────────────────────────────────────────────────────────────────────────────────
@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=20),
                reraise=True,
                retry=tenacity.retry_if_exception_type(RateLimitError))
def _one_completion(msgs: List[dict], model: str) -> str:
    """
    Do a single streaming completion so the connection never goes idle.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True
    )
    pieces = []
    for chunk in stream:
        pieces.append(chunk.choices[0].delta.content or "")
    return "".join(pieces)


def _get_completions_parallel(msgs: List[dict], model: str, k: int) -> List[str]:
    """
    Launch k completions in parallel using a small ThreadPool.
    """
    max_workers = min(16, k)     # cap threads so we don't oversubscribe
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one_completion, msgs, model) for _ in range(k)]
        return [f.result() for f in futures]


# ────────────────────────────────────────────────────────────────────────────────
def query_gpt(cfg, messages: List[dict], num_samples: int = 1
              ) -> Tuple[List[dict], List[str], List[str], float, float]:
    """
    Query the OpenAI chat model with *num_samples* independent streaming calls.
    Falls back to replay directory if `replay_run` is set.
    Returns:
        messages, raw_responses, parsed_code_responses, prompt_cost, response_cost
    """
    logging.info(f"Querying OpenAI API for {num_samples} samples using {cfg.gpt_model}...")

    # ── 1. Handle replay mode ───────────────────────────────────────────────────
    if replay_run:
        global replay_idx
        log_dir_list = sorted(
            os.path.join(root, d)
            for root, dirs, _ in os.walk(replay_run)
            for d in dirs if "query" in d
        )

        with replay_idx_lock:
            log_dir = log_dir_list[replay_idx]
            files = sorted(
                [f for f in os.listdir(log_dir) if "response" in f],
                key=lambda x: int(x.rstrip(".txt").split("-")[-1])
            )
            raw_responses = [
                open(os.path.join(log_dir, f), "r").read()
                for f in files[:num_samples]
            ]
            prompt_cost = response_cost = 0.0
            logging.info(f"Loaded past response from {log_dir}")
            replay_idx = (replay_idx + 1) % len(log_dir_list)

    # ── 2. Live call to OpenAI ──────────────────────────────────────────────────
    else:
        t0 = time.time()
        raw_responses = _get_completions_parallel(messages, cfg.gpt_model, num_samples)
        elapsed = time.time() - t0
        logging.info(f"Received {num_samples} completions in {elapsed:0.1f}s")

        # usage metrics are not returned in streaming mode → estimate token counts
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        response_tokens = sum(len(r.split()) for r in raw_responses)
        prompt_price, resp_price = gpt_pricing[cfg.gpt_model]
        prompt_cost = prompt_price * prompt_tokens
        response_cost = resp_price * response_tokens
        logging.info(f"Approx. cost: ${prompt_cost + response_cost:0.4f}")

    # ── 3. Extract code from each response ──────────────────────────────────────
    parsed_responses = []
    code_patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'^(.*?)$',
    ]
    for resp in raw_responses:
        snippet = ""
        for pat in code_patterns:
            m = re.search(pat, resp, re.DOTALL)
            if m:
                snippet = m.group(1).strip()
                break
        # keep only indented code or import/def lines
        lines = [
            ln for ln in snippet.split("\n")
            if ln.strip() == "" or ln.startswith((" ", "def", "import"))
        ]
        parsed_responses.append("\n".join(lines))

    return messages, raw_responses, parsed_responses, prompt_cost, response_cost

def log_gpt_query(messages, responses, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(f"{save_dir}/prompt.txt", "w") as f:
        f.write("\n\n".join([message["content"] for message in messages]))
    
    for i, response in enumerate(responses):
        with open(f"{save_dir}/response-{i}.txt", "w") as f:
            f.write(response)