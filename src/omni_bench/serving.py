from __future__ import annotations

import os
import subprocess
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import requests

from omni_bench.config import ModelConfig


def vllm_command(model: ModelConfig, *, require_executable: bool = False) -> list[str]:
    executable = shutil.which("vllm")
    if executable is None and require_executable:
        raise RuntimeError(
            "Could not find the 'vllm' executable. Install dependencies with `uv sync`, "
            "or start vLLM manually and run `omni-bench run` without `--serve`."
        )
    cmd = [
        executable or "vllm",
        "serve",
        model.weight_path,
        "--served-model-name",
        model.api_model_name,
        "--host",
        model.vllm.host,
        "--port",
        str(model.vllm.port),
    ]
    if model.vllm.tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(model.vllm.tensor_parallel_size)]
    data_parallel_size = resolve_data_parallel_size(model.vllm.data_parallel_size)
    if data_parallel_size is not None:
        cmd += ["--data-parallel-size", str(data_parallel_size)]
    if model.vllm.max_model_len is not None:
        cmd += ["--max-model-len", str(model.vllm.max_model_len)]
    if model.vllm.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(model.vllm.gpu_memory_utilization)]
    cmd += model.vllm.extra_args
    return cmd


def resolve_data_parallel_size(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if value != "auto":
        raise ValueError("vllm.data_parallel_size must be an integer, null, or 'auto'.")
    return max(1, detect_available_gpus())


def detect_available_gpus() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if devices and devices != ["-1"]:
            return len(devices)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    return 1


def wait_for_server(base_url: str, timeout_s: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{base_url.rstrip('/')}/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(models_url, timeout=5)
            if response.ok:
                print(f"vLLM server is ready: {models_url}", flush=True)
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {models_url}: {last_error}")


def capture_process_output(
    process: subprocess.Popen[str],
    log: TextIO,
    stream_to_terminal: threading.Event,
) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        log.write(line)
        log.flush()
        if stream_to_terminal.is_set():
            sys.stdout.write(line)
            sys.stdout.flush()


@contextmanager
def serve_model(model: ModelConfig, log_dir: Path, timeout_s: float = 900.0) -> Iterator[None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model.name}.vllm.log"
    with log_file.open("w", encoding="utf-8") as log:
        command = vllm_command(model, require_executable=True)
        print(f"Starting vLLM for {model.name}", flush=True)
        print(f"vLLM log: {log_file}", flush=True)
        print(f"Command: {' '.join(command)}", flush=True)
        stream_to_terminal = threading.Event()
        stream_to_terminal.set()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_thread = threading.Thread(
            target=capture_process_output,
            args=(process, log, stream_to_terminal),
            daemon=True,
        )
        output_thread.start()
        try:
            wait_for_server(model.resolved_base_url, timeout_s=timeout_s)
            stream_to_terminal.clear()
            yield
        finally:
            stream_to_terminal.clear()
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            output_thread.join(timeout=5)
