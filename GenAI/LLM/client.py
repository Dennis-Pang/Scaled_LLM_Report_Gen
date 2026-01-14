#!/usr/bin/env python3
"""
SGLang OpenAI-compatible client for Qwen3-VL.

Examples:
    from LLM.client import SGLangLLM

    llm = SGLangLLM()
    print(llm.chat("Hello"))
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

DEFAULT_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_MODEL = os.environ.get("SGLANG_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
DEFAULT_API_KEY = os.environ.get("SGLANG_API_KEY", "EMPTY")
DEFAULT_HOST = os.environ.get("SGLANG_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
DEFAULT_START_TIMEOUT = int(os.environ.get("SGLANG_START_TIMEOUT", "300"))

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEFAULT_DISABLE_CUDNN_CHECK = _env_flag("SGLANG_DISABLE_CUDNN_CHECK", True)


def _ensure_no_proxy(env: dict[str, str]) -> None:
    """Ensure localhost is excluded from proxying for local server calls."""
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in parts:
                parts.append(host)
        env[key] = ",".join(parts)


def normalize_base_url(base_url: str) -> str:
    """Ensure base_url ends with /v1 and has no trailing slash."""
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    """Parse comma-separated GPU IDs into a list of ints."""
    parsed = [int(part.strip()) for part in gpu_ids.split(",") if part.strip()]
    return parsed


def get_available_gpus() -> list[dict[str, Any]]:
    """
    Get available GPU information using nvidia-smi.

    Returns:
        List of GPU info dictionaries.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "id": int(parts[0]),
                "name": parts[1],
                "memory_total": int(parts[2]),
                "memory_used": int(parts[3]),
                "utilization": int(parts[4]),
            }
        )
    return gpus


def select_gpus_for_model(
    num_gpus: int | None = None,
    available_gpus: list[dict[str, Any]] | None = None,
) -> list[int]:
    """
    Prompt user to select GPU IDs.

    Args:
        num_gpus: Optional number of GPUs to select.
        available_gpus: Optional list of GPU info dicts.

    Returns:
        List of selected GPU IDs.
    """
    gpus = available_gpus or get_available_gpus()
    if not gpus:
        raise RuntimeError("No GPUs detected. Ensure nvidia-smi is available.")

    print("Available GPUs:")
    for gpu in gpus:
        print(
            f"  GPU {gpu['id']}: {gpu['name']} | "
            f"Mem {gpu['memory_used']}/{gpu['memory_total']} MB | "
            f"Util {gpu['utilization']}%"
        )

    available_ids = [gpu["id"] for gpu in gpus]
    while True:
        if num_gpus:
            prompt = f"Select {num_gpus} GPU IDs (comma-separated, default first {num_gpus}): "
        else:
            prompt = "Select GPU IDs (comma-separated, default first GPU): "

        raw = input(prompt).strip()
        if not raw:
            selected = available_ids[: num_gpus or 1]
            return selected

        try:
            selected = parse_gpu_ids(raw)
        except ValueError:
            print("Invalid input. Please enter comma-separated integers.")
            continue

        if num_gpus and len(selected) != num_gpus:
            print(f"Please select exactly {num_gpus} GPU(s).")
            continue

        if not selected:
            print("No GPU IDs provided.")
            continue

        if not all(gpu_id in available_ids for gpu_id in selected):
            print(f"Invalid GPU IDs. Available: {available_ids}")
            continue

        return selected


def _models_endpoint(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/models"


def list_models(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> list[str]:
    """
    List available model IDs from the SGLang server.

    Args:
        base_url: SGLang base URL.
        api_key: API key (OpenAI-compatible).

    Returns:
        List of model IDs.
    """
    url = _models_endpoint(base_url)
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    data = payload.get("data", [])
    return [item.get("id", "") for item in data if item.get("id")]


def check_model_loaded(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
) -> bool:
    """
    Check if a model is loaded on the server.

    Args:
        model_name: Model ID to check.
        base_url: SGLang base URL.
        api_key: API key (OpenAI-compatible).

    Returns:
        True if model is loaded, otherwise False.
    """
    models = list_models(base_url=base_url, api_key=api_key)
    return model_name in models


def test_connection(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> bool:
    """
    Test if the SGLang server is reachable.

    Args:
        base_url: SGLang base URL.
        api_key: API key (OpenAI-compatible).

    Returns:
        True if reachable, otherwise False.
    """
    models = list_models(base_url=base_url, api_key=api_key)
    return bool(models)


def _parse_host_port(
    base_url: str,
    host: str | None = None,
    port: int | None = None,
) -> tuple[str, int]:
    parsed = urlparse(base_url)
    resolved_host = host or parsed.hostname or DEFAULT_HOST
    resolved_port = port or parsed.port or DEFAULT_PORT
    return resolved_host, resolved_port


def start_server(
    model_name: str,
    gpu_ids: Sequence[int],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra_args: Sequence[str] | None = None,
    disable_cudnn_check: bool = DEFAULT_DISABLE_CUDNN_CHECK,
) -> subprocess.Popen:
    """
    Start SGLang server with selected GPUs.

    Args:
        model_name: Model ID or path.
        gpu_ids: GPU IDs to use.
        host: Host to bind.
        port: Port to bind.
        extra_args: Extra arguments for server command.

    Returns:
        subprocess.Popen handle for the server.
    """
    env = os.environ.copy()
    if disable_cudnn_check:
        env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    _ensure_no_proxy(env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    server_cmd = os.environ.get("SGLANG_SERVER_CMD")
    server_args = os.environ.get("SGLANG_SERVER_ARGS", "")
    extra = list(extra_args or [])
    if server_args:
        extra.extend(shlex.split(server_args))

    if server_cmd:
        cmd = shlex.split(server_cmd) + extra
    else:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_name,
            "--host",
            host,
            "--port",
            str(port),
            "--tp",
            str(max(len(gpu_ids), 1)),
        ] + extra

    return subprocess.Popen(cmd, env=env, start_new_session=True)


def ensure_server(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    auto_start: bool = True,
    gpu_ids: Sequence[int] | None = None,
    num_gpus_to_select: int | None = None,
    host: str | None = None,
    port: int | None = None,
    timeout: int = DEFAULT_START_TIMEOUT,
    disable_cudnn_check: bool = DEFAULT_DISABLE_CUDNN_CHECK,
) -> bool:
    """
    Ensure SGLang server is running and model is loaded.

    Args:
        model_name: Model ID to check/load.
        base_url: SGLang base URL.
        api_key: API key (OpenAI-compatible).
        auto_start: Whether to start server automatically if not running.
        gpu_ids: GPU IDs to use for server startup.
        num_gpus_to_select: Prompt to select this many GPUs if gpu_ids is None.
        host: Host to bind if starting a server.
        port: Port to bind if starting a server.
        timeout: Seconds to wait for model to load.

    Returns:
        True if server is running and model is loaded, otherwise False.
    """
    if check_model_loaded(model_name=model_name, base_url=base_url, api_key=api_key):
        return True

    if not auto_start:
        return False

    if gpu_ids is None:
        gpu_ids = select_gpus_for_model(num_gpus=num_gpus_to_select)

    resolved_host, resolved_port = _parse_host_port(base_url, host, port)
    start_server(
        model_name=model_name,
        gpu_ids=gpu_ids,
        host=resolved_host,
        port=resolved_port,
        disable_cudnn_check=disable_cudnn_check,
    )

    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_model_loaded(model_name=model_name, base_url=base_url, api_key=api_key):
            return True
        time.sleep(2)

    return False


class SGLangLLM:
    """
    OpenAI-compatible client for SGLang VLM.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        auto_start: bool = True,
        gpu_ids: Sequence[int] | None = None,
        num_gpus_to_select: int | None = None,
        host: str | None = None,
        port: int | None = None,
        start_timeout: int = DEFAULT_START_TIMEOUT,
        disable_cudnn_check: bool = DEFAULT_DISABLE_CUDNN_CHECK,
    ) -> None:
        """
        Initialize SGLangLLM client.

        Args:
            base_url: SGLang base URL.
            model: Model ID.
            api_key: API key (OpenAI-compatible).
            temperature: Default temperature.
            max_tokens: Default max tokens.
            auto_start: Auto-start server if not running.
            gpu_ids: GPU IDs to use (if None, prompt user).
            num_gpus_to_select: If provided, prompt for this many GPUs.
            host: Host to bind if starting a server.
            port: Port to bind if starting a server.
            start_timeout: Time to wait for model load.
        """
        _ensure_no_proxy(os.environ)

        self.base_url = normalize_base_url(base_url or DEFAULT_BASE_URL)
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or DEFAULT_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host, self.port = _parse_host_port(self.base_url, host, port)

        if gpu_ids is None:
            self.gpu_ids = select_gpus_for_model(num_gpus=num_gpus_to_select)
        else:
            self.gpu_ids = list(gpu_ids)

        if auto_start:
            ensure_server(
                model_name=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                auto_start=True,
                gpu_ids=self.gpu_ids,
                host=self.host,
                port=self.port,
                timeout=start_timeout,
                disable_cudnn_check=disable_cudnn_check,
            )

        if OpenAI is None:
            raise ImportError(
                "OpenAI client library is not installed. "
                "Install with: pip install openai"
            ) from _OPENAI_IMPORT_ERROR

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a text-only chat request.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Optional override for temperature.
            max_tokens: Optional override for max tokens.

        Returns:
            Assistant response text.
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_with_images(
        self,
        prompt_or_content: str | list[dict[str, Any]],
        images: Sequence[str | Path] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a multimodal chat request.

        Args:
            prompt_or_content: Prompt text or OpenAI content list.
            images: Optional list of image paths.
            system_prompt: Optional system prompt.
            temperature: Optional override for temperature.
            max_tokens: Optional override for max tokens.

        Returns:
            Assistant response text.
        """
        if isinstance(prompt_or_content, list):
            content = prompt_or_content
        else:
            content = [{"type": "text", "text": prompt_or_content}]
            for image in images or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_data_url(image),
                            "detail": "high",
                        },
                    }
                )

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return self.chat_messages(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send chat request with pre-built messages.

        Args:
            messages: OpenAI-compatible messages list.
            temperature: Optional override for temperature.
            max_tokens: Optional override for max tokens.

        Returns:
            Assistant response text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _image_to_data_url(self, image: str | Path) -> str:
        """
        Convert an image file to a data URL for OpenAI image_url content.

        Args:
            image: Image path or data URL.

        Returns:
            data: URL with base64 payload.
        """
        if isinstance(image, Path):
            path = image
        else:
            if image.startswith("data:") or image.startswith("http"):
                return image
            path = Path(image)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        mime_type = MIME_MAP.get(path.suffix.lower(), "image/png")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SGLang client utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gpus", help="List available GPUs")

    check_parser = subparsers.add_parser("check", help="Check if model is loaded")
    check_parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    check_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL")

    test_parser = subparsers.add_parser("test", help="Test server connection")
    test_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL")

    start_parser = subparsers.add_parser("start", help="Start SGLang server")
    start_parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    start_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL")
    start_parser.add_argument("--gpus", help="Comma-separated GPU IDs")
    start_parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    start_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    start_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_START_TIMEOUT,
        help="Seconds to wait for model load",
    )
    start_parser.add_argument(
        "--disable-cudnn-check",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DISABLE_CUDNN_CHECK,
        help="Disable SGLang CuDNN compatibility check",
    )

    args = parser.parse_args()

    if args.command == "gpus":
        gpus = get_available_gpus()
        if not gpus:
            print("No GPUs detected.")
            return
        for gpu in gpus:
            print(
                f"GPU {gpu['id']}: {gpu['name']} | "
                f"Mem {gpu['memory_used']}/{gpu['memory_total']} MB | "
                f"Util {gpu['utilization']}%"
            )
        return

    if args.command == "check":
        loaded = check_model_loaded(
            model_name=args.model,
            base_url=args.base_url,
            api_key=DEFAULT_API_KEY,
        )
        print(f"Model loaded: {loaded}")
        return

    if args.command == "test":
        ok = test_connection(base_url=args.base_url, api_key=DEFAULT_API_KEY)
        print(f"Server reachable: {ok}")
        return

    if args.command == "start":
        if args.gpus:
            gpu_ids = parse_gpu_ids(args.gpus)
        else:
            gpu_ids = select_gpus_for_model()
        ok = ensure_server(
            model_name=args.model,
            base_url=args.base_url,
            api_key=DEFAULT_API_KEY,
            auto_start=True,
            gpu_ids=gpu_ids,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            disable_cudnn_check=args.disable_cudnn_check,
        )
        print(f"Server ready: {ok}")


if __name__ == "__main__":
    main()
