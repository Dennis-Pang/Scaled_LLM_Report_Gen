#!/usr/bin/env python3
"""
CLI chat utility for the local SGLang server.

Examples:
    python -m LLM.chat --text "Hello"
    python -m LLM.chat --text "Describe this image" --image /path/to/img.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .client import SGLangLLM, parse_gpu_ids


def _ensure_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def _read_prompt(value: str | None) -> str:
    if value:
        return value
    if sys.stdin.isatty():
        raise SystemExit("Error: --text is required when not piping input.")
    return sys.stdin.read().strip()


def _resolve_images(images: Sequence[str]) -> list[Path]:
    resolved = []
    for image in images:
        path = Path(image).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        resolved.append(path)
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with local SGLang server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m LLM.chat --text \"Hello\"\n"
            "  python -m LLM.chat --text \"Describe\" --image /path/to/img.png\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "text", "multimodal"),
        default="auto",
        help="Chat mode (default: auto based on images)",
    )
    parser.add_argument("--text", help="User prompt text (or pipe from stdin)")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path for multimodal input (repeatable)",
    )
    parser.add_argument("--system-prompt", help="Optional system prompt")
    parser.add_argument("--base-url", help="Override base URL")
    parser.add_argument("--model", help="Override model ID")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Auto-start server if not running",
    )
    parser.add_argument(
        "--gpu-ids",
        help="Comma-separated GPU IDs (used when auto-starting)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Prompt for this many GPUs when auto-starting",
    )
    return parser


def main() -> None:
    _ensure_no_proxy()
    parser = build_arg_parser()
    args = parser.parse_args()

    interactive = args.text is None and sys.stdin.isatty()
    images = _resolve_images(args.image)

    mode = args.mode
    if mode == "auto":
        mode = "multimodal" if images else "text"
    if mode == "multimodal" and not images:
        parser.error("--mode multimodal requires at least one --image")

    gpu_ids = parse_gpu_ids(args.gpu_ids) if args.gpu_ids else None
    if not args.auto_start and gpu_ids is None:
        gpu_ids = [0]

    llm = SGLangLLM(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        auto_start=args.auto_start,
        gpu_ids=gpu_ids,
        num_gpus_to_select=args.num_gpus,
    )

    if interactive:
        print("Enter prompt (empty line to exit).")
        while True:
            try:
                prompt = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                break
            if mode == "multimodal":
                response = llm.chat_with_images(
                    prompt,
                    images=images,
                    system_prompt=args.system_prompt,
                )
            else:
                response = llm.chat(
                    prompt,
                    system_prompt=args.system_prompt,
                )
            print(response)
    else:
        prompt = _read_prompt(args.text)
        if mode == "multimodal":
            response = llm.chat_with_images(
                prompt,
                images=images,
                system_prompt=args.system_prompt,
            )
        else:
            response = llm.chat(
                prompt,
                system_prompt=args.system_prompt,
            )
        print(response)


if __name__ == "__main__":
    main()
