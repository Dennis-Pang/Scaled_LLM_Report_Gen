#!/usr/bin/env python3
"""
Generate a summary using VLM from sample data.

Usage:
    python generate_sum.py --sample-id sample1 --output summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from LLM.client import SGLangLLM
from scripts.vlm_data_loader import PatientDataLoader

DEFAULT_SYSTEM_PROMPT = (
    "You are a healthcare clinician. Provide a clear, concise summary of the patient data. "
    "The case includes both text and images. Do not describe the images in detail; instead, "
    "briefly state your key observations from the images and integrate them with the textual information."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLM summary from sample data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_sum.py --sample-id sample1\n"
            "  python generate_sum.py --sample-id sample1 --output summary.txt\n"
        ),
    )
    parser.add_argument(
        "--sample-id",
        default="sample1",
        help="Sample ID (default: sample1)",
    )
    parser.add_argument(
        "--patient-id",
        dest="sample_id",
        help="Deprecated: use --sample-id",
    )
    parser.add_argument(
        "--output",
        help="Optional output file path",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Exclude images from VLM input",
    )
    parser.add_argument(
        "--base-url",
        help="Override SGLang base URL",
    )
    parser.add_argument(
        "--model",
        help="Override model ID",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = PatientDataLoader(args.sample_id)
    messages = loader.build_vlm_messages(
        include_images=not args.no_images,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]

    llm = SGLangLLM(
        base_url=args.base_url,
        model=args.model,
        auto_start=args.auto_start,
        gpu_ids=gpu_ids,
        num_gpus_to_select=args.num_gpus,
    )

    response = llm.chat_messages(messages)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(response, encoding="utf-8")
        print(f"Summary saved to {output_path}")
    else:
        print(response)


if __name__ == "__main__":
    main()
