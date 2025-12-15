"""
Simple demo that sends a structured-extraction request to the local SGLang
OpenAI-compatible endpoint running on port 30000.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from openai import OpenAI

# Ensure local traffic bypasses the corporate proxy.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

BASE_URL = os.environ.get("SGLANG_OPENAI_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SGLANG_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3-32B")


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    match = re.search(r"\{.*\}", text, re.S)
    candidate = match.group(0) if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model did not return valid JSON: {text}") from None


def run_structured_extraction(note: str) -> Dict[str, Any]:
    """Send the medical note to the chat-completions endpoint and parse JSON."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system_prompt = (
        "You are an assistant that extracts structured medical data. "
        "Always respond with strict JSON using the provided schema."
    )
    schema_description = (
        "Return JSON with keys: diagnosis (string), medications (list of strings), "
        "allergies (list of strings), follow_up (string)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{schema_description}\n\nPatient note:\n{note}"},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    return _extract_json_block(content)


def main() -> None:
    sample_note = (
        "Ms. Li, 54-year-old female with type 2 diabetes, reports improved glucose "
        "control on metformin 1g BID. Still has mild neuropathy in feet. "
        "Allergic to penicillin. Recommend continuing metformin, add gabapentin "
        "100mg nightly, schedule follow-up in 4 weeks."
    )
    result = run_structured_extraction(sample_note)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
