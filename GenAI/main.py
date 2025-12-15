"""
Simple demo that sends a structured-extraction request to the local SGLang
OpenAI-compatible endpoint running on port 30000.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict
from pydantic import BaseModel, Field
from openai import OpenAI

# Ensure local traffic bypasses the corporate proxy.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

BASE_URL = os.environ.get("SGLANG_OPENAI_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SGLANG_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3-32B")

class MedicalData(BaseModel):
    diagnosis: str = Field(description="The diagnosis of the patient")
    medications: list[str] = Field(description="The medications the patient is taking")
    allergies: list[str] = Field(description="The allergies the patient has")
    follow_up: str = Field(description="The follow-up instructions for the patient")


def run(prompt: str) -> Dict[str, Any]:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system_prompt = (
        "You are an assistant that extracts structured medical data. "
        "Always respond with strict JSON using the provided schema."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "MedicalData",
            # convert the pydantic model to json schema
            "schema": MedicalData.model_json_schema(),
        },
    },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    med_data = MedicalData.model_validate_json(content)
    return med_data.model_dump_json()


def main() -> None:
    sample_note = (
        "Ms. Li, 54-year-old female with type 2 diabetes, reports improved glucose "
        "control on metformin 1g BID. Still has mild neuropathy in feet. "
        "Allergic to penicillin. Recommend continuing metformin, add gabapentin "
        "100mg nightly, schedule follow-up in 4 weeks."
    )
    result = run(sample_note)
    print(result)


if __name__ == "__main__":
    main()
