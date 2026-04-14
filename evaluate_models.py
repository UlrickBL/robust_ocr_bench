"""
Evaluate vision-language models on the robust OCR benchmark dataset.

Supports:
  - Anthropic Claude (claude-opus-4-6, claude-sonnet-4-6, …)
  - OpenAI GPT-4o / GPT-4-vision
  - Local transformers models (LLaVA, Qwen2-VL, InternVL, …)

Usage examples:
    # Anthropic
    python evaluate_models.py \
        --dataset YOUR_HF_USERNAME/robust-ocr-bench \
        --provider anthropic \
        --model claude-sonnet-4-6

    # OpenAI
    python evaluate_models.py \
        --dataset YOUR_HF_USERNAME/robust-ocr-bench \
        --provider openai \
        --model gpt-4o

    # Local (HF transformers)
    python evaluate_models.py \
        --dataset YOUR_HF_USERNAME/robust-ocr-bench \
        --provider local \
        --model llava-hf/llava-1.5-7b-hf

    # Limit to specific variations
    python evaluate_models.py ... --variations original rotate_90 invert

    # Save results to JSON
    python evaluate_models.py ... --output results.json
"""

from __future__ import annotations  # enables list[X] | None syntax on Python 3.9

import argparse
import base64
import io
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNOTATION_KEYS = [
    "date",
    "doc_no_receipt_no",
    "seller_address",
    "seller_gst_id",
    "seller_name",
    "seller_phone",
    "total_amount",
    "total_tax",
]

EXTRACTION_PROMPT = """\
You are a receipt OCR specialist. Extract the following key information from this receipt image and return ONLY a valid JSON object — no markdown, no explanation.

Required fields (use empty string "" if not found):
- date
- doc_no_receipt_no
- seller_address
- seller_gst_id
- seller_name
- seller_phone
- total_amount
- total_tax

Output format (JSON only):
{
  "date": "...",
  "doc_no_receipt_no": "...",
  "seller_address": "...",
  "seller_gst_id": "...",
  "seller_name": "...",
  "seller_phone": "...",
  "total_amount": "...",
  "total_tax": "..."
}"""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def image_to_base64(image_data: Any, fmt: str = "PNG") -> str:
    """Convert various image representations to a base64-encoded PNG string."""
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        img = Image.fromarray(np.array(image_data))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()


def image_to_pil(image_data: Any) -> Image.Image:
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        return image_data
    return Image.fromarray(np.array(image_data))


# ---------------------------------------------------------------------------
# Abstract model interface
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Minimal interface every model provider must implement."""

    @abstractmethod
    def extract(self, image_data: Any) -> dict[str, str]:
        """Run extraction on a single image. Returns a dict of field→value."""
        ...

    def _parse_json_response(self, text: str) -> dict[str, str]:
        """Best-effort JSON extraction from a model response."""
        # Try to extract a JSON block if wrapped in markdown
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        # Fallback: find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            result = json.loads(text)
            return {k: str(result.get(k, "") or "") for k in ANNOTATION_KEYS}
        except json.JSONDecodeError:
            return {k: "" for k in ANNOTATION_KEYS}


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicModel(BaseModel):
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 512):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, image_data: Any) -> dict[str, str]:
        import anthropic
        b64 = image_to_base64(image_data)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        raw = response.content[0].text
        return self._parse_json_response(raw)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIModel(BaseModel):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512):
        from openai import OpenAI  # lazy import
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, image_data: Any) -> dict[str, str]:
        b64 = image_to_base64(image_data)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content or ""
        return self._parse_json_response(raw)


# ---------------------------------------------------------------------------
# Local HuggingFace transformers provider
# ---------------------------------------------------------------------------

class LocalTransformersModel(BaseModel):
    """
    Generic wrapper for HuggingFace vision-language models.
    Tested with: llava-hf/llava-1.5-7b-hf, Qwen/Qwen2-VL-7B-Instruct, etc.
    The model must support the transformers pipeline("image-to-text") interface
    or have a processor + model.generate() interface.
    """

    def __init__(self, model_id: str, device: str = "auto", max_new_tokens: int = 512):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        print(f"Loading local model {model_id} …")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model_hf = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def extract(self, image_data: Any) -> dict[str, str]:
        import torch

        pil_img = image_to_pil(image_data).convert("RGB")

        # Build a chat-style prompt (works for most VLMs)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        # Use apply_chat_template if available
        if hasattr(self.processor, "apply_chat_template"):
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text_prompt = EXTRACTION_PROMPT

        inputs = self.processor(
            text=text_prompt,
            images=[pil_img],
            return_tensors="pt",
        ).to(self.model_hf.device)

        with torch.no_grad():
            output_ids = self.model_hf.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        # Decode only the new tokens
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return self._parse_json_response(raw)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

PROVIDERS = {
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
    "local": LocalTransformersModel,
}


def build_model(provider: str, model: str, **kwargs) -> BaseModel:
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    return cls(model, **kwargs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_value(v: str) -> str:
    """Strip whitespace/punctuation differences for a lenient comparison."""
    return re.sub(r"\s+", " ", v.strip().lower())


def field_exact_match(pred: str, gold: str) -> bool:
    return normalize_value(pred) == normalize_value(gold)


def compute_metrics(
    predictions: list[dict[str, str]],
    ground_truths: list[dict[str, str]],
) -> dict:
    """
    Returns per-field accuracy and macro-averaged accuracy.
    For text fields we use exact match after normalisation.
    """
    per_field: dict[str, list[bool]] = {k: [] for k in ANNOTATION_KEYS}

    for pred, gold in zip(predictions, ground_truths):
        for key in ANNOTATION_KEYS:
            match = field_exact_match(pred.get(key, ""), gold.get(key, ""))
            per_field[key].append(match)

    field_accuracy = {
        k: round(np.mean(v) * 100, 2) for k, v in per_field.items()
    }
    macro_accuracy = round(np.mean(list(field_accuracy.values())), 2)

    return {
        "macro_accuracy": macro_accuracy,
        "field_accuracy": field_accuracy,
        "n_samples": len(predictions),
    }


def compute_metrics_by_variation(
    records: list[dict],
) -> dict[str, dict]:
    """Group records by variation_id and compute metrics per group."""
    groups: dict[str, tuple[list, list]] = {}
    for rec in records:
        vid = rec["variation_id"]
        if vid not in groups:
            groups[vid] = ([], [])
        groups[vid][0].append(rec["prediction"])
        groups[vid][1].append(rec["ground_truth"])

    return {
        vid: compute_metrics(preds, golds)
        for vid, (preds, golds) in groups.items()
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: BaseModel,
    dataset,
    variations: list[str] | None = None,
    delay_between_calls: float = 0.5,
) -> list[dict]:
    """
    Run model.extract() on every sample and collect results.

    Returns a list of result records, one per sample.
    """
    results = []

    for row in tqdm(dataset, desc="Evaluating"):
        vid = row["variation_id"]
        if variations and vid not in variations:
            continue

        ground_truth = {k: str(row.get(k, "") or "") for k in ANNOTATION_KEYS}

        try:
            prediction = model.extract(row["image"])
            error = None
        except Exception as exc:
            prediction = {k: "" for k in ANNOTATION_KEYS}
            error = str(exc)

        results.append({
            "sample_idx": row["sample_idx"],
            "source_idx": row["source_idx"],
            "variation_id": vid,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "error": error,
        })

        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(overall: dict, by_variation: dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    print(f"  OVERALL  — macro accuracy: {overall['macro_accuracy']}%  "
          f"(n={overall['n_samples']})")
    print("=" * 60)
    for field, acc in overall["field_accuracy"].items():
        print(f"  {field:<25} {acc:>6.2f}%")

    print("\n" + "=" * 60)
    print("  BY VARIATION")
    print("=" * 60)
    for vid, metrics in by_variation.items():
        print(f"\n  [{vid}]  macro: {metrics['macro_accuracy']}%  "
              f"(n={metrics['n_samples']})")
        for field, acc in metrics["field_accuracy"].items():
            print(f"    {field:<25} {acc:>6.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR models on robust-ocr-bench")
    parser.add_argument("--dataset", required=True,
                        help="HF dataset id, e.g. myuser/robust-ocr-bench")
    parser.add_argument("--provider", required=True,
                        choices=list(PROVIDERS.keys()),
                        help="Model provider")
    parser.add_argument("--model", required=True,
                        help="Model name/id (e.g. claude-sonnet-4-6, gpt-4o, or a HF model id)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to evaluate on")
    parser.add_argument("--variations", nargs="*",
                        help="Subset of variation_ids to evaluate (default: all)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between API calls (default 0.5)")
    parser.add_argument("--output", default=None,
                        help="Path to save full results as JSON")
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset} …")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} rows.")

    # Build model
    model = build_model(args.provider, args.model, max_tokens=args.max_tokens)

    # Run evaluation
    results = evaluate(
        model,
        ds,
        variations=args.variations,
        delay_between_calls=args.delay,
    )

    if not results:
        print("No results — check --variations filter.")
        return

    # Compute metrics
    overall = compute_metrics(
        [r["prediction"] for r in results],
        [r["ground_truth"] for r in results],
    )
    by_variation = compute_metrics_by_variation(results)

    # Print report
    print_report(overall, by_variation)

    # Save
    if args.output:
        output = {
            "model": args.model,
            "provider": args.provider,
            "dataset": args.dataset,
            "overall": overall,
            "by_variation": by_variation,
            "records": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
