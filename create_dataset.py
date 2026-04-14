from __future__ import annotations

import argparse
import io
import random

import numpy as np
from PIL import Image, ImageOps
from datasets import Dataset, load_dataset
from tqdm import tqdm


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

VARIATION_CONFIGS = [
    ("original",  "Original image, no changes",                 None),
    ("rotate_90", "Rotated 90° counter-clockwise",              lambda img: rotate_image(img, 90)),
    ("rotate_180","Rotated 180°",                               lambda img: rotate_image(img, 180)),
    ("rotate_20", "Tilted ~20° (slight clockwise tilt)",        lambda img: rotate_image(img, -20)),
    ("invert",    "Colour-inverted",                            invert_image),
    ("noisy_bg",  "Thumbnail on white canvas with pixel noise", noisy_background),
]


def rotate_image(img: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    fill_color = (255, 255, 255) if img.mode == "RGB" else 255
    return img.rotate(angle, expand=expand, fillcolor=fill_color)


def invert_image(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img.convert("RGB"))


def noisy_background(
    img: Image.Image,
    scale: float = 0.4,
    canvas_factor: float = 1.6,
    noise_density: float = 0.02,
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)

    small_w = int(img.width * scale)
    small_h = int(img.height * scale)
    small = img.convert("RGB").resize((small_w, small_h), Image.LANCZOS)

    canvas_w = int(small_w * canvas_factor)
    canvas_h = int(small_h * canvas_factor)

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    n_noise = int(canvas_w * canvas_h * noise_density)
    xs = rng.integers(0, canvas_w, size=n_noise)
    ys = rng.integers(0, canvas_h, size=n_noise)
    colors = rng.choice([0, 255], size=n_noise)
    for x, y, c in zip(xs, ys, colors):
        canvas[y, x] = [c, c, c]

    canvas_img = Image.fromarray(canvas, "RGB")

    paste_x = (canvas_w - small_w) // 2
    paste_y = (canvas_h - small_h) // 2
    canvas_img.paste(small, (paste_x, paste_y))

    return canvas_img


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def build_variations(pil_img: Image.Image, base_seed: int = 0) -> list[dict]:
    results = []
    for var_id, desc, fn in VARIATION_CONFIGS:
        if fn is None:
            out = pil_img.convert("RGB")
        elif var_id == "noisy_bg":
            out = fn(pil_img, seed=base_seed)
        else:
            out = fn(pil_img)
        results.append({
            "variation_id": var_id,
            "variation_description": desc,
            "image_bytes": pil_to_bytes(out),
        })
    return results


def extract_annotation(row: dict) -> dict:
    ann = row.get("annotations", {}) or {}
    if isinstance(ann, dict):
        return {k: str(ann.get(k, "") or "") for k in ANNOTATION_KEYS}
    return {k: "" for k in ANNOTATION_KEYS}


def build_dataset(n_samples: int = 50, seed: int = 42) -> Dataset:
    print("Loading source dataset …")
    src = load_dataset("nanonets/key_information_extraction", split="train")

    rng = random.Random(seed)
    indices = rng.sample(range(len(src)), min(n_samples, len(src)))
    print(f"Sampled {len(indices)} receipts.")

    rows = []
    for sample_idx, src_idx in enumerate(tqdm(indices, desc="Building variations")):
        row = src[src_idx]

        raw_image = row.get("image") or row.get("images")
        if isinstance(raw_image, bytes):
            pil_img = Image.open(io.BytesIO(raw_image))
        elif isinstance(raw_image, Image.Image):
            pil_img = raw_image
        else:
            pil_img = Image.fromarray(np.array(raw_image))

        annotation = extract_annotation(row)
        variations = build_variations(pil_img, base_seed=sample_idx)

        for var in variations:
            record = {
                "sample_idx": sample_idx,
                "source_idx": src_idx,
                "variation_id": var["variation_id"],
                "variation_description": var["variation_description"],
                "image": var["image_bytes"],
            }
            record.update(annotation)
            rows.append(record)

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Build robust OCR benchmark dataset")
    parser.add_argument("--hf_repo", required=True,
                        help="HuggingFace repo id, e.g. myuser/robust-ocr-bench")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(n_samples=args.n_samples, seed=args.seed)

    print(f"\nDataset built: {len(dataset)} rows "
          f"({args.n_samples} receipts × {len(VARIATION_CONFIGS)} variations)")
    print(dataset)

    print(f"\nPushing to https://huggingface.co/datasets/{args.hf_repo} …")
    dataset.push_to_hub(args.hf_repo, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
