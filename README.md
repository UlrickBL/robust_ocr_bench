# robust-ocr-bench

Benchmark for evaluating vision-language models on key information extraction from receipts, under various image degradations.

Source dataset: [nanonets/key_information_extraction](https://huggingface.co/datasets/nanonets/key_information_extraction)

## What it does

1. **`create_dataset.py`** — samples 50 receipts and creates 6 variations of each image, then pushes the result to HuggingFace.
2. **`evaluate_models.py`** — runs a model on the dataset and reports per-field and per-variation accuracy.

## Image variations

| ID | Description |
|----|-------------|
| `original` | No change |
| `rotate_90` | Rotated 90° |
| `rotate_180` | Rotated 180° |
| `rotate_20` | Tilted ~20° |
| `invert` | Colour-inverted |
| `noisy_bg` | Thumbnail on a noisy white canvas |

## Extracted fields

`date`, `doc_no_receipt_no`, `seller_address`, `seller_gst_id`, `seller_name`, `seller_phone`, `total_amount`, `total_tax`

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # for pushing/loading private datasets
```

## Usage

### 1. Build and push the dataset

```bash
python create_dataset.py --hf_repo YOUR_HF_USERNAME/robust-ocr-bench
```

### 2. Evaluate a model

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-... python evaluate_models.py \
  --dataset YOUR_HF_USERNAME/robust-ocr-bench \
  --provider anthropic \
  --model claude-sonnet-4-6

# OpenAI
OPENAI_API_KEY=sk-... python evaluate_models.py \
  --dataset YOUR_HF_USERNAME/robust-ocr-bench \
  --provider openai \
  --model gpt-4o

# Local (HuggingFace transformers)
python evaluate_models.py \
  --dataset YOUR_HF_USERNAME/robust-ocr-bench \
  --provider local \
  --model llava-hf/llava-1.5-7b-hf
```

Optional flags:
- `--variations original rotate_90` — evaluate only specific variations
- `--output results.json` — save full results to JSON
- `--delay 1.0` — seconds between API calls (default 0.5)
