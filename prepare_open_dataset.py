import argparse
import json
import os
from pathlib import Path
from typing import Any

# Keep all external cache inside project for portability.
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _pick(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        v = record.get(k)
        if v is not None and v != "":
            return v
    return default


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    asin = _pick(record, ["asin", "parent_asin"], default="unknown")
    title = _pick(record, ["title", "product_title"], default="Unknown Product")
    brand = _pick(record, ["brand"], default="Unknown")
    category = _pick(record, ["category", "main_category"], default="Unknown")
    rating = _to_float(_pick(record, ["rating", "overall", "stars"]))
    text = _pick(record, ["text", "reviewText", "review_body", "content"], default="")
    if isinstance(category, list):
        category = " > ".join(str(x) for x in category if x)
    text = str(text).strip()
    if not text:
        return None

    return {
        "asin": str(asin),
        "title": str(title),
        "brand": str(brand),
        "category": str(category),
        "rating": rating if rating is not None else 0.0,
        "text": text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public Amazon reviews dataset.")
    parser.add_argument(
        "--dataset",
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="HuggingFace dataset id.",
    )
    parser.add_argument(
        "--config",
        default="raw_review_All_Beauty",
        help="HuggingFace dataset config/subset name.",
    )
    parser.add_argument("--split", default="full", help="Dataset split name.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of normalized rows to save.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/reviews_open.jsonl",
        help="Output jsonl path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    print(
        f"Loading dataset={args.dataset}, config={args.config}, split={args.split} (streaming)..."
    )
    stream = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    seen = set()
    with out_path.open("w", encoding="utf-8") as f:
        for record in stream:
            normalized = normalize_record(record)
            if not normalized:
                continue
            key = (normalized["asin"], normalized["text"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            saved += 1
            if saved >= args.max_samples:
                break

    print(f"✅ Saved {saved} samples to {out_path}")
    print("Next step: python src/build_index.py")


if __name__ == "__main__":
    main()
