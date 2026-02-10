import argparse
import json
import os
from pathlib import Path

# Keep all caches inside project directory.
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", str(CACHE_DIR / "llama_index"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR / "sentence_transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build index from reviews JSONL.")
    parser.add_argument(
        "--input",
        default="data/processed/reviews_open.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/index",
        help="Output index directory.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(CACHE_DIR / "embeddings"),
    )

    docs = []
    for r in load_jsonl(input_path):
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        meta = {
            "asin": r.get("asin"),
            "title": r.get("title"),
            "brand": r.get("brand"),
            "category": r.get("category"),
            "rating": r.get("rating"),
        }
        docs.append(Document(text=text, metadata=meta))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(out_dir))
    print(f"Saved index to {out_dir} with {len(docs)} documents.")


if __name__ == "__main__":
    main()
