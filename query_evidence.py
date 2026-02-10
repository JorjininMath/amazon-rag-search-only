import argparse
import os
from pathlib import Path

# Keep all caches inside project directory.
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", str(CACHE_DIR / "llama_index"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR / "sentence_transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from p1_routing import detect_intent, rerank_candidates, to_float_or_none


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query top-k evidence from local index.")
    parser.add_argument("--query", required=True, help="Question text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks.")
    parser.add_argument("--index-dir", default="data/index", help="Index directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Missing index directory: {index_dir}")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(CACHE_DIR / "embeddings"),
    )

    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=max(args.top_k * 3, args.top_k))
    nodes = retriever.retrieve(args.query)

    if not nodes:
        print("No evidence found.")
        return

    intent = detect_intent(args.query)
    candidates = []
    for node in nodes:
        meta = node.metadata or {}
        candidates.append(
            {
                "score": to_float_or_none(getattr(node, "score", None)) or 0.0,
                "asin": meta.get("asin"),
                "title": meta.get("title"),
                "rating": to_float_or_none(meta.get("rating")),
                "text": (node.text or "").replace("\n", " ").strip(),
            }
        )
    ranked = rerank_candidates(candidates, intent=intent, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Intent: {intent}")
    print("--- Top Evidence ---")
    for i, row in enumerate(ranked, start=1):
        raw_score = row.get("score")
        rerank_score = row.get("rerank_score")
        raw_score_text = f"{float(raw_score):.4f}" if raw_score is not None else "-"
        rerank_score_text = (
            f"{float(rerank_score):.4f}" if rerank_score is not None else "-"
        )
        text = row.get("text", "")
        snippet = text[:180] + ("..." if len(text) > 180 else "")
        print(
            f"{i}. rerank_score={rerank_score_text} | raw_score={raw_score_text} | "
            f"asin={row.get('asin', '-')} | title={row.get('title', '-')} | "
            f"rating={row.get('rating', '-')}"
        )
        print(f"   {snippet}")


if __name__ == "__main__":
    main()
