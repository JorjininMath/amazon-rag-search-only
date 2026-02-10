import csv
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

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from p1_routing import detect_intent, rerank_candidates, to_float_or_none

GOLDEN_PATH = Path("data/eval/golden_queries.jsonl")
RESULTS_PATH = Path("data/eval/baseline_results.csv")
DETAILS_PATH = Path("data/eval/baseline_evidence.jsonl")
INDEX_DIR = Path("data/index")
TOP_K = 5


def load_golden_queries(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    by_id = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id"):
                by_id[row["id"]] = row
    return by_id


def main() -> None:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Missing {GOLDEN_PATH}")
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Missing {INDEX_DIR}. Run build_index.py first.")

    queries = load_golden_queries(GOLDEN_PATH)
    existing = load_existing_results(RESULTS_PATH)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(CACHE_DIR / "embeddings"),
    )
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=max(TOP_K * 3, TOP_K))

    output_rows = []
    detail_records = []

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        nodes = retriever.retrieve(query_text)
        intent = detect_intent(query_text)
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

        ranked = rerank_candidates(candidates, intent=intent, top_k=TOP_K)

        evidence = []
        low_rating_count = 0
        for row in ranked:
            rating = row.get("rating")
            if rating is not None and rating <= 2:
                low_rating_count += 1
            evidence.append(
                {
                    "score": to_float_or_none(row.get("score")),
                    "rerank_score": to_float_or_none(row.get("rerank_score")),
                    "asin": row.get("asin"),
                    "title": row.get("title"),
                    "rating": rating,
                    "text": row.get("text", ""),
                }
            )

        prev = existing.get(qid, {})
        output_rows.append(
            {
                "id": qid,
                "query": query_text,
                "top_k": str(TOP_K),
                "low_rating_count": str(low_rating_count),
                "meets_expectation": prev.get("meets_expectation", ""),
                "notes": prev.get("notes", ""),
            }
        )
        detail_records.append(
            {
                "id": qid,
                "query": query_text,
                "intent": q.get("intent", ""),
                "detected_intent": intent,
                "expected_rule": q.get("expected_rule", ""),
                "top_k": TOP_K,
                "low_rating_count": low_rating_count,
                "evidence": evidence,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "query",
                "top_k",
                "low_rating_count",
                "meets_expectation",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    with DETAILS_PATH.open("w", encoding="utf-8") as f:
        for r in detail_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Updated: {RESULTS_PATH}")
    print(f"Saved details: {DETAILS_PATH}")
    print(f"Total queries processed: {len(output_rows)}")


if __name__ == "__main__":
    main()
