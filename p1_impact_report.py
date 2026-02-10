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
INDEX_DIR = Path("data/index")
OUT_CSV = Path("data/eval/p1_impact.csv")
TOP_K = 5


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _first_rank_for_max_rating(evidence: list[dict], max_rating: float) -> int | None:
    for i, e in enumerate(evidence, start=1):
        r = e.get("rating")
        if r is not None and float(r) <= max_rating:
            return i
    return None


def _first_rank_for_keywords(evidence: list[dict], keywords: list[str]) -> int | None:
    keys = [k.lower() for k in keywords]
    for i, e in enumerate(evidence, start=1):
        text = str(e.get("text", "")).lower()
        if any(k in text for k in keys):
            return i
    return None


def _polarity_mix_hit(evidence: list[dict]) -> tuple[bool, int | None]:
    neg_rank = None
    pos_rank = None
    for i, e in enumerate(evidence, start=1):
        r = e.get("rating")
        if r is None:
            continue
        r = float(r)
        if neg_rank is None and r <= 2:
            neg_rank = i
        if pos_rank is None and r >= 4:
            pos_rank = i
    hit = (neg_rank is not None) and (pos_rank is not None)
    if not hit:
        return False, None
    return True, min(neg_rank, pos_rank)


def _compute_best_rank(golden: dict, evidence: list[dict]) -> int | None:
    checks: list[bool] = []
    ranks: list[int] = []

    if "target_max_rating" in golden:
        rank = _first_rank_for_max_rating(evidence, float(golden["target_max_rating"]))
        ok = rank is not None
        checks.append(ok)
        if ok:
            ranks.append(rank)

    if "target_keywords" in golden:
        rank = _first_rank_for_keywords(evidence, list(golden["target_keywords"]))
        ok = rank is not None
        checks.append(ok)
        if ok:
            ranks.append(rank)

    if golden.get("target_polarity_mix"):
        ok, rank = _polarity_mix_hit(evidence)
        checks.append(ok)
        if ok and rank is not None:
            ranks.append(rank)

    logic = golden.get("target_logic", "all").lower()
    if not checks:
        return None
    if logic == "any":
        hit = any(checks)
    else:
        hit = all(checks)
    if not hit or not ranks:
        return None
    return min(ranks)


def _polarity(rating: float | None) -> str:
    if rating is None:
        return "unknown"
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def main() -> None:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Missing {GOLDEN_PATH}")
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Missing {INDEX_DIR}. Run build_index.py first.")

    goldens = _load_jsonl(GOLDEN_PATH)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(CACHE_DIR / "embeddings"),
    )
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=max(TOP_K * 3, TOP_K))

    rows: list[dict] = []

    for g in goldens:
        qid = g["id"]
        query_text = g["query"]
        intent = detect_intent(query_text)

        nodes = retriever.retrieve(query_text)
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

        before_top = candidates[:TOP_K]
        after_top = rerank_candidates(candidates, intent=intent, top_k=TOP_K)

        best_before = _compute_best_rank(g, before_top)
        best_after = _compute_best_rank(g, after_top)

        before_hit = 1 if best_before is not None else 0
        after_hit = 1 if best_after is not None else 0

        before_idx = (best_before - 1) if best_before is not None else None
        after_idx = (best_after - 1) if best_after is not None else None

        before_asin = before_top[before_idx]["asin"] if before_idx is not None else ""
        after_asin = after_top[after_idx]["asin"] if after_idx is not None else ""

        before_rating = (
            before_top[before_idx]["rating"] if before_idx is not None else None
        )
        after_rating = (
            after_top[after_idx]["rating"] if after_idx is not None else None
        )

        before_pol = _polarity(before_rating)
        after_pol = _polarity(after_rating)

        rows.append(
            {
                "id": qid,
                "intent": g.get("intent", ""),
                "query": query_text,
                "detected_intent": intent,
                "hit_before": before_hit,
                "hit_after": after_hit,
                "best_rank_before": best_before if best_before is not None else "",
                "best_rank_after": best_after if best_after is not None else "",
                "delta_rank": (
                    (best_before - best_after)
                    if best_before is not None and best_after is not None
                    else ""
                ),
                "asin_before": before_asin or "",
                "asin_after": after_asin or "",
                "asin_changed": (
                    "yes"
                    if before_asin and after_asin and before_asin != after_asin
                    else "no"
                ),
                "polarity_before": before_pol,
                "polarity_after": after_pol,
                "polarity_changed": (
                    "yes"
                    if before_pol != "unknown"
                    and after_pol != "unknown"
                    and before_pol != after_pol
                    else "no"
                ),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "intent",
                "query",
                "detected_intent",
                "hit_before",
                "hit_after",
                "best_rank_before",
                "best_rank_after",
                "delta_rank",
                "asin_before",
                "asin_after",
                "asin_changed",
                "polarity_before",
                "polarity_after",
                "polarity_changed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved impact report: {OUT_CSV}")


if __name__ == "__main__":
    main()

