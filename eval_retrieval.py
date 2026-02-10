import csv
import json
from collections import defaultdict
from pathlib import Path

GOLDEN_PATH = Path("data/eval/golden_queries.jsonl")
EVIDENCE_PATH = Path("data/eval/baseline_evidence.jsonl")
OUT_CSV = Path("data/eval/p0_5_metrics.csv")
OUT_JSON = Path("data/eval/p0_5_summary.json")


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


def _eval_one(golden: dict, evidence_row: dict) -> dict:
    evidence = evidence_row.get("evidence", [])
    checks: list[bool] = []
    ranks: list[int] = []
    reasons: list[str] = []

    if "target_max_rating" in golden:
        rank = _first_rank_for_max_rating(evidence, float(golden["target_max_rating"]))
        ok = rank is not None
        checks.append(ok)
        if ok:
            ranks.append(rank)
        else:
            reasons.append("no_low_rating_match")

    if "target_keywords" in golden:
        rank = _first_rank_for_keywords(evidence, list(golden["target_keywords"]))
        ok = rank is not None
        checks.append(ok)
        if ok:
            ranks.append(rank)
        else:
            reasons.append("no_keyword_match")

    if golden.get("target_polarity_mix"):
        ok, rank = _polarity_mix_hit(evidence)
        checks.append(ok)
        if ok and rank is not None:
            ranks.append(rank)
        else:
            reasons.append("no_polarity_mix")

    logic = golden.get("target_logic", "all").lower()
    if not checks:
        hit = False
        reasons.append("no_target_defined")
    elif logic == "any":
        hit = any(checks)
    else:
        hit = all(checks)

    return {
        "id": golden["id"],
        "intent": golden.get("intent", ""),
        "query": golden["query"],
        "hit": 1 if hit else 0,
        # best_rank is only defined when final hit condition is satisfied.
        "best_rank": min(ranks) if (hit and ranks) else "",
        "reason": "ok" if hit else ";".join(reasons),
    }


def main() -> None:
    goldens = _load_jsonl(GOLDEN_PATH)
    evidence_rows = {r["id"]: r for r in _load_jsonl(EVIDENCE_PATH)}

    results: list[dict] = []
    for g in goldens:
        ev = evidence_rows.get(g["id"], {"evidence": []})
        results.append(_eval_one(g, ev))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "intent", "query", "hit", "best_rank", "reason"]
        )
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    hit = sum(r["hit"] for r in results)
    by_intent = defaultdict(list)
    by_intent_best_rank = defaultdict(list)

    best_ranks: list[int] = []
    for r in results:
        by_intent[r["intent"]].append(r["hit"])
        rank = r["best_rank"]
        if rank != "":
            rank = int(rank)
            best_ranks.append(rank)
            by_intent_best_rank[r["intent"]].append(rank)

    hit_at_1 = (
        sum(1 for rank in best_ranks if rank <= 1) / total if total else 0.0
    )
    hit_at_3 = (
        sum(1 for rank in best_ranks if rank <= 3) / total if total else 0.0
    )
    hit_at_5 = (
        sum(1 for rank in best_ranks if rank <= 5) / total if total else 0.0
    )
    avg_best_rank = sum(best_ranks) / len(best_ranks) if best_ranks else None

    summary = {
        "total_queries": total,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        # Keep legacy key for compatibility with previous notes/scripts.
        "hit_at_k": hit_at_5,
        "avg_best_rank": avg_best_rank,
        "by_intent": {
            intent: {
                "hit_at_1": (
                    sum(1 for rank in by_intent_best_rank[intent] if rank <= 1) / len(vals)
                    if vals
                    else 0.0
                ),
                "hit_at_3": (
                    sum(1 for rank in by_intent_best_rank[intent] if rank <= 3) / len(vals)
                    if vals
                    else 0.0
                ),
                "hit_at_5": (
                    sum(1 for rank in by_intent_best_rank[intent] if rank <= 5) / len(vals)
                    if vals
                    else 0.0
                ),
                "avg_best_rank": (
                    sum(by_intent_best_rank[intent]) / len(by_intent_best_rank[intent])
                    if by_intent_best_rank[intent]
                    else None
                ),
            }
            for intent, vals in by_intent.items()
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics: {OUT_CSV}")
    print(f"Saved summary: {OUT_JSON}")
    print(f"Hit@1: {summary['hit_at_1']:.3f}")
    print(f"Hit@3: {summary['hit_at_3']:.3f}")
    print(f"Hit@5: {summary['hit_at_5']:.3f} ({hit}/{total})")
    if summary["avg_best_rank"] is not None:
        print(f"Avg Best Rank: {summary['avg_best_rank']:.3f}")
    for intent, metrics in summary["by_intent"].items():
        print(
            f"- {intent}: "
            f"hit@1={metrics['hit_at_1']:.3f}, "
            f"hit@3={metrics['hit_at_3']:.3f}, "
            f"hit@5={metrics['hit_at_5']:.3f}, "
            f"avg_best_rank={metrics['avg_best_rank'] if metrics['avg_best_rank'] is not None else 'NA'}"
        )


if __name__ == "__main__":
    main()
