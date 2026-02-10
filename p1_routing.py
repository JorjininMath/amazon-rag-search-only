from __future__ import annotations


NEGATIVE_KEYWORDS = {
    "negative",
    "bad",
    "worse",
    "worst",
    "dislike",
    "complaint",
    "complaints",
    "drawback",
    "drawbacks",
    "issue",
    "issues",
    "problem",
    "problems",
    "irritation",
    "allergy",
    "allergic",
    "garbage",
    "terrible",
    "disappointed",
}

COMPARE_KEYWORDS = {
    "compare",
    "comparison",
    "vs",
    "versus",
    "difference",
    "better than",
    "positive and negative",
}


def to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_intent(query: str) -> str:
    text = (query or "").strip().lower()
    if not text:
        return "general"

    if any(kw in text for kw in COMPARE_KEYWORDS):
        return "compare"
    if any(kw in text for kw in NEGATIVE_KEYWORDS):
        return "negative"
    return "general"


def _rating_boost(rating: float | None, intent: str) -> float:
    if rating is None:
        return 0.0
    if intent == "negative":
        # Prioritize low-rated evidence for complaint-oriented queries.
        return max(0.0, 3.0 - rating) * 0.35
    if intent == "compare":
        # Prefer stronger opinions (very positive or very negative) for compare.
        return abs(rating - 3.0) * 0.20
    return 0.0


def _polarity(rating: float | None) -> str:
    if rating is None:
        return "unknown"
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def _diversify_compare(items: list[dict], top_k: int) -> list[dict]:
    if not items:
        return []

    selected: list[dict] = []
    selected_asins: set[str] = set()
    selected_polarities: set[str] = set()
    remaining = items[:]

    def pick_best(predicate):
        best_idx = None
        best_score = -10**9
        for i, item in enumerate(remaining):
            if not predicate(item):
                continue
            score = float(item.get("rerank_score", 0.0))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            return None
        return remaining.pop(best_idx)

    # First, pick strongest non-neutral opinion as anchor.
    first = pick_best(lambda x: _polarity(x.get("rating")) in {"positive", "negative"})
    if first is None:
        first = pick_best(lambda x: True)
    if first is not None:
        selected.append(first)
        if first.get("asin"):
            selected_asins.add(first["asin"])
        selected_polarities.add(_polarity(first.get("rating")))

    # Then, try to pick opposite polarity and a different ASIN.
    if remaining and len(selected) < top_k:
        need_polarity = (
            "negative" if "positive" in selected_polarities else "positive"
        )
        second = pick_best(
            lambda x: _polarity(x.get("rating")) == need_polarity
            and x.get("asin") not in selected_asins
        )
        if second is None:
            second = pick_best(
                lambda x: _polarity(x.get("rating")) == need_polarity
            )
        if second is None:
            second = pick_best(lambda x: x.get("asin") not in selected_asins)
        if second is None:
            second = pick_best(lambda x: True)
        if second is not None:
            selected.append(second)
            if second.get("asin"):
                selected_asins.add(second["asin"])
            selected_polarities.add(_polarity(second.get("rating")))

    # Fill remaining slots with score + diversity bonus.
    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -10**9
        for i, item in enumerate(remaining):
            asin = item.get("asin")
            diversity_bonus = 0.12 if asin and asin not in selected_asins else 0.0
            polarity = _polarity(item.get("rating"))
            polarity_bonus = (
                0.10
                if polarity in {"positive", "negative"}
                and polarity not in selected_polarities
                else 0.0
            )
            score = float(item.get("rerank_score", 0.0)) + diversity_bonus + polarity_bonus
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        asin = chosen.get("asin")
        if asin:
            selected_asins.add(asin)
        selected_polarities.add(_polarity(chosen.get("rating")))

    return selected


def rerank_candidates(candidates: list[dict], intent: str, top_k: int) -> list[dict]:
    scored = []
    for c in candidates:
        score = float(c.get("score", 0.0))
        rating = c.get("rating")
        c = dict(c)
        c["rerank_score"] = score + _rating_boost(rating, intent)
        scored.append(c)

    scored.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)

    if intent == "compare":
        return _diversify_compare(scored, top_k)
    return scored[:top_k]
