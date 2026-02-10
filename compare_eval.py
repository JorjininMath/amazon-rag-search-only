import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(after, before):
    a = _safe_float(after)
    b = _safe_float(before)
    if a is None or b is None:
        return None
    return a - b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval summaries before and after a change."
    )
    parser.add_argument(
        "--before",
        default="data/eval/p0_5_summary_before_p1.json",
        help="Path to baseline summary json.",
    )
    parser.add_argument(
        "--after",
        default="data/eval/p0_5_summary.json",
        help="Path to new summary json.",
    )
    parser.add_argument(
        "--out-json",
        default="data/eval/p1_vs_p0_5_delta.json",
        help="Output delta json path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)
    out_json = Path(args.out_json)

    if not before_path.exists():
        raise FileNotFoundError(f"Missing before summary: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"Missing after summary: {after_path}")

    before = _load_json(before_path)
    after = _load_json(after_path)

    overall = {
        "hit_at_1": {
            "before": before.get("hit_at_1"),
            "after": after.get("hit_at_1"),
            "delta": _delta(after.get("hit_at_1"), before.get("hit_at_1")),
        },
        "hit_at_3": {
            "before": before.get("hit_at_3"),
            "after": after.get("hit_at_3"),
            "delta": _delta(after.get("hit_at_3"), before.get("hit_at_3")),
        },
        "hit_at_5": {
            "before": before.get("hit_at_5"),
            "after": after.get("hit_at_5"),
            "delta": _delta(after.get("hit_at_5"), before.get("hit_at_5")),
        },
        "avg_best_rank": {
            "before": before.get("avg_best_rank"),
            "after": after.get("avg_best_rank"),
            "delta": _delta(after.get("avg_best_rank"), before.get("avg_best_rank")),
        },
    }

    before_intents = before.get("by_intent", {})
    after_intents = after.get("by_intent", {})
    all_intents = sorted(set(before_intents.keys()) | set(after_intents.keys()))

    by_intent = {}
    for intent in all_intents:
        b = before_intents.get(intent, {})
        a = after_intents.get(intent, {})
        by_intent[intent] = {
            "hit_at_1": {
                "before": b.get("hit_at_1"),
                "after": a.get("hit_at_1"),
                "delta": _delta(a.get("hit_at_1"), b.get("hit_at_1")),
            },
            "hit_at_3": {
                "before": b.get("hit_at_3"),
                "after": a.get("hit_at_3"),
                "delta": _delta(a.get("hit_at_3"), b.get("hit_at_3")),
            },
            "hit_at_5": {
                "before": b.get("hit_at_5"),
                "after": a.get("hit_at_5"),
                "delta": _delta(a.get("hit_at_5"), b.get("hit_at_5")),
            },
            "avg_best_rank": {
                "before": b.get("avg_best_rank"),
                "after": a.get("avg_best_rank"),
                "delta": _delta(a.get("avg_best_rank"), b.get("avg_best_rank")),
            },
        }

    result = {
        "before_file": str(before_path),
        "after_file": str(after_path),
        "overall": overall,
        "by_intent": by_intent,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved delta: {out_json}")
    print("Overall:")
    for k, row in overall.items():
        print(
            f"- {k}: before={row['before']} after={row['after']} delta={row['delta']}"
        )


if __name__ == "__main__":
    main()
