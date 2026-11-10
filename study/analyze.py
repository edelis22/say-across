"""Summarize completed blinded ratings with item-cluster bootstrap intervals."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

from .blind import METRICS, read_jsonl


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def bootstrap(differences, seed=2409, replicates=5000):
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    estimates = [mean(rng.choices(differences, k=len(differences))) for _ in range(replicates)]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def alpha(groups):
    """Krippendorff alpha with interval distance for ordered ratings and binary safety."""
    groups = [values for values in groups if len(values) >= 2]
    counts = Counter(value for values in groups for value in values)
    n = sum(counts.values())
    if n < 2:
        return None
    observed = sum(sum((a - b) ** 2 for a in values for b in values) / (len(values) - 1)
                   for values in groups) / n
    expected = sum(na * nb * (a - b) ** 2 for a, na in counts.items()
                   for b, nb in counts.items()) / (n * (n - 1))
    return 1 - observed / expected if expected else None


def summarize(outputs, ratings):
    by_arm = defaultdict(list)
    for row in outputs:
        by_arm[row["arm"]].append(row)
    report = {"arms": {}, "paired_full_minus_baseline": {}}
    for arm, rows in by_arm.items():
        latencies = [row["elapsed_ms"] for row in rows
                     if row["elapsed_ms"] is not None and not row["cold_start"]]
        judged = [ratings[(row["id"], arm)] for row in rows if ratings.get((row["id"], arm))]
        arm_report = {
            "outputs": len(rows),
            "status": dict(Counter(row["result"]["status"] for row in rows)),
            "reasons": dict(Counter(row["result"]["reason"] for row in rows if row["result"].get("reason"))),
            "rated_items": len(judged),
            "warm_latency_n": len(latencies),
            "warm_latency_p50_p95_ms": [quantile(latencies, p) for p in (0.5, 0.95)] if latencies else None,
            "item_macro_means": {m: mean(mean(r[m] for r in group.values()) for group in judged)
                                 if judged else None for m in METRICS},
        }
        calibration = []
        for row in rows:
            group = ratings.get((row["id"], arm), {})
            probability = row["result"].get("adequacy_probability")
            if probability is not None and len(group) >= 3:
                calibration.append((probability, int(sum(r["adequacy"] >= 4 for r in group.values())
                                                     > len(group) / 2)))
        bins = []
        for index in range(5):
            pairs = [(p, y) for p, y in calibration if min(int(p * 5), 4) == index]
            bins.append({"range": [index / 5, (index + 1) / 5], "n": len(pairs),
                         "predicted": mean(p for p, _ in pairs) if pairs else None,
                         "observed": mean(y for _, y in pairs) if pairs else None})
        arm_report["calibration"] = {
            "n": len(calibration),
            "brier": mean((p - y) ** 2 for p, y in calibration) if calibration else None,
            "bins": bins,
        }
        report["arms"][arm] = arm_report

    for baseline in by_arm.keys() - {"full"}:
        metrics = {}
        for metric in METRICS:
            differences = []
            for row in by_arm.get("full", []):
                full = ratings.get((row["id"], "full"), {})
                other = ratings.get((row["id"], baseline), {})
                shared = sorted(full.keys() & other.keys())
                if shared:
                    differences.append(mean(full[r][metric] - other[r][metric] for r in shared))
            metrics[metric] = {
                "paired_items": len(differences),
                "difference": mean(differences) if differences else None,
                "item_bootstrap_95": bootstrap(differences),
            }
        report["paired_full_minus_baseline"][baseline] = metrics

    report["krippendorff_alpha"] = {
        metric: alpha([[r[metric] for r in group.values()] for group in ratings.values()])
        for metric in METRICS
    }
    return report


def load_ratings(key, files):
    ratings = defaultdict(dict)
    seen = set()
    for path in files:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                trial = row["trial_id"]
                if trial in seen or trial not in key:
                    raise ValueError("Duplicate or unknown trial ID.")
                seen.add(trial)
                scores = [row[metric].strip() for metric in METRICS]
                if not any(scores):
                    continue
                scores = dict(zip(METRICS, map(int, scores)))
                if scores["safety"] not in (0, 1) or any(
                    not 1 <= scores[m] <= 5 for m in METRICS if m != "safety"
                ):
                    raise ValueError("A rating was outside its allowed range.")
                mapping = key[trial]
                ratings[(mapping["id"], mapping["arm"])][mapping["rater"]] = scores
    return ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("private_key", type=Path)
    parser.add_argument("ratings", nargs="+", type=Path)
    args = parser.parse_args()
    key = {row["trial_id"]: row for row in read_jsonl(args.private_key)}
    ratings = load_ratings(key, args.ratings)
    print(json.dumps(summarize(read_jsonl(args.outputs), ratings), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
