"""Evaluate automatic emotion detection without translation or API keys."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean

from pragmatics.config import EMOTION_MODEL, EMOTION_REVISION, LANGUAGES
from pragmatics.emotion import EmotionDetector, LABELS, POLICY_VERSION
from pragmatics.pipeline import validate_text

from .blind import read_jsonl


def load_cases(path):
    cases = read_jsonl(path)
    seen = set()
    for case in cases:
        validate_text(case["source"], case.get("before", ""), case.get("after", ""))
        if not case["id"] or case["id"] in seen or case["language"] not in LANGUAGES:
            raise ValueError("Use unique IDs and configured language codes.")
        seen.add(case["id"])
        for key in ("group", "stratum", "provenance"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"An emotion case is missing {key}.")
        gold = case["gold"]
        if gold is not None and (
            not isinstance(gold, list) or not gold or any(label not in LABELS for label in gold)
            or len(set(gold)) != len(gold) or ("neutral" in gold and len(gold) != 1)
        ):
            raise ValueError("Gold must be null, ['neutral'], or a unique set of emotion labels.")
    if not cases:
        raise ValueError("Enter at least one case.")
    return cases


def collect(cases, detector):
    for case in cases:
        source = case["source"]
        before, after = case.get("before", ""), case.get("after", "")
        source_only = detector.detect(source)
        with_context = detector.detect(source, before, after) if before or after else source_only
        yield {
            **{key: case[key] for key in ("id", "language", "group", "stratum", "gold", "provenance")},
            "sentence_only": asdict(source_only),
            "context_available": asdict(with_context),
        }


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def metrics(rows, condition):
    clear = [(set(row["gold"]), row[condition]) for row in rows if row["gold"] is not None]
    ambiguous = [row[condition] for row in rows if row["gold"] is None]
    scores = {label: [] for label in LABELS}
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    predicted_n = exact = 0
    for gold, reading in clear:
        predicted = set(reading["labels"])
        support.update(gold)
        tp.update(predicted & gold)
        fp.update(predicted - gold)
        fn.update(gold - predicted)
        predicted_n += bool(predicted)
        exact += predicted == gold
        for label, score in reading["scores"].items():
            scores[label].append((score - (label in gold)) ** 2)
    per_label = {
        label: {
            "support": support[label], "tp": tp[label], "fp": fp[label], "fn": fn[label],
            "precision": tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else None,
            "recall": tp[label] / support[label] if support[label] else None,
            "f1": f1(tp[label], fp[label], fn[label]),
            "brier": mean(scores[label]) if scores[label] else None,
        }
        for label in LABELS
    }
    supported = [per_label[label]["f1"] for label in LABELS if support[label]]
    return {
        "items": len(rows),
        "status_counts": dict(Counter(row[condition]["status"] for row in rows)),
        "clear_items": len(clear),
        "clear_label_coverage": predicted_n / len(clear) if clear else None,
        "clear_exact_set_accuracy": exact / len(clear) if clear else None,
        "clear_exact_given_prediction": exact / predicted_n if predicted_n else None,
        "clear_micro_f1": f1(tp.total(), fp.total(), fn.total()),
        "clear_macro_f1_gold_supported": mean(supported) if supported else None,
        "ambiguous_items": len(ambiguous),
        "ambiguous_uncertain_rate": (
            sum(reading["status"] == "uncertain" for reading in ambiguous) / len(ambiguous)
            if ambiguous else None
        ),
        "per_label": per_label,
    }


def summarize(rows):
    languages, strata, contrasts = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        languages[row["language"]].append(row)
        strata[row["stratum"]].append(row)
        if row["stratum"] == "contextual_contrast" and row["gold"] is not None:
            contrasts[row["group"]].append(row)
    contrast_groups = [group for group in contrasts.values() if len(group) > 1]

    report = {}
    for condition in ("sentence_only", "context_available"):
        report[condition] = {
            "overall": metrics(rows, condition),
            "by_language": {key: metrics(group, condition) for key, group in languages.items()},
            "by_stratum": {key: metrics(group, condition) for key, group in strata.items()},
            "contrast_groups": len(contrast_groups),
            "contrast_groups_all_correct": sum(
                all(set(row[condition]["labels"]) == set(row["gold"]) for row in group)
                for group in contrast_groups
            ),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    rows = list(collect(cases, EmotionDetector()))
    report = {
        "model": EMOTION_MODEL, "revision": EMOTION_REVISION, "policy": POLICY_VERSION,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "report": summarize(rows), "predictions": rows,
    }
    if any(row["sentence_only"]["scores"] for row in rows):
        report["versions"] = {
            name: version(name) for name in ("onnxruntime", "numpy", "tokenizers", "huggingface-hub")
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if all(row["sentence_only"]["status"] == "unavailable" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
