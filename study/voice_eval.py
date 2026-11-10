"""Evaluate named vocal labels and acoustic intensity on held-out recordings."""

import argparse
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np

from pragmatics.prosody import (
    LABEL_MARGIN, LABELS, MARGIN, _delivery, choose_label, features, intensity_level, load_model, probabilities,
)
from study.train_prosody import load_rows, matched_intensity

EMODB_LABELS = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
                "F": "joy", "T": "sadness", "N": "neutral"}


@dataclass(frozen=True)
class Item:
    speaker: str
    truth: str
    match: tuple[str, ...] | None
    vector: np.ndarray | None


def ravdess_item(row):
    actor, label, _, vector, clip = row
    _, _, _, intensity, sentence, take, _ = clip.split("-")
    return Item(str(actor), LABELS[label], (str(actor), sentence, take) if intensity == "01" else None, vector)


def emodb_items(path):
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            file = Path(name)
            if file.suffix != ".wav" or file.parent.name != "wav":
                continue
            clip = file.stem
            with wave.open(BytesIO(archive.read(name))) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError(f"Unexpected audio format in {name}")
                signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            yield Item(clip[:2], EMODB_LABELS[clip[5]], (clip[:2], clip[2:5]), features(signal))


def summarize(items, predictions):
    confusion = {label: Counter() for label in sorted({item.truth for item in items})}
    matched = {}
    for item, prediction in zip(items, predictions, strict=True):
        confusion[item.truth][prediction or "abstain"] += 1
        if item.match is not None and item.truth in {"neutral", "disgust"}:
            matched.setdefault(item.match, {}).setdefault(item.truth, prediction)
    pairs = [pair for pair in matched.values() if len(pair) == 2]
    false_sources = {label: counts["disgust"] for label, counts in confusion.items() if label != "disgust"}
    return {
        "clips": len(items),
        "speakers": len({item.speaker for item in items}),
        "returned": sum(prediction is not None for prediction in predictions),
        "scored_returned": sum(prediction is not None and item.truth in LABELS
                               for item, prediction in zip(items, predictions, strict=True)),
        "correct_returned": sum(item.truth == prediction for item, prediction in zip(items, predictions, strict=True)),
        "true_disgust": confusion.get("disgust", {}).get("disgust", 0),
        "false_disgust": sum(false_sources.values()),
        "non_disgust": sum(item.truth != "disgust" for item in items),
        "false_disgust_by_label": false_sources,
        "neutral_disgust_pairs": len(pairs),
        "both_correct_pairs": sum(pair == {"neutral": "neutral", "disgust": "disgust"} for pair in pairs),
        "confusion": confusion,
    }


def evaluate(items, model):
    scores = [probabilities(item.vector, model) if item.vector is not None else None for item in items]
    policies = {
        "legacy": (dict.fromkeys(LABELS, model["threshold"]), MARGIN),
        "display_rule": (model.get("label_thresholds", {}), LABEL_MARGIN),
    }
    return {
        policy: summarize(items, [choose_label(row, thresholds, margin) if row is not None else None for row in scores])
        for policy, (thresholds, margin) in policies.items()
    }


def intensity_pairs(rows, model):
    matched = matched_intensity(rows)
    methods = {"heuristic": lambda vector: _delivery(vector).exaggeration}
    if "intensity" in model:
        methods["learned_score"] = lambda vector: intensity_level(vector, model)
        methods["blended_playback"] = lambda vector: _delivery(vector, intensity=intensity_level(vector, model)).exaggeration
    result = {}
    for method, score in methods.items():
        pairs = [(score(normal), score(strong)) for normal, strong in matched]
        result[method] = {"pairs": len(pairs), "strong_higher": sum(strong > normal for normal, strong in pairs),
                          "tied": sum(strong == normal for normal, strong in pairs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ravdess", type=Path)
    parser.add_argument("--emodb", type=Path)
    args = parser.parse_args()
    model = load_model()
    rows = load_rows(args.ravdess, set(model["test_actors"]))
    report = {f"ravdess_{window}": evaluate([ravdess_item(row) for row in rows if row[2] == window], model)
              for window in ("full", "live")}
    report["ravdess_matched_intensity"] = intensity_pairs(rows, model)
    if args.emodb:
        report["emodb_full"] = evaluate(list(emodb_items(args.emodb)), model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
