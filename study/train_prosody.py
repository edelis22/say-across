"""Train and evaluate the voice-only model using the RAVDESS speech ZIP."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np

from pragmatics.prosody import FEATURE_VERSION, LABEL_MARGIN, LABELS, MARGIN, features, normalize_features, probabilities

DEV = {5, 6, 13, 14}
TEST = {7, 8, 19, 20}


def load_rows(path, actors=None):
    rows = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".wav") or not Path(name).name.startswith("03-01-"):
                continue
            clip = Path(name).stem
            fields = list(map(int, clip.split("-")))
            label, actor = fields[2] - 1, fields[-1]
            if actors is not None and actor not in actors:
                continue
            with wave.open(BytesIO(archive.read(name))) as wav:
                if wav.getframerate() != 48000 or wav.getsampwidth() != 2:
                    raise ValueError(f"Unexpected recording format in {name}")
                channels = wav.getnchannels()
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                if channels > 1:
                    pcm = pcm.reshape(-1, channels).mean(axis=1)
            usable = len(pcm) // 3 * 3
            audio = pcm[:usable].reshape(-1, 3).mean(axis=1).astype(np.float32) / 32768
            onset = np.flatnonzero(np.abs(audio) > max(0.001, np.max(np.abs(audio)) * 0.04))
            start = max(0, int(onset[0]) - 1600) if len(onset) else 0
            for window, signal in (("live", audio[start:start + 32000]), ("full", audio)):
                rows.append((actor, label, window, features(signal), clip))
    return rows


def fit(rows):
    x = np.stack([row[3] for row in rows])
    y = np.array([row[1] for row in rows])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
    z = np.clip((x - center) / scale, -5, 5)
    weights, bias = np.zeros((z.shape[1], len(LABELS))), np.zeros(len(LABELS))
    onehot = np.eye(len(LABELS))[y]
    for _ in range(700):
        logits = z @ weights + bias
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        gradient = (scores / scores.sum(axis=1, keepdims=True) - onehot) / len(y)
        weights -= 0.25 * (z.T @ gradient + 0.015 * weights)
        bias -= 0.25 * gradient.sum(axis=0)
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
            "bias": bias.tolist()}


def assess(rows, model, threshold):
    usable = [row for row in rows if row[3] is not None]
    y = np.array([row[1] for row in usable])
    scores = probabilities(np.stack([row[3] for row in usable]), model)
    chosen = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    accept = (ordered[:, -1] >= threshold) & (ordered[:, -1] - ordered[:, -2] >= MARGIN)
    count = int(accept.sum())
    return {"clips": len(rows), "usable": len(usable), "accepted": count,
            "correct_accepted": int(np.count_nonzero(accept & (chosen == y))),
            "top1_correct_usable": int(np.count_nonzero(chosen == y))}


def display_thresholds(rows, model):
    """Require support across development speakers before showing a named label."""
    usable = [row for row in rows if row[2] == "full" and row[3] is not None]
    if not usable:
        return {}
    truth = np.array([row[1] for row in usable])
    actors = np.array([row[0] for row in usable])
    scores = probabilities(np.stack([row[3] for row in usable]), model)
    order = scores.argsort(axis=1)
    best = order[:, -1]
    top_two = np.take_along_axis(scores, order[:, -2:], axis=1)
    separated = top_two[:, 1] - top_two[:, 0] >= LABEL_MARGIN
    thresholds = {}
    for index, label in enumerate(LABELS):
        for percent in range(40, 86, 5):
            floor = percent / 100
            accepted = (best == index) & (scores[:, index] >= floor) & separated
            if (accepted.sum() >= 12 and len(set(actors[accepted])) >= 3
                    and (truth[accepted] == index).mean() >= 0.80):
                thresholds[label] = floor
                break
    return thresholds


def matched_intensity(rows):
    paired = {}
    for actor, label, window, vector, clip in rows:
        if window != "full" or vector is None or label == 0:
            continue
        _, _, _, intensity, sentence, take, _ = clip.split("-")
        paired.setdefault((actor, label, sentence, take), {})[intensity] = vector
    return [(pair["01"], pair["02"]) for pair in paired.values() if len(pair) == 2]


def fit_intensity(rows, model):
    """Train on strong versus normal deliveries of the same speaker and sentence."""
    normalized = normalize_features(np.stack(matched_intensity(rows)), model)
    deltas = normalized[:, 1] - normalized[:, 0]
    weights = np.zeros(deltas.shape[1])
    for _ in range(400):
        gradient = (1 / (1 + np.exp(np.clip(deltas @ weights, -30, 30)))) @ deltas / len(deltas)
        weights += 0.12 * (gradient - 0.1 * weights)
    projected = normalized @ weights
    low, high = np.percentile(projected, [10, 90])
    return {"weights": weights.tolist(), "center": float(np.median(projected)),
            "scale": max(float((high - low) / 2), 1e-5), "train_pairs": len(deltas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.archive)
    train = [row for row in rows if row[0] not in DEV | TEST and row[3] is not None]
    dev = [row for row in rows if row[0] in DEV]
    test = [row for row in rows if row[0] in TEST]
    model = fit(train)
    candidates = []
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        metrics = assess(dev, model, threshold)
        if metrics["accepted"] >= 25:
            precision = metrics["correct_accepted"] / metrics["accepted"]
            candidates.append((threshold, metrics["accepted"], precision))
    passing = [candidate for candidate in candidates if candidate[2] >= 0.55]
    if passing:
        threshold = max(passing, key=lambda candidate: candidate[1])[0]
    elif candidates:
        threshold = max(candidates, key=lambda candidate: (candidate[2], candidate[1]))[0]
    else:
        threshold = 0.50
    with args.archive.open("rb") as stream:
        archive_sha = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            archive_sha.update(block)
    model.update({
        "labels": LABELS, "features_version": FEATURE_VERSION, "threshold": threshold,
        "label_thresholds": display_thresholds(dev, model),
        "intensity": fit_intensity(train, model),
        "training_source": "Livingstone and Russo, RAVDESS speech, DOI 10.5281/zenodo.1188976",
        "archive_sha256": archive_sha.hexdigest(), "license": "CC-BY-NC-SA-4.0",
        "train_rows": len(train), "dev_actors": sorted(DEV), "test_actors": sorted(TEST),
        "dev": assess(dev, model, threshold),
        "test": {window: assess([row for row in test if row[2] == window], model, threshold)
                 for window in ("live", "full")},
    })
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in ("threshold", "label_thresholds", "dev", "test")}, indent=2))


if __name__ == "__main__":
    main()
