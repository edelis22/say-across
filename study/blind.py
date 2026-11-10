"""Produce randomized per-rater packets and a separate coordinator key."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from .run import load_items

METRICS = ("adequacy", "tone", "culture", "safety")
RATING_FIELDS = ("trial_id", *METRICS, "notes")


def prepare(items, outputs, raters, seed):
    rng = random.Random(seed)
    packets = {rater: [] for rater in raters}
    key = []
    by_id = {item["id"]: item["sentence"] for item in items}
    for rater, packet in packets.items():
        ordered = [row for row in outputs if row["result"]["output"]]
        rng.shuffle(ordered)
        for row in ordered:
            trial_id = f"{rng.getrandbits(128):032x}"
            packet.append({"trial_id": trial_id, **by_id[row["id"]],
                           "translation": row["result"]["output"]})
            key.append({"trial_id": trial_id, "id": row["id"], "arm": row["arm"],
                        "rater": rater})
    return packets, key


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--raters", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rater-prefix", default="rater", help="Unique cohort label for split runs.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.rater_prefix):
        parser.error("Use letters, numbers, underscores or hyphens in rater prefixes.")
    if args.raters < 2:
        parser.error("At least two raters are required.")
    if not (args.run_dir / "complete").exists():
        parser.error("The collection is incomplete.")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        parser.error("Dataset hash differs from the collection manifest.")
    items = load_items(args.dataset)
    outputs = read_jsonl(args.run_dir / "outputs.jsonl")
    raters = [f"{args.rater_prefix}_{index + 1}" for index in range(args.raters)]
    packets, key = prepare(items, outputs, raters, args.seed)
    args.packet_dir.mkdir(mode=0o700, parents=True)
    write_jsonl(args.packet_dir / "PRIVATE_key.jsonl", key)
    for rater, packet in packets.items():
        write_jsonl(args.packet_dir / f"{rater}.jsonl", packet)
        with (args.packet_dir / f"{rater}.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, RATING_FIELDS)
            writer.writeheader()
            writer.writerows({"trial_id": row["trial_id"]} for row in packet)


if __name__ == "__main__":
    main()
