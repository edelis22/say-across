"""Collect outputs, metadata and operational failures in a new private run directory."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from pragmatics.adaptation import OpenAIAdapter, PROMPT, RemoteError
from pragmatics.config import EMOTION_MODEL, EMOTION_REVISION, NLLB_MODEL, NLLB_REVISION, PROMPT_VERSION
from pragmatics.emotion import EmotionDetector, POLICY_VERSION
from pragmatics.pipeline import Sentence, run
from pragmatics.translation import NLLB

from .baselines import GoogleNMT


class Capture:
    """Reuse the full-condition response to isolate the effect of the surface checks."""

    def __init__(self, adapter):
        self.adapter = adapter

    def adapt(self, fields):
        try:
            self.response = self.adapter.adapt(fields)
        except RemoteError as error:
            self.response = error
            raise
        return self.response

    def replay(self):
        response = self.response

        class Replay:
            def adapt(self, fields):
                if isinstance(response, RemoteError):
                    raise response
                return response

        return Replay()


def load_items(path):
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items or len({row["id"] for row in items}) != len(items):
        raise ValueError("Dataset needs at least one item and unique IDs.")
    for row in items:
        Sentence(**row["sentence"]).validate()
    return items


def collect(item, nllb, adapter=None, google=None, ablations=False, *, detector=None):
    sentence = Sentence(**item["sentence"])
    base = run(sentence, nllb)
    yield "nllb", asdict(base), base.nllb_ms
    if google:
        start = perf_counter()
        try:
            output = google.translate(sentence.source, sentence.source_language,
                                      sentence.target_language)
            yield "google_nmt", {"output": output, "status": "baseline_only"}, (
                perf_counter() - start) * 1000
        except RuntimeError as error:
            yield "google_nmt", {"output": "", "status": "failed", "reason": str(error)}, (
                perf_counter() - start) * 1000
    if adapter:
        capture = Capture(adapter)
        full = run(sentence, nllb, capture, baseline=base.output, emotion_detector=detector)
        yield "full", asdict(full), base.nllb_ms + full.emotion_ms + full.adaptation_ms
        if ablations:
            stripped = replace(sentence, before="", after="", locale="", tone="preserve",
                               relationship="unspecified", intended_emotion="")
            no_context = run(stripped, nllb, adapter, baseline=base.output,
                             emotion_detector=detector)
            yield "no_context", asdict(no_context), (
                base.nllb_ms + no_context.emotion_ms + no_context.adaptation_ms)
            unguarded = run(sentence, nllb, capture.replay(), baseline=base.output, guarded=False,
                            emotion_reading=full.emotion)
            yield "no_surface_guards", asdict(unguarded), None  # Replay timing is artificial.
            if detector is not None:
                # An identical request needs no second paid call.
                no_hint_adapter = adapter if full.emotion_hints else capture.replay()
                no_hint = run(sentence, nllb, no_hint_adapter, baseline=base.output)
                yield "no_emotion_signal", asdict(no_hint), (
                    base.nllb_ms + no_hint.adaptation_ms if full.emotion_hints else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--openai", action="store_true", help="Send dataset to OpenAI")
    parser.add_argument("--google", action="store_true", help="Send source sentences to Google")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection in adapted arms.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.ablations and not args.openai:
        parser.error("--ablations requires --openai")
    for enabled, key in ((args.openai, "OPENAI_API_KEY"), (args.google, "GOOGLE_TRANSLATE_API_KEY")):
        if enabled and not os.environ.get(key):
            parser.error(f"Set {key} in your shell first.")
    items = load_items(args.dataset)
    args.run_dir.mkdir(mode=0o700, parents=True)
    adapter = OpenAIAdapter() if args.openai else None
    google = GoogleNMT() if args.google else None
    translator = NLLB(args.device)
    detector = EmotionDetector() if args.openai and not args.no_emotion else None
    dependencies = ["torch", "transformers", "sentencepiece"]
    if detector is not None:
        dependencies += ["onnxruntime", "numpy", "tokenizers", "huggingface-hub"]
    manifest = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "nllb": NLLB_MODEL, "revision": NLLB_REVISION,
        "emotion_model": EMOTION_MODEL if detector else None,
        "emotion_revision": EMOTION_REVISION if detector else None,
        "emotion_policy": POLICY_VERSION if detector else None,
        "llm_requested": adapter.model if adapter else None,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "google": "Cloud Translation Basic v2 general NMT" if google else None,
        "device": args.device, "machine": platform.platform(),
        "python": platform.python_version(),
        "versions": {pkg: version(pkg) for pkg in dependencies},
        "ablations": args.ablations,
        "items": len(items),
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.run_dir / "outputs.jsonl").open("x", encoding="utf-8") as out:
        for index, item in enumerate(items):
            for arm, result, elapsed in collect(item, translator, adapter, google, args.ablations,
                                                detector=detector):
                out.write(json.dumps({"id": item["id"], "arm": arm, "result": result,
                                      "elapsed_ms": elapsed, "cold_start": index == 0},
                                     ensure_ascii=False) + "\n")
                out.flush()
    (args.run_dir / "complete").touch()


if __name__ == "__main__":
    main()
