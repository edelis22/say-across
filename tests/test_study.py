from dataclasses import asdict
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from pragmatics.emotion import EmotionReading
from study.analyze import alpha, summarize
from study.blind import prepare
from study.run import collect, load_items
from tests.test_pipeline import CANDIDATE, SENTENCE


class StudyTests(TestCase):
    def test_pilot_covers_all_ordered_pairs(self):
        rows = load_items(Path("study/data/pilot.jsonl"))
        pairs = {(row["sentence"]["source_language"], row["sentence"]["target_language"])
                 for row in rows}
        self.assertEqual(len(pairs), 12)

    def test_collection_blinding_analysis(self):
        item = {"id": "test-01", "sentence": asdict(SENTENCE)}
        translator = Mock()
        translator.translate.return_value = "2 copias."
        google = Mock()
        google.translate.return_value = "Envíe solamente 2 copias."
        adapter = Mock()
        adapter.adapt.side_effect = [({**CANDIDATE, "translation": "3 copias."}, "fixture"),
                                     (CANDIDATE, "fixture")]
        outputs = [{"id": item["id"], "arm": arm, "result": result, "elapsed_ms": ms,
                    "cold_start": False}
                   for arm, result, ms in collect(item, translator, adapter, google, True)]
        self.assertEqual(adapter.adapt.call_count, 2)
        self.assertEqual(translator.translate.call_count, 1)
        self.assertEqual(outputs[-1]["result"]["output"], "3 copias.")
        raters = ["a", "b", "c"]
        packets, key = prepare([item], outputs, raters, 27)
        self.assertEqual((packets, key), prepare([item], outputs, raters, 27))
        self.assertEqual(len({row["trial_id"] for row in key}), 15)
        self.assertNotIn("arm", packets["a"][0])
        self.assertNotIn("id", packets["a"][0])
        ratings = {(item["id"], row["arm"]): {
            rater: {"adequacy": 4, "tone": 3, "culture": 3, "safety": 1}
            for rater in raters} for row in outputs}
        report = summarize(outputs, ratings)
        self.assertEqual(report["arms"]["full"]["status"], {"fallback": 1})
        self.assertEqual(report["arms"]["no_context"]["calibration"]["n"], 1)
        self.assertAlmostEqual(report["arms"]["no_context"]["calibration"]["brier"], 0.04)
        self.assertEqual(report["paired_full_minus_baseline"]["nllb"]["tone"]["difference"], 0)

    def test_no_emotion_ablation_removes_only_the_detector_hint(self):
        item = {"id": "test-02", "sentence": asdict(SENTENCE)}
        translator = Mock()
        translator.translate.return_value = "2 copias."
        adapter = Mock()
        adapter.adapt.return_value = CANDIDATE, "fixture"
        detector = Mock()
        detector.detect.return_value = EmotionReading("detected", ("gratitude",), scores={"gratitude": 0.9})

        outputs = list(collect(item, translator, adapter, ablations=True, detector=detector))
        self.assertEqual(outputs[-1][0], "no_emotion_signal")
        self.assertEqual(adapter.adapt.call_count, 3)
        full = adapter.adapt.call_args_list[0].args[0]
        without_hint = adapter.adapt.call_args_list[-1].args[0]
        self.assertEqual(full["detected_emotions"], ["gratitude"])
        self.assertEqual(without_hint, {**full, "detected_emotions": []})

        adapter.reset_mock()
        detector.detect.return_value = EmotionReading("uncertain")
        outputs = list(collect(item, translator, adapter, ablations=True, detector=detector))
        self.assertEqual(adapter.adapt.call_count, 2)
        self.assertEqual(outputs[-1][1]["output"], outputs[1][1]["output"])
        self.assertIsNone(outputs[-1][2])

    def test_agreement(self):
        self.assertEqual(alpha([[1, 1, 1], [5, 5, 5]]), 1)
        self.assertIsNone(alpha([[1, 1, 1], [1, 1, 1]]))
        self.assertLess(alpha([[1, 5], [1, 5]]), 0)

    def test_blind_and_analyze_cli(self):
        import csv
        import hashlib
        import json
        import subprocess
        import sys
        from tempfile import TemporaryDirectory

        dataset = Path("study/data/pilot.jsonl")
        item = load_items(dataset)[0]
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "complete").touch()
            (base / "manifest.json").write_text(json.dumps({
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()
            }))
            outputs = base / "outputs.jsonl"
            outputs.write_text(json.dumps({
                "id": item["id"], "arm": "nllb", "result": {"output": "Ejemplo.",
                                                          "status": "baseline_only"},
                "elapsed_ms": 12, "cold_start": False
            }) + "\n")
            packets = base / "packets"
            subprocess.run([sys.executable, "-m", "study.blind", str(dataset), str(base),
                            str(packets), "--seed", "29"], check=True)
            ratings = packets / "rater_1.csv"
            with ratings.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, row = reader.fieldnames, next(reader)
            with ratings.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerow({**row, "adequacy": 5, "tone": 4, "culture": 4, "safety": 1})
            report = subprocess.check_output([sys.executable, "-m", "study.analyze", str(outputs),
                                               str(packets / "PRIVATE_key.jsonl"), str(ratings)])
            self.assertEqual(json.loads(report)["arms"]["nllb"]["item_macro_means"]["tone"], 4)
