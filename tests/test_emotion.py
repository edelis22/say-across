from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from pragmatics.emotion import EmotionDetector, EmotionReading, LABELS, MAX_TOKENS, decide
from study.emotion_eval import collect, load_cases, summarize


def scores(**values):
    return {label: values.get(label, 0.01) for label in LABELS}


THRESHOLDS = dict.fromkeys(LABELS, 0.55)


class EmotionTests(TestCase):
    def test_mixed_affect_preserves_multiple_labels(self):
        reading = decide(scores(joy=0.87, gratitude=0.97), THRESHOLDS)
        self.assertEqual(reading.status, "detected")
        self.assertEqual(reading.labels, ("gratitude", "joy"))

    def test_weak_evidence_is_separate_from_neutral(self):
        self.assertEqual(decide(scores(neutral=0.8), THRESHOLDS).status, "neutral")
        self.assertEqual(decide(scores(neutral=0.4), THRESHOLDS).status, "uncertain")
        self.assertEqual(decide(scores(neutral=0.8, anger=0.79), THRESHOLDS).status, "uncertain")

    def test_context_can_change_decision_for_the_same_source(self):
        alone = scores(neutral=0.45, joy=0.4)
        happy = decide(alone, THRESHOLDS, scores(joy=0.9), scores(joy=0.3))
        annoyed = decide(alone, THRESHOLDS, scores(annoyance=0.9), scores(annoyance=0.6))
        self.assertEqual(happy.labels, ("joy",))
        self.assertEqual(annoyed.labels, ("annoyance",))
        self.assertEqual(annoyed.mode, "with_context")
        self.assertEqual(annoyed.sentence_scores, alone)

    def test_context_only_distractor_cannot_introduce_emotion_by_itself(self):
        reading = decide(scores(neutral=0.9), THRESHOLDS,
                         scores(anger=0.9, neutral=0.2), scores(anger=0.95))
        self.assertEqual((reading.status, reading.reason), ("uncertain", "emotion_also_in_context_alone"))

    def test_classifier_receives_only_text(self):
        model = Mock(thresholds=THRESHOLDS)
        model.encode.side_effect = lambda text: SimpleNamespace(ids=[1], text=text)
        model.score.side_effect = [scores(), scores(joy=0.9), scores(joy=0.3)]
        with patch("pragmatics.emotion._OnnxEmotionModel", return_value=model):
            reading = EmotionDetector().detect("Great.", "It worked.", "We can go home.")
        self.assertEqual(reading.labels, ("joy",))
        self.assertEqual([call.args[0] for call in model.encode.call_args_list],
                         ["Great.", "It worked.\nGreat.\nWe can go home.", "It worked.\nWe can go home."])

    def test_overlong_context_keeps_entire_sentence(self):
        model = Mock(thresholds=THRESHOLDS)
        short = SimpleNamespace(ids=[1, 2])
        long = SimpleNamespace(ids=list(range(MAX_TOKENS + 1)))
        model.encode.side_effect = [short, long, long, long]
        model.score.return_value = scores(fear=0.9)
        with patch("pragmatics.emotion._OnnxEmotionModel", return_value=model):
            detector = EmotionDetector()
            fallback = detector.detect("Short.", "Long context.")
            unavailable = detector.detect("Long sentence.")
        self.assertEqual((fallback.labels, fallback.reason), (("fear",), "context_over_token_limit"))
        self.assertEqual(unavailable.reason, "sentence_over_token_limit")
        model.score.assert_called_once_with(short)

    def test_unavailable_model_is_distinct_from_uncertain(self):
        with patch("pragmatics.emotion._OnnxEmotionModel", side_effect=ImportError):
            reading = EmotionDetector().detect("Hello")
        self.assertEqual(reading.status, "unavailable")

    def test_pilot_has_no_self_report_field(self):
        cases = load_cases(Path("study/data/emotion-pilot.jsonl"))
        self.assertEqual(len({case["language"] for case in cases}), 4)
        self.assertTrue(all("intended_emotion" not in case for case in cases))

    def test_evaluation_counts_abstention_and_matched_context_effect(self):
        cases = [
            {"id": f"item-{index}", "language": "eng_Latn", "group": "paired",
             "stratum": "contextual_contrast", "source": "Great.", "before": context,
             "gold": gold, "provenance": "fixture"}
            for index, (context, gold) in enumerate((("Good news.", ["joy"]), ("Bad news.", ["anger"]),
                                                     ("", None)))
        ]
        detector = Mock()
        detector.detect.side_effect = [
            EmotionReading("uncertain"), EmotionReading("detected", ("joy",)),
            EmotionReading("uncertain"), EmotionReading("detected", ("anger",)),
            EmotionReading("uncertain"),
        ]
        rows = list(collect(cases, detector))
        report = summarize(rows)
        self.assertEqual(report["sentence_only"]["overall"]["clear_label_coverage"], 0)
        self.assertEqual(report["context_available"]["overall"]["clear_exact_set_accuracy"], 1)
        self.assertEqual(report["context_available"]["overall"]["ambiguous_uncertain_rate"], 1)
        self.assertEqual(report["context_available"]["contrast_groups_all_correct"], 1)
