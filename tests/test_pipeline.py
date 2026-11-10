from dataclasses import replace
import json
import os
from unittest import TestCase
from unittest.mock import Mock, patch

from pragmatics.adaptation import OpenAIAdapter
from pragmatics.demo import DemoAdapter, DemoTranslator, SOURCE
from pragmatics.emotion import EmotionReading
from pragmatics.pipeline import Sentence, run

SENTENCE = Sentence("Only send 2 copies.", "eng_Latn", "spa_Latn", locale="es-MX")
CANDIDATE = {
    "translation": "Envíe solo 2 copias.",
    "explanation": "Uses a polite imperative.",
    "ambiguity": "",
    "abstain": False,
    "adequacy_probability": 0.8,
}


class PipelineTests(TestCase):
    def setUp(self):
        self.translator = Mock()
        self.translator.translate.return_value = "Envíe únicamente 2 copias."
        self.adapter = Mock()
        self.adapter.adapt.return_value = (CANDIDATE.copy(), "test-model")

    def test_baseline_only(self):
        result = run(SENTENCE, self.translator)
        self.assertEqual(result.status, "baseline_only")
        self.assertEqual(result.output, result.baseline)

    def test_preference_and_context_passed_separately(self):
        sentence = replace(SENTENCE, before="Message to a vendor.", intended_emotion="concern")
        detector = Mock()
        detector.detect.return_value = EmotionReading("detected", ("anger",), scores={"anger": 0.85})
        result = run(sentence, self.translator, self.adapter, emotion_detector=detector)
        self.assertEqual(result.status, "adapted")
        self.assertEqual(result.intended_emotion, "concern")
        self.assertEqual(result.emotion.labels, ("anger",))
        detector.detect.assert_called_once_with(sentence.source, sentence.before, "")
        self.assertEqual(self.adapter.adapt.call_args.args[0]["detected_emotions"], [])
        self.translator.translate.assert_called_once_with(SENTENCE.source, "eng_Latn", "spa_Latn")
        self.assertEqual(self.adapter.adapt.call_args.args[0]["before"], sentence.before)

    def test_automatic_emotion_label_with_blank_intent(self):
        detector = Mock()
        detector.detect.return_value = EmotionReading("detected", ("gratitude", "joy"), scores={"gratitude": 0.92, "joy": 0.42})
        result = run(SENTENCE, self.translator, emotion_detector=detector)
        self.assertEqual(result.emotion.labels, ("gratitude", "joy"))
        self.assertEqual(result.status, "baseline_only")

        result = run(SENTENCE, self.translator, self.adapter, emotion_detector=detector)
        self.assertEqual(result.status, "adapted")
        self.assertEqual(self.adapter.adapt.call_args.args[0]["detected_emotions"], ["gratitude"])

    def test_only_detected_labels_can_influence_adaptation(self):
        for status, labels in (("uncertain", ()), ("unavailable", ()), ("neutral", ("neutral",))):
            reading = EmotionReading(status, labels)
            result = run(SENTENCE, self.translator, self.adapter, emotion_reading=reading)
            self.assertEqual(result.status, "adapted")
            self.assertEqual(result.emotion.status, status)
            self.assertEqual(self.adapter.adapt.call_args.args[0]["detected_emotions"], [])

    def test_surface_check_and_ablation(self):
        candidate = {**CANDIDATE, "translation": "Envíe solo 3 copias."}
        self.adapter.adapt.return_value = (candidate, "test-model")
        result = run(SENTENCE, self.translator, self.adapter)
        self.assertEqual((result.status, result.reason), ("fallback", "numeric_surface_mismatch"))
        self.assertEqual(result.output, result.baseline)
        self.assertEqual(result.candidate, candidate["translation"])
        result = run(SENTENCE, self.translator, self.adapter, guarded=False)
        self.assertEqual(result.output, candidate["translation"])

    def test_unicode_numbers(self):
        candidate = {**CANDIDATE, "translation": "Envíe solo ２ copias."}
        self.adapter.adapt.return_value = (candidate, "test-model")
        self.assertEqual(run(SENTENCE, self.translator, self.adapter).status, "adapted")

    def test_url_change(self):
        candidate = {**CANDIDATE, "translation": "Envíe solo 2 copias. https://example.net"}
        self.adapter.adapt.return_value = (candidate, "test-model")
        self.assertEqual(run(SENTENCE, self.translator, self.adapter).reason, "url_surface_mismatch")

    def test_malformed_or_abstaining_model_output(self):
        for candidate in (None, {}, {**CANDIDATE, "adequacy_probability": True},
                          {**CANDIDATE, "adequacy_probability": float("nan")},
                          {**CANDIDATE, "abstain": True}):
            with self.subTest(candidate=candidate):
                self.adapter.adapt.return_value = (candidate, "test-model")
                self.assertEqual(run(SENTENCE, self.translator, self.adapter).status, "fallback")

    def test_invalid_input_avoids_model_calls(self):
        for sentence in (replace(SENTENCE, source=" "), replace(SENTENCE, locale="ja-JP"),
                         replace(SENTENCE, target_language="eng_Latn")):
            with self.assertRaises(ValueError):
                run(sentence, self.translator, self.adapter)
        self.translator.translate.assert_not_called()
        self.adapter.adapt.assert_not_called()

    def test_fixtures_cover_all_demo_targets(self):
        for target in ("spa_Latn", "ell_Grek", "jpn_Jpan"):
            result = run(Sentence(SOURCE, "eng_Latn", target), DemoTranslator(), DemoAdapter())
            self.assertEqual(result.status, "adapted")
            self.assertNotEqual(result.baseline, result.output)

    def test_missing_key_and_remote_failure(self):
        transport = Mock(side_effect=OSError("credential-like error detail"))
        with patch.dict(os.environ, {}, clear=True):
            result = run(SENTENCE, self.translator, OpenAIAdapter(transport=transport))
            self.assertEqual(result.reason, "missing_api_key")
            transport.assert_not_called()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}):
            result = run(SENTENCE, self.translator, OpenAIAdapter(transport=transport))
            self.assertEqual(result.reason, "provider_or_parse_error")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake"})
    def test_responses_wire_contract(self):
        response = {"status": "completed", "model": "test-snapshot", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(CANDIDATE)}]}
        ]}
        transport = Mock(return_value=response)
        result = run(SENTENCE, self.translator, OpenAIAdapter(model="test-model", transport=transport))
        self.assertEqual(result.response_model, "test-snapshot")
        url, headers, payload = transport.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer fake")
        self.assertNotIn("fake", json.dumps(payload))
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])
        response["output"][0]["content"] = [{"type": "refusal", "refusal": "private detail"}]
        self.assertEqual(run(SENTENCE, self.translator, OpenAIAdapter(transport=transport)).reason,
                         "provider_refusal")
