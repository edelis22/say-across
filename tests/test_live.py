"""Opt-in checks with installed dependencies, model weights and service credentials."""

import asyncio
import base64
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from pragmatics.adaptation import OpenAIAdapter
from pragmatics.config import LANGUAGES
from pragmatics.emotion import EmotionDetector, EmotionReading
from pragmatics.pipeline import Sentence, run
from pragmatics.translation import NLLB
from study.baselines import GoogleNMT


SPEECH_FIXTURES = {
    "eng_Latn": "Thank you for helping me.",
    "spa_Latn": "Gracias por ayudarme.",
    "ell_Grek": "Ευχαριστώ που με βοηθήσατε.",
    "jpn_Jpan": "手伝ってくれてありがとうございます。",
    "fra_Latn": "Merci de m’avoir aidé.",
    "deu_Latn": "Danke, dass Sie mir geholfen haben.",
    "ita_Latn": "Grazie per avermi aiutato.",
    "por_Latn": "Agradeço muito pela sua ajuda.",
    "nld_Latn": "Bedankt dat u me hebt geholpen.",
    "zho_Hans": "谢谢你帮助我。",
    "kor_Hang": "도와주셔서 감사합니다.",
    "arb_Arab": "شكرا لك على مساعدتي.",
}


def _recognizer_input(audio):
    from pragmatics.speech import load_recording, wav_base64

    with TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.wav"
        path.write_bytes(base64.b64decode(wav_base64(audio)))
        return load_recording(path)


class LiveChecks(TestCase):
    @skipUnless(importlib.util.find_spec("gradio"), "Install pinned Gradio for UI construction.")
    def test_gradio_builds(self):
        from app import build_app
        self.assertIn("Click", build_app(demo=True).title)

    @skipUnless(importlib.util.find_spec("gradio"), "Install pinned Gradio for endpoint checks.")
    def test_gradio_emotion_endpoint(self):
        from app import build_app

        with patch("app.EmotionDetector") as detector, patch("app.NLLB") as nllb:
            detector.return_value.detect.return_value = EmotionReading("detected", ("joy",))
            ui = build_app()
            route = next(fn for fn in ui.fns.values() if fn.api_name == "detect_emotion")
            result = asyncio.run(ui.call_function(route, ["We got the grant!", "", ""]))
            self.assertIn("joy", result["prediction"][2])
            detector.return_value.detect.assert_called_once_with("We got the grant!", "", "")
            nllb.return_value.translate.assert_not_called()

    @skipUnless(os.environ.get("RUN_EMOTION_SMOKE") == "1", "Emotion model download/cache is opt-in.")
    def test_emotion(self):
        detector = EmotionDetector()
        for text, label in (
            ("Thank you for sticking up for me when nobody else did.", "gratitude"),
            ("Me tiembla todo cada vez que oigo un ruido detrás de la puerta.", "fear"),
            ("Σ’ ευχαριστώ που έμεινες μαζί μου μέχρι να έρθει η αδελφή μου.", "gratitude"),
            ("犬が死んでから、家の中がやけに寂しい。", "sadness"),
        ):
            with self.subTest(text=text):
                reading = detector.detect(text)
                self.assertEqual(reading.status, "detected", reading.reason)
                self.assertIn(label, reading.labels)

    @skipUnless(os.environ.get("RUN_NLLB_SMOKE") == "1", "Model download/cache is opt-in.")
    def test_nllb(self):
        model = NLLB()
        for source, src, tgt in (("Good morning.", "eng_Latn", "jpn_Jpan"),
                                 ("Καλημέρα.", "ell_Grek", "spa_Latn"),
                                 ("Buenos días.", "spa_Latn", "ell_Grek"),
                                 ("おはようございます。", "jpn_Jpan", "eng_Latn")):
            with self.subTest(src=src, tgt=tgt):
                translated = model.translate(source, src, tgt)
                self.assertTrue(translated.strip())
                self.assertNotEqual(translated, source)

    @skipUnless(os.environ.get("RUN_SPEECH_SMOKE") == "1", "Local multilingual speech checks are opt-in.")
    def test_spoken_languages(self):
        import numpy as np
        from pragmatics.speech import MacVoice, SpeechError, WhisperTranscriber, speak

        native = MacVoice(os.environ["NATIVE_VOICE"]) if os.environ.get("NATIVE_VOICE") else None
        if native:
            self.addCleanup(native.close)
        turbo = os.environ.get("WHISPER_TURBO") == "1"
        recognizer = WhisperTranscriber(fast=os.environ.get("FAST_WHISPER") == "1", turbo=turbo)
        model = NLLB(turn_beams=1, fast_model=os.environ.get("FAST_NLLB_MODEL"))
        self.assertEqual(set(SPEECH_FIXTURES), set(LANGUAGES))
        for language, text in SPEECH_FIXTURES.items():
            with self.subTest(language=language):
                try:
                    rate, pcm = native.synthesize(text, language) if native else speak(text, language)
                except SpeechError:
                    rate, pcm = speak(text, language)
                self.assertGreater(len(pcm), rate // 3)
                self.assertTrue(np.any(pcm))
                transcript = recognizer.transcribe_audio(_recognizer_input((rate, pcm)), language)
                self.assertTrue(transcript.strip())
                if language != "eng_Latn":
                    self.assertTrue(model.translate_turn(transcript, language, "eng_Latn").strip())
                    self.assertTrue(model.translate_turn(SPEECH_FIXTURES["eng_Latn"], "eng_Latn", language).strip())
        if turbo:
            phrase = "Ευχαριστώ πάρα πολύ."
            audio = native.synthesize(phrase, "ell_Grek") if native else speak(phrase, "ell_Grek")
            direction, transcript = recognizer.transcribe_pair(_recognizer_input(audio), "eng_Latn", "ell_Grek")
            self.assertEqual(direction, "ell_Grek")
            self.assertEqual(transcript.strip(".!").casefold(), phrase.rstrip(".").casefold())

    @skipUnless(os.environ.get("VOICE_ENGINE"), "Set VOICE_ENGINE to the Python with local voice weights.")
    def test_voice_match(self):
        import numpy as np
        from pragmatics.prosody import delivery
        from pragmatics.speech import speak
        from pragmatics.voice_match import VoiceMatch

        voice = VoiceMatch(os.environ["VOICE_ENGINE"])
        self.addCleanup(voice.close)
        reference = _recognizer_input(speak(SPEECH_FIXTURES["eng_Latn"], "eng_Latn"))
        style = delivery(reference)
        voice.warmup()
        for language, text in SPEECH_FIXTURES.items():
            with self.subTest(language=language):
                rate, pcm = voice.synthesize(text, language, reference, style)
                self.assertEqual(rate, 24000)
                self.assertGreater(len(pcm), rate // 3)
                self.assertTrue(np.any(pcm))

    @skipUnless(os.environ.get("VOICE_ENGINE") and os.environ.get("NATIVE_VOICE"),
                "Set VOICE_ENGINE and NATIVE_VOICE for the synthetic profile check.")
    def test_personal_voice_profile(self):
        import numpy as np
        from pragmatics.prosody import Delivery, delivery
        from pragmatics.speech import MacVoice
        from pragmatics.voice_match import VoiceMatch, _encode_reference

        native = MacVoice(os.environ["NATIVE_VOICE"])
        engine = VoiceMatch(os.environ["VOICE_ENGINE"])
        self.addCleanup(native.close)
        self.addCleanup(engine.close)
        sample = "Hi, it's me. I'm glad we have time to talk today. What should we try next? I think this is a good place to start."
        reference = _recognizer_input(native.synthesize(sample, "eng_Latn", "en-US"))
        profile = _encode_reference(reference[:9 * 16000])
        sentence = "That's wonderful. I'm so glad you're here."
        for locale, tone, expected in (
            ("en-US", Delivery(143, -2, 38, 0.15), True),
            ("en-US", Delivery(198, 4, 65, 0.65), True),
            ("en-GB", Delivery(143, -2, 38, 0.15), False),
        ):
            with self.subTest(locale=locale, rate=tone.rate):
                signal = _recognizer_input(native.synthesize(sentence, "eng_Latn", locale, style=tone))
                audio, used = engine.synthesize_profiled("Me alegra mucho que estés aquí.", "spa_Latn", signal,
                                                        delivery(signal), profile)
                self.assertEqual(used, expected)
                self.assertEqual(audio[0], 24000)
                self.assertTrue(np.any(audio[1]))

    @skipUnless(os.environ.get("RUN_OPENAI_SMOKE") == "1", "Paid API check is opt-in.")
    def test_openai(self):
        result = run(Sentence("Good morning.", "eng_Latn", "spa_Latn", relationship="peer"),
                     None, OpenAIAdapter(), baseline="Buenos días.")
        self.assertEqual(result.status, "adapted", result.reason)
        self.assertTrue(result.response_model)

    @skipUnless(os.environ.get("RUN_GOOGLE_SMOKE") == "1", "Paid API check is opt-in.")
    def test_google(self):
        result = GoogleNMT().translate("Good morning.", "eng_Latn", "spa_Latn")
        self.assertTrue(result)
        self.assertNotEqual(result, "Good morning.")
