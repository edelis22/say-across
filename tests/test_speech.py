import asyncio
import base64
import importlib.util
import json
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from pragmatics.emotion import EmotionReading
from pragmatics.prosody import Delivery
from pragmatics.speech import SpeechError, apply_vocal_emphasis, choose_spoken_language, decode_live_phrase
from pragmatics.voice_match import _decode_profile


@skipUnless(importlib.util.find_spec("gradio"), "Install Gradio for voice UI tests.")
class VoiceTests(TestCase):
    def setUp(self):
        self.enterContext(patch("app.speech_available", return_value=True))

    def test_recording_uses_the_transcript_and_speaks_the_translation(self):
        from app import build_app

        style = Delivery(150, -3, 30, 0.2)
        reading = EmotionReading("detected", ("calm",), mode="voice_prosody", scores={"calm": 0.8})
        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.load_recording", return_value="PCM fixture"), \
                patch("app.analyze_turn", return_value=(reading, style, [])) as voice, \
                patch("app.speak", return_value=(22050, b"fixture")) as playback:
            recognizer.return_value.transcribe_audio.return_value = "Good morning."
            nllb.return_value.translate_turn.return_value = "Buenos días."
            ui = build_app(detect_emotion=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "translate_recording")
            inputs = ["sample.wav", "eng_Latn", "spa_Latn", "", "", "unspecified", "es-MX",
                      "preserve", "", False]
            prediction = asyncio.run(ui.call_function(handler, inputs))["prediction"]
            self.assertEqual(prediction[:2], ("Good morning.", "Buenos días."))
            self.assertEqual(prediction[-2], (22050, b"fixture"))
            voice.assert_called_once_with("PCM fixture", enabled=False, source_language="eng_Latn")
            recognizer.return_value.transcribe_audio.assert_called_once_with("PCM fixture", "eng_Latn")
            self.assertIn("vocal expression", prediction[3].lower())
            nllb.return_value.translate_turn.assert_called_once_with("Good morning.", "eng_Latn", "spa_Latn")
            playback.assert_called_once_with("Buenos días.", "spa_Latn", "es-MX", style=style)

    def test_unreadable_recording_clears_audio_and_skips_translation(self):
        from app import build_app

        with patch("app.WhisperTranscriber"), patch("app.NLLB") as nllb, \
                patch("app.load_recording", side_effect=SpeechError("Try a WAV file.")):
            ui = build_app(detect_emotion=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "translate_recording")
            inputs = ["bad.wav", "eng_Latn", "spa_Latn", "", "", "unspecified", "",
                      "preserve", "", False]
            prediction = asyncio.run(ui.call_function(handler, inputs))["prediction"]
            self.assertEqual(prediction[-2:], (None, "Try a WAV file."))
            nllb.return_value.translate_turn.assert_not_called()

    def test_live_phrase_uses_pcm_without_the_remote_adapter(self):
        from app import build_app

        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.speak", side_effect=SpeechError("Voice unavailable.")) as playback, \
                patch("app.EmotionDetector") as wording, patch("app.OpenAIAdapter") as remote:
            recognizer.return_value.transcribe_audio.return_value = "Good morning."
            nllb.return_value.translate_turn.return_value = "Buenos días."
            ui = build_app(allow_remote=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 4000).decode()
            events = handler.fn(encoded, "eng_Latn", "spa_Latn", "expressive")
            preview = next(events)
            self.assertEqual(preview["stage"], "translated")
            self.assertEqual(preview["translation"], "Buenos días.")
            playback.assert_not_called()
            result, = events
            self.assertEqual(result["stage"], "ready")
            self.assertEqual(result["translation"], "Buenos días.")
            self.assertEqual(result["playback_error"], "Voice unavailable.")
            self.assertEqual(result["audio"], "")
            audio, language = recognizer.return_value.transcribe_audio.call_args.args
            self.assertEqual(len(audio), 4000)
            self.assertEqual(language, "eng_Latn")
            remote.assert_not_called()
            wording.return_value.detect.assert_not_called()
            self.assertEqual(result["emotion"], "")

    def test_matching_voice_receives_source_waveform_and_has_a_fallback(self):
        from app import build_app

        style = Delivery(190, 4, 62, 0.65)
        reading = EmotionReading("uncertain", mode="voice_prosody")
        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.VoiceMatch") as engine, patch("app.analyze_turn", return_value=(reading, style, [])), \
                patch("app.speak", return_value=(22050, np.array([0, 2], dtype="<i2"))) as playback:
            recognizer.return_value.transcribe_audio.return_value = "Good morning."
            nllb.return_value.translate_turn.return_value = "Buenos días."
            engine.return_value.synthesize.return_value = (24000, np.array([0, 3], dtype="<i2"))
            ui = build_app(clone_executable="engine")
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            _, result = handler.fn(encoded, "eng_Latn", "spa_Latn", "match")
            self.assertEqual(result["voice_engine"], "AI voice match")
            text, language, reference, selected_style = engine.return_value.synthesize.call_args.args
            self.assertEqual((text, language, selected_style), ("Buenos días.", "spa_Latn", style))
            self.assertIs(reference, recognizer.return_value.transcribe_audio.call_args.args[0])
            self.assertTrue(result["audio"])
            playback.assert_not_called()

            engine.return_value.synthesize.side_effect = SpeechError("Matching failed.")
            _, fallback = handler.fn(encoded, "eng_Latn", "spa_Latn", "match")
            self.assertEqual(fallback["voice_engine"], "Expressive voice")
            self.assertIn("Matching failed.", fallback["playback_error"])
            playback.assert_called_once_with("Buenos días.", "spa_Latn", style=style)

    def test_draft_caption_translates_without_synthesizing_or_analyzing_tone(self):
        from app import build_app

        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.analyze_turn") as prosody, patch("app.speak") as playback:
            recognizer.return_value.transcribe_audio.return_value = "Good morning."
            nllb.return_value.translate_turn.return_value = "صباح الخير."
            ui = build_app(allow_remote=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_caption")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            self.assertEqual(handler.fn(encoded, "eng_Latn", "arb_Arab"),
                             {"transcript": "Good morning.", "translation": "صباح الخير.",
                              "source_language": "eng_Latn", "target_language": "arb_Arab"})
            too_long = base64.b64encode(b"\x00\x01" * (16000 * 8 + 1)).decode()
            self.assertEqual(handler.fn(too_long, "eng_Latn", "arb_Arab"), {})
            recognizer.return_value.transcribe_audio.assert_called_once()
            prosody.assert_not_called()
            playback.assert_not_called()

    def test_same_language_captions_work_without_translation_or_audio(self):
        from app import build_app

        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.speech_available", return_value=False), patch("app.speak") as playback:
            recognizer.return_value.transcribe_audio.return_value = "Bonjour."
            ui = build_app(allow_remote=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            preview, final = handler.fn(encoded, "fra_Latn", "fra_Latn", "expressive")
            self.assertEqual(preview["translation"], "Bonjour.")
            self.assertEqual(preview["voice_mode"], "captions")
            self.assertFalse(final["audio"])
            nllb.return_value.translate_turn.assert_not_called()
            playback.assert_not_called()

    def test_caption_only_mode_keeps_translated_cues_without_playback(self):
        from app import build_app

        reading = EmotionReading("uncertain", mode="voice_prosody")
        style = Delivery(170, 0, 50, 0.4)
        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.analyze_turn", return_value=(reading, style, [(0.8, "pitch_up")])), \
                patch("app.speak") as playback:
            recognizer.return_value.transcribe_audio.return_value = "Good morning."
            nllb.return_value.translate_turn.return_value = "Bonjour."
            ui = build_app(allow_remote=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            preview, final = handler.fn(encoded, "eng_Latn", "fra_Latn", "captions")
            self.assertEqual(preview["cues"][0]["text"], "La voix devient plus aiguë")
            self.assertEqual(final["translation"], "Bonjour.")
            self.assertFalse(final["audio"])
            playback.assert_not_called()

    def test_live_phrase_rejects_malformed_pcm(self):
        for payload in ("invalid?!", base64.b64encode(b"a" * 7001).decode()):
            with self.subTest(payload_length=len(payload)), self.assertRaises(SpeechError):
                decode_live_phrase(payload)

    def test_personal_profile_is_used_only_in_the_assigned_direction(self):
        from app import build_app

        audio = (24000, np.array([0, 3], dtype="<i2"))
        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.VoiceMatch") as engine:
            nllb.return_value.translate_turn.side_effect = ["Buenos días.", "Good morning."]
            engine.return_value.synthesize_profiled.return_value = (audio, True)
            engine.return_value.synthesize.return_value = audio
            recognizer.return_value.transcribe_pair.side_effect = [("eng_Latn", "Good morning."),
                                                                  ("spa_Latn", "Buenos días.")]
            ui = build_app(allow_remote=False, clone_executable="engine")
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            settings = json.dumps({"two_way": True, "profile_language": "eng_Latn"})
            _, first = handler.fn(encoded, "eng_Latn", "spa_Latn", "match", settings, "voice sample")
            _, second = handler.fn(encoded, "eng_Latn", "spa_Latn", "match", settings, "voice sample")
            self.assertEqual(first["voice_engine"], "AI voice match · personal profile")
            self.assertEqual(second["voice_engine"], "AI voice match")
            engine.return_value.synthesize_profiled.assert_called_once()
            self.assertEqual(engine.return_value.synthesize_profiled.call_args.args[-1], "voice sample")
            self.assertEqual(engine.return_value.synthesize.call_args.args[1], "eng_Latn")

    def test_interpreter_reverses_direction_and_uses_the_selected_spoken_tone(self):
        from app import build_app

        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.speak", return_value=(22050, np.array([0, 2], dtype="<i2"))) as playback:
            recognizer.return_value.transcribe_pair.return_value = ("spa_Latn", "No me tomes el pelo.")
            nllb.return_value.translate_turn.return_value = "Don't try to fool me."
            ui = build_app(allow_remote=False)
            handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            settings = json.dumps({"two_way": True, "tone": "excited"})
            preview, final = handler.fn(encoded, "eng_Latn", "spa_Latn", "expressive", settings)
            self.assertEqual((preview["source_language"], preview["target_language"]), ("spa_Latn", "eng_Latn"))
            self.assertEqual(preview["transcript"], "No me tomes el pelo.")
            self.assertTrue(preview["context_note"])
            nllb.return_value.translate_turn.assert_called_once_with("No intentes engañarme.", "spa_Latn", "eng_Latn")
            self.assertEqual(playback.call_args.args[1], "eng_Latn")
            self.assertEqual(playback.call_args.kwargs["style"].rate, 194)
            self.assertTrue(final["audio"])

    def test_greek_thanks_preserves_the_original_and_uses_reviewed_captions_and_voice(self):
        from app import build_app

        with patch("app.WhisperTranscriber") as recognizer, patch("app.NLLB") as nllb, \
                patch("app.speak", return_value=(22050, np.array([0, 2], dtype="<i2"))) as playback:
            raw = "Ευχαριστώ πάρω πολύ."
            recognizer.return_value.transcribe_pair.return_value = ("ell_Grek", raw)
            ui = build_app(allow_remote=False)
            routes = {fn.api_name: fn.fn for fn in ui.fns.values()}
            encoded = base64.b64encode(b"\x00\x01" * 16000).decode()
            settings = json.dumps({"two_way": True})
            draft = routes["live_caption"](encoded, "eng_Latn", "ell_Grek", settings)
            preview, final = routes["live_phrase"](encoded, "eng_Latn", "ell_Grek", "expressive", settings)
            for result in (draft, preview, final):
                self.assertEqual((result["source_language"], result["target_language"]), ("ell_Grek", "eng_Latn"))
                self.assertEqual((result["transcript"], result["translation"]), (raw, "Thank you very much."))
            self.assertTrue(final["context_note"])
            nllb.return_value.translate_turn.assert_not_called()
            self.assertEqual(playback.call_args.args[:2], ("Thank you very much.", "eng_Latn"))


class VoicePlaybackTests(TestCase):
    def test_profile_bounds_and_trims_voice_audio(self):
        def encode(samples):
            return base64.b64encode(samples.astype("<i2").tobytes()).decode()

        signal = np.zeros(9 * 16000)
        signal[16000:5 * 16000] = 7000 * np.sin(np.arange(4 * 16000) * 0.1)
        self.assertTrue(4 * 16000 < len(_decode_profile(encode(signal))) < 5 * 16000)
        for unusable in (np.zeros(9 * 16000), np.ones(2 * 16000) * 7000, np.tile(signal, 2)):
            with self.subTest(length=len(unusable)), self.assertRaises(SpeechError):
                _decode_profile(encode(unusable))

    def test_unclear_or_outside_pair_language_abstains(self):
        self.assertEqual(choose_spoken_language({"en": 0.1, "es": 0.85}, "eng_Latn", "spa_Latn"), "spa_Latn")
        for scores in ({"en": 0.3, "es": 0.28}, {"en": 0.32, "es": 0.1, "fr": 0.55}):
            with self.subTest(scores=scores), self.assertRaises(SpeechError):
                choose_spoken_language(scores, "eng_Latn", "spa_Latn")

    def test_emphasis_follows_the_source_curve_without_clipping(self):
        samples = np.tile(np.array([-27000, 27000], dtype="<i2"), 12000)
        rising = Delivery(170, 0, 45, 0.4, (-3, 3))
        falling = Delivery(170, 0, 45, 0.4, (3, -3))
        rate, up = apply_vocal_emphasis((24000, samples), rising)
        _, down = apply_vocal_emphasis((24000, samples), falling)
        self.assertEqual((rate, len(up)), (24000, len(samples)))
        self.assertLess(abs(int(up[0])), abs(int(up[-1])))
        self.assertGreater(abs(int(down[0])), abs(int(down[-1])))
        self.assertLessEqual(np.abs(up.astype(np.int32)).max(), 32767)
        np.testing.assert_array_equal(np.sign(up), np.sign(samples))
