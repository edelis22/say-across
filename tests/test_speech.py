import asyncio
import base64
import importlib.util
import json
import subprocess
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from pragmatics.emotion import EmotionReading
from pragmatics.prosody import Delivery
from pragmatics.speech import (
    SpeechError, VOICES, apply_vocal_emphasis, choose_spoken_language, decode_live_phrase,
    installed_voices, resolve_voice_range, speak, voice_candidates,
)
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
            playback.assert_called_once_with("Buenos días.", "spa_Latn", "es-MX", style=style,
                                             voice_range="auto")

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
            playback.assert_called_once_with("Buenos días.", "spa_Latn", style=style,
                                             voice_range="auto")

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


class VoiceSelectionTests(TestCase):
    def setUp(self):
        installed_voices.cache_clear()
        self.addCleanup(installed_voices.cache_clear)

    def test_the_installed_voice_listing_is_parsed(self):
        listing = ("Alex                en_US    # Most people recognize me by my voice.\n"
                   "Grandma (Spanish (Mexico))  es_MX    # Hola, soy Grandma.\n"
                   "Ting-Ting           zh_CN    # 您好\n"
                   "unparseable line without a locale\n")
        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.subprocess.run",
                      return_value=SimpleNamespace(stdout=listing)):
            self.assertEqual(installed_voices(),
                             frozenset({"Alex", "Grandma (Spanish (Mexico))", "Ting-Ting"}))

    def test_an_unreadable_listing_does_not_block_playback(self):
        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.subprocess.run", side_effect=OSError):
            self.assertEqual(installed_voices(), frozenset())

    def test_no_requested_range_keeps_the_previous_default_voice(self):
        """Default playback must be byte-identical to the behaviour before ranges."""
        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.installed_voices", return_value=frozenset()):
            for locale, voice in VOICES.items():
                self.assertEqual(voice_candidates(locale, ""), (voice,))

    def test_absent_candidates_are_skipped_and_the_default_comes_last(self):
        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.installed_voices",
                      return_value=frozenset({"Tom", "Samantha", "Kate"})):
            # Alex is not installed on this host, so Tom is tried first.
            self.assertEqual(voice_candidates("en-US", "lower"), ("Tom", "Samantha"))
            self.assertEqual(voice_candidates("en-US", "higher"), ("Samantha",))
            # Daniel is the locale default and stays last even when absent.
            self.assertEqual(voice_candidates("en-GB", "higher"), ("Kate", "Daniel"))

    def test_espeak_renders_a_band_as_a_speaker_variant(self):
        with patch("pragmatics.speech.sys.platform", "linux"):
            self.assertEqual(voice_candidates("de-DE", "lower"), ("de+m3", "de"))
            self.assertEqual(voice_candidates("de-DE", "higher"), ("de+f3", "de"))
            self.assertEqual(voice_candidates("de-DE", ""), ("de",))
            self.assertEqual(voice_candidates("xx-XX", "lower"), ())

    def test_the_band_comes_from_the_recording_only_in_auto(self):
        measured = Delivery(168, 0, 45, 0.4, (), "higher")
        self.assertEqual(resolve_voice_range("auto", measured), "higher")
        self.assertEqual(resolve_voice_range("lower", measured), "lower")
        self.assertEqual(resolve_voice_range("default", measured), "")
        self.assertEqual(resolve_voice_range("auto", Delivery(168, 0, 45, 0.4)), "")
        self.assertEqual(resolve_voice_range("auto", None), "")

    def test_playback_falls_back_when_a_matched_voice_cannot_render(self):
        attempted = []

        def run(command, **kwargs):
            voice = command[command.index("-v") + 1]
            attempted.append(voice)
            if voice != "Samantha":
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(stdout=b"")

        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.installed_voices",
                      return_value=frozenset({"Alex", "Tom", "Samantha"})), \
                patch("pragmatics.speech.subprocess.run", side_effect=run), \
                patch("pragmatics.speech.read_pcm", return_value=(22050, b"fixture")):
            self.assertEqual(speak("Hello.", "eng_Latn", voice_range="lower"), (22050, b"fixture"))
        self.assertEqual(attempted, ["Alex", "Tom", "Samantha"])

    def test_playback_reports_the_default_voice_when_every_candidate_fails(self):
        with patch("pragmatics.speech.sys.platform", "darwin"), \
                patch("pragmatics.speech.installed_voices", return_value=frozenset({"Alex"})), \
                patch("pragmatics.speech.subprocess.run", side_effect=OSError):
            with self.assertRaises(SpeechError) as caught:
                speak("Hello.", "eng_Latn", voice_range="lower")
        self.assertIn("Samantha", str(caught.exception))


@skipUnless(importlib.util.find_spec("gradio"), "Install Gradio for assistant UI tests.")
class AssistantFeatureTests(TestCase):
    """The restating and lookup features sit beside the translation, never inside it."""

    def setUp(self):
        self.enterContext(patch("app.speech_available", return_value=True))
        self.enterContext(patch("app.assistant_available", return_value=True))
        self.assistant = self.enterContext(patch("app.ClaudeAssistant")).return_value

    def build(self, **kwargs):
        from app import build_app

        self.enterContext(patch("app.WhisperTranscriber")).return_value \
            .transcribe_audio.return_value = "Le haremos una endoscopia."
        self.enterContext(patch("app.NLLB")).return_value \
            .translate_turn.return_value = "We will perform an endoscopy on you."
        self.enterContext(patch("app.speak", return_value=(22050, np.array([0, 1], dtype="<i2"))))
        return build_app(allow_remote=False, **kwargs)

    @staticmethod
    def live_turn(ui, settings):
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_phrase")
        encoded = base64.b64encode(b"\x00\x01" * 8000).decode()
        events = list(handler.fn(encoded, "spa_Latn", "eng_Latn", "expressive",
                                 json.dumps(settings), ""))
        return events[0], events[-1]

    def test_a_turn_is_restated_beside_the_translation_at_the_selected_level(self):
        self.assistant.simplify.return_value = ({
            "rendered": "We want to look inside your stomach with a small camera.",
            "glossary": [{"term": "endoscopy", "meaning": "a camera on a thin tube"}],
            "omitted": "", "abstain": False,
        }, "claude-opus-5-5-fixture")
        preview, result = self.live_turn(self.build(), {"level": "plain"})
        self.assertEqual(preview["stage"], "translated")
        self.assertNotIn("explanation", preview, "captions must not wait for the restating")
        # The translation itself is untouched; the simpler wording is additional.
        self.assertEqual(result["translation"], "We will perform an endoscopy on you.")
        self.assertEqual(result["explanation"]["rendered"],
                         "We want to look inside your stomach with a small camera.")
        self.assertEqual(result["explanation"]["glossary"],
                         [{"term": "endoscopy", "meaning": "a camera on a thin tube"}])
        self.assertFalse(result["explanation"]["check_numbers"])
        request = self.assistant.simplify.call_args.kwargs
        self.assertEqual(request["translation"], "We will perform an endoscopy on you.")
        self.assertEqual(request["source"], "Le haremos una endoscopia.")
        self.assertEqual((request["language"], request["level"]), ("eng_Latn", "plain"))

    def test_the_default_level_leaves_the_existing_path_completely_untouched(self):
        preview, result = self.live_turn(self.build(), {})
        self.assertEqual(result["translation"], "We will perform an endoscopy on you.")
        self.assertEqual(result["explanation"], {})
        self.assertEqual(result["level"], "as_spoken")
        self.assistant.simplify.assert_not_called()

    def test_changed_quantities_are_flagged_rather_than_hidden(self):
        self.assistant.simplify.return_value = (
            {"rendered": "Take 3 tablets.", "glossary": [], "omitted": "", "abstain": False}, "fixture")
        with patch("app.NLLB") as nllb:
            nllb.return_value.translate_turn.return_value = "Take 2 tablets."
            _, result = self.live_turn(self.build(), {"level": "simple"})
        self.assertTrue(result["explanation"]["check_numbers"])
        self.assertEqual(result["explanation"]["rendered"], "Take 3 tablets.")

    def test_an_abstention_keeps_the_translation_and_explains_itself(self):
        self.assistant.simplify.return_value = (
            {"rendered": "", "glossary": [], "omitted": "", "abstain": True}, "fixture")
        _, result = self.live_turn(self.build(), {"level": "simple"})
        self.assertEqual(result["translation"], "We will perform an endoscopy on you.")
        self.assertIn("faithfully", result["explanation"]["error"])

    def test_a_provider_failure_is_reported_in_plain_words(self):
        from pragmatics.assistant import AssistantError

        self.assistant.simplify.side_effect = AssistantError("provider_unreachable")
        _, result = self.live_turn(self.build(), {"level": "plain"})
        self.assertEqual(result["translation"], "We will perform an endoscopy on you.")
        self.assertIn("Could not reach", result["explanation"]["error"])

    def test_a_lookup_returns_the_answer_in_the_asked_language_with_its_sources(self):
        self.assistant.look_up.return_value = (
            {"answer": "Buy a ticket at the kiosk before boarding.",
             "sources": [{"url": "https://example.gr/tickets", "title": "Tickets"}],
             "searched": True}, "claude-opus-5-5-fixture")
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_lookup")
        result = handler.fn("How do I pay for a bus?", "eng_Latn", "Athens, Greece", "at the bus stop")
        self.assertEqual(result["answer"], "Buy a ticket at the kiosk before boarding.")
        self.assertEqual(result["sources"], [{"url": "https://example.gr/tickets", "title": "Tickets"}])
        self.assertTrue(result["searched"])
        self.assertEqual(result["error"], "")
        request = self.assistant.look_up.call_args.kwargs
        self.assertEqual(request["question"], "How do I pay for a bus?")
        self.assertEqual(request["language"], "eng_Latn")
        self.assertEqual(request["place"], "Athens, Greece")

    def test_an_answer_can_be_repeated_in_another_language(self):
        self.assistant.say_in.return_value = ("Compra el billete antes de subir.", "fixture")
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_answer_language")
        result = handler.fn("Buy a ticket before you board.", "spa_Latn")
        self.assertEqual(result["answer"], "Compra el billete antes de subir.")
        self.assertEqual(result["language"], "spa_Latn")
        self.assertEqual(result["error"], "")
        self.assertEqual(self.assistant.say_in.call_args.kwargs,
                         {"answer": "Buy a ticket before you board.", "language": "spa_Latn"})
        self.assistant.look_up.assert_not_called()

    def test_an_answer_is_read_aloud_in_the_language_it_is_written_in(self):
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_answer_audio")
        with patch("app.speak", return_value=(22050, np.array([0, 1], dtype="<i2"))) as playback:
            result = handler.fn("Compra el billete.", "spa_Latn")
        self.assertTrue(result["audio"])
        playback.assert_called_once_with("Compra el billete.", "spa_Latn")

    def test_reading_aloud_refuses_empty_and_oversized_text(self):
        from pragmatics.assistant import MAX_PASSAGE_CHARS

        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_answer_audio")
        self.assertIn("nothing", handler.fn("  ", "spa_Latn")["error"])
        self.assertIn("too long", handler.fn("x" * (MAX_PASSAGE_CHARS + 1), "spa_Latn")["error"])
        self.assertIn("configured language", handler.fn("Hola.", "xxx_Xxxx")["error"])

    def test_a_failed_lookup_reports_a_readable_reason(self):
        from pragmatics.assistant import AssistantError

        self.assistant.look_up.side_effect = AssistantError("rate_limited")
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "live_lookup")
        result = handler.fn("Anything?", "eng_Latn", "", "")
        self.assertEqual(result["answer"], "")
        self.assertIn("rate limited", result["error"])

    def test_the_text_tools_restate_the_adapted_translation_when_there_is_one(self):
        self.assistant.simplify.return_value = (
            {"rendered": "A camera looks inside.", "glossary": [], "omitted": "", "abstain": False}, "fixture")
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "restate")
        rendered, detail = handler.fn("Literal wording.", "Adapted wording.", "eng_Latn", "plain", "")
        self.assertEqual(rendered, "A camera looks inside.")
        self.assertEqual(detail["status"], "rendered")
        self.assertEqual(self.assistant.simplify.call_args.kwargs["translation"], "Adapted wording.")

    def test_the_text_tools_need_a_translation_first(self):
        ui = self.build()
        handler = next(fn for fn in ui.fns.values() if fn.api_name == "restate")
        rendered, detail = handler.fn("", "", "eng_Latn", "plain", "")
        self.assertEqual(rendered, "")
        self.assertEqual(detail["status"], "input_error")
        self.assistant.simplify.assert_not_called()


@skipUnless(importlib.util.find_spec("gradio"), "Install Gradio for assistant UI tests.")
class AssistantDisabledTests(TestCase):
    def test_without_a_credential_the_features_are_absent_rather_than_broken(self):
        from app import build_app

        with patch("app.speech_available", return_value=True), patch("app.WhisperTranscriber"), \
                patch("app.NLLB"), patch("app.assistant_available", return_value=False):
            ui = build_app(allow_remote=False)
        names = {fn.api_name for fn in ui.fns.values()}
        self.assertNotIn("live_lookup", names)
        self.assertNotIn("restate", names)
        self.assertIn("live_phrase", names)

    def test_the_features_can_be_switched_off_even_with_a_credential_present(self):
        """Covers --no-assistant and a hosted deployment that has not opted in."""
        from app import build_app

        with patch("app.speech_available", return_value=True), patch("app.WhisperTranscriber"), \
                patch("app.NLLB"), patch("app.assistant_available", return_value=True), \
                patch("app.ClaudeAssistant") as built:
            ui = build_app(allow_remote=False, allow_assistant=False)
        names = {fn.api_name for fn in ui.fns.values()}
        self.assertNotIn("live_lookup", names)
        self.assertNotIn("live_answer_language", names)
        self.assertNotIn("restate", names)
        built.assert_not_called()
