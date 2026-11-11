from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pragmatics.captions import caption_cues
from pragmatics.config import LANGUAGES
from pragmatics.emotion import EmotionReading
from pragmatics.config import HIGHER_PITCH_FLOOR_HZ, LOWER_PITCH_CEILING_HZ
from pragmatics.interpreter import intended_delivery
from pragmatics.prosody import (
    LABELS, analyze_turn, delivery, detect_voice, features, load_model, pitch_voice_range, voice_pitch,
)


class ProsodyTests(TestCase):
    def test_silence_and_very_short_audio_abstain(self):
        for audio in (np.zeros(32000), np.ones(2000)):
            reading = detect_voice(audio)
            self.assertEqual(reading.status, "uncertain")
            self.assertEqual(reading.mode, "voice_prosody")

    def test_pitch_and_relative_energy_use_audio_only(self):
        time = np.arange(32000) / 16000
        audio = 0.15 * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * time)) * np.sin(
            2 * np.pi * (150 * time + 20 * time ** 2)
        )
        np.testing.assert_allclose(features(audio), features(audio * 0.5), atol=1e-7)
        reading = detect_voice(audio)
        self.assertEqual(len(reading.scores), 8)
        self.assertAlmostEqual(sum(reading.scores.values()), 1)

    def test_ambiguous_scores_abstain(self):
        with patch("pragmatics.prosody.features", return_value=np.zeros(14)), patch(
            "pragmatics.prosody.probabilities", return_value=np.ones(8) / 8
        ):
            self.assertEqual(detect_voice(np.zeros(32000)).status, "uncertain")

    def test_different_audio_changes_delivery_even_when_the_label_is_uncertain(self):
        time = np.arange(32000) / 16000
        steady = 0.15 * np.sin(2 * np.pi * 150 * time)
        animated = 0.15 * (0.7 + 0.3 * np.sin(2 * np.pi * 6 * time)) * np.sin(
            2 * np.pi * (150 * time + 35 * time ** 2)
        )
        with patch("pragmatics.prosody.probabilities", return_value=np.ones(8) / 8):
            _, steady_style, _ = analyze_turn(steady, source_language="spa_Latn")
            _, animated_style, _ = analyze_turn(animated, source_language="spa_Latn")
        self.assertGreater(animated_style.exaggeration, steady_style.exaggeration)
        self.assertGreater(animated_style.pitch_range, steady_style.pitch_range)

    def test_unvalidated_disgust_score_abstains_without_substituting_another_label(self):
        scores = np.full(len(LABELS), 0.01)
        scores[LABELS.index("disgust")] = 0.82
        scores[LABELS.index("calm")] = 0.12
        with patch("pragmatics.prosody.features", return_value=np.zeros(14)), \
                patch("pragmatics.prosody.probabilities", return_value=scores), \
                patch("pragmatics.prosody.load_model", return_value={"label_thresholds": {"calm": 0.11}}):
            reading = detect_voice(np.zeros(32000))
        self.assertEqual(reading.status, "uncertain")
        self.assertFalse(reading.labels)

    def test_playback_responds_to_delivery_independently_of_named_labels(self):
        time = np.arange(12800) / 16000
        wave = np.sin(2 * np.pi * 155 * time)
        soft_to_loud = np.concatenate((wave * 0.06, wave * 0.18))
        loud_to_soft = np.concatenate((wave * 0.18, wave * 0.06))
        with patch("pragmatics.prosody._classify", return_value=EmotionReading("detected", ("disgust",))):
            _, style, _ = analyze_turn(soft_to_loud)
        self.assertEqual(style, delivery(soft_to_loud))
        self.assertLess(style.emphasis_db[0], style.emphasis_db[-1])
        self.assertGreater(delivery(loud_to_soft).emphasis_db[0], delivery(loud_to_soft).emphasis_db[-1])

    def test_cues_track_changes_in_order_without_emotion_labels(self):
        time = np.arange(12800) / 16000
        quiet = 0.06 * np.sin(2 * np.pi * 130 * time)
        bright = 0.16 * np.sin(2 * np.pi * 195 * time)
        audio = np.concatenate((quiet, bright, quiet))
        reading, _, changes = analyze_turn(audio, source_language="spa_Latn")
        self.assertEqual(reading.status, "unavailable")
        self.assertEqual(changes, [(0.8, "louder"), (0.8, "pitch_up"), (1.6, "pitch_down"), (1.6, "softer")])
        self.assertEqual(analyze_turn(audio * 0.7)[2], changes)
        self.assertEqual(analyze_turn(np.tile(quiet, 3))[2], [])
        self.assertEqual(analyze_turn(audio, enabled=False)[2], [])

    def test_cue_labels_are_available_in_all_caption_languages(self):
        changes = [(0.8, cue) for cue in ("louder", "softer", "pitch_up", "pitch_down", "faster", "slower")]
        for target in LANGUAGES:
            with self.subTest(target=target):
                self.assertEqual(len({cue["text"] for cue in caption_cues(changes, target)}), len(changes))
        self.assertEqual(caption_cues([(0.8, "pitch_up")], "arb_Arab")[0]["text"], "ترتفع نبرة الصوت")

    def test_rhythm_cues_follow_changes_in_the_energy_envelope(self):
        time = np.arange(12800) / 16000
        pitch = np.sin(2 * np.pi * 155 * time)
        slow = 0.14 * (0.70 + 0.29 * np.cos(2 * np.pi * 3 * time)) * pitch
        quick = 0.14 * (0.70 + 0.29 * np.cos(2 * np.pi * 8 * time)) * pitch
        changes = analyze_turn(np.concatenate((slow, quick, slow)))[2]
        self.assertEqual(changes, [(0.8, "faster"), (1.6, "slower")])

    def test_model_records_disjoint_dev_and_test_actors(self):
        model = load_model()
        self.assertTrue(set(model["dev_actors"]).isdisjoint(model["test_actors"]))
        self.assertGreater(model["train_rows"], 0)


class VoiceRangeTests(TestCase):
    @staticmethod
    def voiced(f0, seconds=3):
        """A pulse train whose period the autocorrelation can actually find."""
        time = np.arange(16000 * seconds) / 16000
        return 0.2 * np.sign(np.sin(2 * np.pi * f0 * time)) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * time))

    def test_measured_pitch_tracks_the_recording(self):
        for f0 in (105, 128, 198, 240):
            measured, _ = voice_pitch(self.voiced(f0))
            self.assertAlmostEqual(measured, f0, delta=3, msg=f"{f0} Hz was read as {measured}")

    def test_bands_are_assigned_outside_the_overlap_and_abstain_inside_it(self):
        self.assertEqual(voice_pitch(self.voiced(112))[1], "lower")
        self.assertEqual(voice_pitch(self.voiced(215))[1], "higher")
        # Adult speaking ranges overlap here, so no band is claimed.
        self.assertEqual(voice_pitch(self.voiced(165))[1], "")

    def test_thresholds_bound_the_decision(self):
        self.assertEqual(pitch_voice_range(None), "")
        self.assertEqual(pitch_voice_range(LOWER_PITCH_CEILING_HZ), "lower")
        self.assertEqual(pitch_voice_range(HIGHER_PITCH_FLOOR_HZ), "higher")
        self.assertEqual(pitch_voice_range(LOWER_PITCH_CEILING_HZ + 0.1), "")
        self.assertEqual(pitch_voice_range(HIGHER_PITCH_FLOOR_HZ - 0.1), "")

    def test_too_little_voiced_audio_abstains(self):
        self.assertEqual(voice_pitch(np.zeros(32000)), (None, ""))
        self.assertEqual(voice_pitch(self.voiced(120, seconds=0.2)), (None, ""))

    def test_absolute_pitch_stays_out_of_the_classifier_features(self):
        """The published voice model must keep seeing only clip-relative pitch."""
        low, high = self.voiced(110), self.voiced(220)
        self.assertAlmostEqual(voice_pitch(low)[0] - voice_pitch(high)[0], -110, delta=4)
        # One octave apart, yet every classifier feature agrees: the 12-semitone
        # difference is removed before _summarize sees the track.
        np.testing.assert_allclose(features(low), features(high), atol=0.2)

    def test_the_band_reaches_playback_on_the_delivery_controls(self):
        self.assertEqual(delivery(self.voiced(115)).voice_range, "lower")
        self.assertEqual(analyze_turn(self.voiced(225))[1].voice_range, "higher")

    def test_an_explicit_tone_preset_keeps_the_measured_band(self):
        measured = analyze_turn(self.voiced(115))[1]
        self.assertEqual(intended_delivery(measured, "calm").voice_range, "lower")
