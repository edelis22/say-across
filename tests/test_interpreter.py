from unittest import TestCase

from pragmatics.interpreter import InterpreterSettings, local_interpretation, reviewed_spoken_translation


class InterpreterTests(TestCase):
    def test_reviewed_thanks_require_a_complete_turn_and_the_correct_direction(self):
        for text in ("Ευχαριστώ πάρα πολύ.", "Ευχαριστώ πάρω πολύ.", "ευχαριστώ παραπολή.",
                     "Ευχαριστώ, παραπολή!"):
            self.assertEqual(reviewed_spoken_translation(text, "ell_Grek", "eng_Latn"), "Thank you very much.")
            self.assertIsNone(reviewed_spoken_translation(text, "ell_Grek", "spa_Latn"))
        for text in ("Ευχάλιστο.", "Ευχαριστώ πάρα πολύ, αλλά δεν μπορώ.", "Ευχαριστώ, θα πάρω πολύ.",
                     "Ευχαριστώ πάρα πολύ;"):
            self.assertIsNone(reviewed_spoken_translation(text, "ell_Grek", "eng_Latn"))
        self.assertEqual(reviewed_spoken_translation("Muchas gracias.", "spa_Latn", "eng_Latn"), "Thank you very much.")

    def test_ambiguous_idiom_needs_compatible_context(self):
        def interpret(context, text="Break a leg!"):
            return local_interpretation(text, "eng_Latn", "spa_Latn", InterpreterSettings(situation=context))

        self.assertEqual(interpret("Before an audition")[1], "Good luck!")
        self.assertEqual(interpret("First aid after an injury")[1], "Break a leg!")
        self.assertEqual(interpret("Before an audition", "She could break a leg.")[1], "She could break a leg.")
        self.assertEqual(interpret("Showing the dessert", "It's a piece of cake.")[1], "It's a piece of cake.")
        self.assertEqual(interpret("About the homework", "It's a piece of cake.")[1], "It's very easy.")

    def test_explicit_relationship_changes_register_for_a_reviewed_phrase(self):
        close = InterpreterSettings(relationship="friend")
        formal = InterpreterSettings(relationship="supervisor")
        phrase = "Thank you for helping me."
        self.assertEqual(local_interpretation(phrase, "eng_Latn", "jpn_Jpan", close)[0], "手伝ってくれてありがとう。")
        self.assertEqual(local_interpretation(phrase, "eng_Latn", "jpn_Jpan", formal)[0],
                         "手伝ってくださってありがとうございます。")

    def test_procedural_language_resolves_only_with_explicit_context(self):
        phrase = "Let's table the motion."
        def interpret(context):
            return local_interpretation(phrase, "eng_Latn", "spa_Latn", InterpreterSettings(situation=context))[1]

        self.assertEqual(interpret("A US Senate procedure"), "Let's set the motion aside.")
        self.assertEqual(interpret("A UK Parliament procedure"), "Let's formally put forward the motion.")
        self.assertEqual(interpret("A meeting"), phrase)
