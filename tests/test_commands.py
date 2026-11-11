from unittest import TestCase

from pragmatics.commands import parse_command


class WakePhraseTests(TestCase):
    def test_a_lookup_request_is_recognized_and_trimmed(self):
        for spoken, query in (
            ("Hey Click, please look up what the current policy is in Uganda.",
             "current policy is in uganda"),
            ("Hey, click. Look up the current policy in Uganda", "the current policy in uganda"),
            ("Click, tell me about the bus system here", "the bus system here"),
            ("Hey Click, what is the visa policy in Uganda?", "visa policy in uganda"),
        ):
            self.assertEqual(parse_command(spoken), ("lookup", query), msg=spoken)

    def test_mishearings_of_the_name_still_reach_the_app(self):
        """Recognizers rarely return the name spelled as intended."""
        for spoken in ("Hey Clique, what is the policy in Uganda?",
                       "Hey Klick, look up the bus fare",
                       "Hey Clic, tell me about visas"):
            command = parse_command(spoken)
            self.assertIsNotNone(command, msg=spoken)
            self.assertEqual(command[0], "lookup", msg=spoken)

    def test_a_request_to_explain_is_separated_from_a_question(self):
        for spoken in ("hey click explain that simpler",
                       "Click, explain that in simpler terms",
                       "Hey Click, what does that mean?",
                       "Hey Click, I don't understand"):
            self.assertEqual(parse_command(spoken), ("simplify", ""), msg=spoken)

    def test_ordinary_speech_is_never_taken_as_a_command(self):
        """A false positive silently swallows a sentence meant for translation."""
        for spoken in ("I need you to click the button",
                       "Could you please click on the link",
                       "Buenos dias, como estas?",
                       "We will perform an upper endoscopy under conscious sedation.",
                       "The click of the latch was loud",
                       ""):
            self.assertIsNone(parse_command(spoken), msg=spoken)

    def test_the_name_alone_asks_for_nothing(self):
        for spoken in ("Click", "Hey Click", "Hey Click, please"):
            self.assertIsNone(parse_command(spoken), msg=spoken)

    def test_the_name_must_come_at_the_start_of_the_turn(self):
        self.assertIsNone(parse_command("I was talking and then click look up the weather"))
