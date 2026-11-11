import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from pragmatics.assistant import (
    AssistantError, ClaudeAssistant, LOOKUP_PROMPT, SAY_IN_PROMPT, SIMPLIFY_PROMPT,
    assistant_available, validate_request, validate_simplification,
)
from pragmatics.config import ASSISTANT_MODEL


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def reply(content, *, stop_reason="end_turn", model="claude-opus-5-5-fixture"):
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=model)


class FakeClient:
    """Stand in for anthropic.Anthropic and record the request it was given."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def request(self):
        return self.requests[-1]


GOOD_CANDIDATE = {
    "rendered": "We want to look inside your stomach with a small camera.",
    "glossary": [{"term": "endoscopy", "meaning": "a camera on a thin tube that looks inside the body"}],
    "omitted": "The original named the specific scope used.",
    "abstain": False,
}


class SimplificationValidationTests(TestCase):
    def test_a_well_formed_candidate_is_accepted(self):
        self.assertIsNone(validate_simplification(GOOD_CANDIDATE))

    def test_abstention_and_empty_output_are_rejected(self):
        self.assertEqual(validate_simplification({**GOOD_CANDIDATE, "abstain": True}), "model_abstained")
        self.assertEqual(validate_simplification({**GOOD_CANDIDATE, "rendered": "   "}), "empty_candidate")

    def test_wrong_shapes_are_rejected(self):
        for candidate in (
            None,
            "text",
            {key: value for key, value in GOOD_CANDIDATE.items() if key != "omitted"},
            {**GOOD_CANDIDATE, "extra": 1},
            {**GOOD_CANDIDATE, "rendered": 5},
            {**GOOD_CANDIDATE, "abstain": "no"},
            {**GOOD_CANDIDATE, "glossary": "endoscopy"},
            {**GOOD_CANDIDATE, "glossary": [{"term": "a"}]},
            {**GOOD_CANDIDATE, "glossary": [{"term": "a", "meaning": 2}]},
            {**GOOD_CANDIDATE, "rendered": "x" * 6001},
        ):
            self.assertEqual(validate_simplification(candidate), "invalid_schema",
                             msg=f"accepted {candidate!r}")

    def test_request_text_is_bounded(self):
        self.assertEqual(validate_request("  hello  ", limit=10, field="question"), "hello")
        for value, reason in ((" ", "empty_question"), ("x" * 11, "question_too_long"), (5, "invalid_question")):
            with self.assertRaises(AssistantError) as caught:
                validate_request(value, limit=10, field="question")
            self.assertEqual(str(caught.exception), reason)


class SimplifyTests(TestCase):
    def test_the_request_carries_the_schema_the_audience_and_a_fallback(self):
        client = FakeClient(reply([block("text", text=json.dumps(GOOD_CANDIDATE))]))
        candidate, model = ClaudeAssistant(client=client).simplify(
            translation="Realizaremos una endoscopia.", language="spa_Latn", level="plain",
            source="We will perform an endoscopy.", situation="hospital intake",
        )
        request = client.request
        self.assertEqual(request["model"], ASSISTANT_MODEL)
        self.assertEqual(request["system"], SIMPLIFY_PROMPT)
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", request["betas"])
        self.assertEqual(request["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("tools", request, "simplification must not reach the web")
        self.assertNotIn("thinking", request, "Opus 5.5 rejects an explicit thinking setting")
        fields = json.loads(request["messages"][0]["content"])
        self.assertEqual(fields["translation"], "Realizaremos una endoscopia.")
        self.assertEqual(fields["language"], "Spanish")
        self.assertIn("high-school", fields["audience"])
        self.assertEqual(candidate, GOOD_CANDIDATE)
        self.assertEqual(model, "claude-opus-5-5-fixture")

    def test_unsupported_level_and_language_never_reach_the_provider(self):
        client = FakeClient(reply([]))
        assistant = ClaudeAssistant(client=client)
        for kwargs, reason in (
            ({"level": "as_spoken", "language": "spa_Latn"}, "unsupported_level"),
            ({"level": "nope", "language": "spa_Latn"}, "unsupported_level"),
            ({"level": "plain", "language": "xxx_Xxxx"}, "unsupported_language"),
        ):
            with self.assertRaises(AssistantError) as caught:
                assistant.simplify(translation="Hola.", **kwargs)
            self.assertEqual(str(caught.exception), reason)
        self.assertEqual(client.requests, [])

    def test_a_policy_decline_is_reported_rather_than_read(self):
        client = FakeClient(reply([block("text", text="{}")], stop_reason="refusal"))
        with self.assertRaises(AssistantError) as caught:
            ClaudeAssistant(client=client).simplify(translation="Hola.", language="spa_Latn", level="plain")
        self.assertEqual(str(caught.exception), "provider_refusal")

    def test_unparseable_output_is_reported(self):
        client = FakeClient(reply([block("text", text="not json")]))
        with self.assertRaises(AssistantError) as caught:
            ClaudeAssistant(client=client).simplify(translation="Hola.", language="spa_Latn", level="plain")
        self.assertEqual(str(caught.exception), "invalid_response")

    def test_provider_failures_map_to_reason_codes(self):
        for error, reason in ((TimeoutError("slow"), "provider_or_parse_error"),
                              (type("RateLimitError", (Exception,), {})(), "rate_limited"),
                              (type("APIConnectionError", (Exception,), {})(), "provider_unreachable"),
                              (type("AuthenticationError", (Exception,), {})(), "credential_rejected")):
            client = FakeClient(error=error)
            with self.assertRaises(AssistantError) as caught:
                ClaudeAssistant(client=client).simplify(
                    translation="Hola.", language="spa_Latn", level="plain")
            self.assertEqual(str(caught.exception), reason)


class LookupTests(TestCase):
    def test_the_answer_and_its_sources_come_back_with_search_enabled(self):
        client = FakeClient(reply([
            block("web_search_tool_result", content=[
                SimpleNamespace(url="https://example.gr/tickets", title="Buying tickets"),
                SimpleNamespace(url="https://example.gr/tickets", title="Duplicate"),
            ]),
            block("text", text="Buy a ticket at the kiosk.  "),
            block("text", text="Validate it on board."),
        ]))
        answer, model = ClaudeAssistant(client=client).look_up(
            question="How do I pay for a bus ticket?", language="eng_Latn", place="Athens, Greece")
        request = client.request
        self.assertEqual(request["system"], LOOKUP_PROMPT)
        self.assertEqual(request["tools"][0]["type"], "web_search_20260209")
        self.assertNotIn("code_execution", json.dumps(request["tools"]),
                         "the search tool runs code itself; a second environment confuses the model")
        self.assertNotIn("format", request["output_config"])
        self.assertEqual(answer["answer"], "Buy a ticket at the kiosk.\n\nValidate it on board.")
        self.assertEqual(answer["sources"], [{"url": "https://example.gr/tickets", "title": "Buying tickets"}])
        self.assertTrue(answer["searched"])
        self.assertEqual(model, "claude-opus-5-5-fixture")

    def test_a_search_error_block_does_not_become_a_source(self):
        # A failed search returns one object where a successful one returns a list.
        client = FakeClient(reply([
            block("web_search_tool_result", content=SimpleNamespace(error_code="max_uses_exceeded")),
            block("text", text="I could not search, so here is the general picture."),
        ]))
        answer, _ = ClaudeAssistant(client=client).look_up(question="How do buses work?", language="eng_Latn")
        self.assertEqual(answer["sources"], [])
        self.assertTrue(answer["searched"])

    def test_an_answer_with_no_text_is_rejected(self):
        client = FakeClient(reply([block("text", text="   ")]))
        with self.assertRaises(AssistantError) as caught:
            ClaudeAssistant(client=client).look_up(question="Anything?", language="eng_Latn")
        self.assertEqual(str(caught.exception), "empty_answer")

    def test_a_blank_question_never_reaches_the_provider(self):
        client = FakeClient(reply([]))
        with self.assertRaises(AssistantError) as caught:
            ClaudeAssistant(client=client).look_up(question="   ", language="eng_Latn")
        self.assertEqual(str(caught.exception), "empty_question")
        self.assertEqual(client.requests, [])


class CredentialTests(TestCase):
    def test_availability_follows_the_environment(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}, clear=False):
            self.assertFalse(assistant_available())
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            self.assertTrue(assistant_available())

    def test_a_missing_credential_is_reported_before_any_client_is_built(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}, clear=False):
            with self.assertRaises(AssistantError) as caught:
                ClaudeAssistant().simplify(translation="Hola.", language="spa_Latn", level="plain")
            self.assertEqual(str(caught.exception), "missing_api_key")

    def test_the_model_can_be_overridden_from_the_environment(self):
        client = FakeClient(reply([block("text", text=json.dumps(GOOD_CANDIDATE))]))
        with patch.dict("os.environ", {"ASSISTANT_LLM_MODEL": "claude-opus-5-5-pinned"}, clear=False):
            ClaudeAssistant(client=client).simplify(
                translation="Hola.", language="spa_Latn", level="plain")
        self.assertEqual(client.request["model"], "claude-opus-5-5-pinned")


class SayInTests(TestCase):
    """Repeating an answer in another language must not re-answer the question."""

    def test_an_answer_is_restated_without_searching_again(self):
        client = FakeClient(reply([block("text", text="Compra el billete antes de subir.")]))
        restated, model = ClaudeAssistant(client=client).say_in(
            answer="Buy a ticket before you board.", language="spa_Latn")
        request = client.request
        self.assertEqual(request["system"], SAY_IN_PROMPT)
        self.assertNotIn("tools", request, "restating must not reach the web again")
        fields = json.loads(request["messages"][0]["content"])
        self.assertEqual(fields, {"answer": "Buy a ticket before you board.", "language": "Spanish"})
        self.assertEqual(restated, "Compra el billete antes de subir.")
        self.assertEqual(model, "claude-opus-5-5-fixture")

    def test_paragraphs_are_joined_and_blank_output_is_rejected(self):
        client = FakeClient(reply([block("text", text=" Primero. "), block("text", text="Despu\u00e9s.")]))
        restated, _ = ClaudeAssistant(client=client).say_in(answer="A. B.", language="spa_Latn")
        self.assertEqual(restated, "Primero.\n\nDespu\u00e9s.")
        blank = FakeClient(reply([block("text", text="  ")]))
        with self.assertRaises(AssistantError) as caught:
            ClaudeAssistant(client=blank).say_in(answer="A.", language="spa_Latn")
        self.assertEqual(str(caught.exception), "empty_answer")

    def test_an_empty_answer_or_unknown_language_never_reaches_the_provider(self):
        client = FakeClient(reply([]))
        assistant = ClaudeAssistant(client=client)
        with self.assertRaises(AssistantError) as caught:
            assistant.say_in(answer="  ", language="spa_Latn")
        self.assertEqual(str(caught.exception), "empty_answer")
        with self.assertRaises(AssistantError) as caught:
            assistant.say_in(answer="A.", language="xxx_Xxxx")
        self.assertEqual(str(caught.exception), "unsupported_language")
        self.assertEqual(client.requests, [])
