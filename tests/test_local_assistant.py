import json
from unittest import TestCase
from unittest.mock import patch

from pragmatics.assistant import AssistantError, validate_simplification
from pragmatics.local_assistant import (
    LocalAssistant, PREFERRED_MODELS, choose_model, installed_models, local_assistant_available,
)
from pragmatics.research import ResearchError, _passages, search


class FakeOllama:
    """Record the request and return a canned /api/chat reply."""

    def __init__(self, content, model="qwen2.5:7b", error=None):
        self.content = content
        self.model = model
        self.error = error
        self.requests = []

    def __call__(self, host, path, payload, timeout):
        self.requests.append({"host": host, "path": path, "payload": payload, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return {"model": self.model, "message": {"role": "assistant", "content": self.content},
                "done": True}

    @property
    def payload(self):
        return self.requests[-1]["payload"]


PASSAGES = [
    {"title": "Public transport in Athens", "url": "https://en.wikipedia.org/wiki/Public_transport_in_Athens",
     "extract": "Tickets are sold at kiosks and must be validated on board." * 3},
    {"title": "Athens Metro", "url": "https://en.wikipedia.org/wiki/Athens_Metro",
     "extract": "The metro shares a ticketing system with buses and trams." * 3},
]

GOOD_JSON = json.dumps({
    "rendered": "A camera on a tube looks inside your stomach.",
    "glossary": [{"term": "endoscopy", "meaning": "a camera on a thin tube"}],
    "omitted": "",
    "abstain": False,
})


class ModelSelectionTests(TestCase):
    def test_a_preferred_model_wins_over_whatever_else_is_installed(self):
        self.assertEqual(choose_model(("zephyr:latest", "llama3.1:8b")), "llama3.1:8b")
        self.assertEqual(choose_model(("qwen2.5:7b", "llama3.1:8b")), "qwen2.5:7b")

    def test_a_tagged_variant_of_a_preferred_family_is_accepted(self):
        self.assertEqual(choose_model(("qwen2.5:14b-instruct",)), "qwen2.5:14b-instruct")

    def test_any_installed_model_is_better_than_none(self):
        self.assertEqual(choose_model(("zephyr:latest",)), "zephyr:latest")
        self.assertIsNone(choose_model(()))

    def test_an_explicit_choice_overrides_the_preference_order(self):
        with patch.dict("os.environ", {"LOCAL_LLM_MODEL": "mymodel:latest"}, clear=False):
            self.assertEqual(choose_model(("qwen2.5:7b",)), "mymodel:latest")

    def test_a_stopped_ollama_reads_as_no_models_rather_than_an_error(self):
        with patch("pragmatics.local_assistant.urlopen", side_effect=OSError):
            self.assertEqual(installed_models("http://127.0.0.1:11434"), ())
            self.assertFalse(local_assistant_available())


class LocalSimplifyTests(TestCase):
    def test_the_request_is_schema_constrained_and_deterministic(self):
        ollama = FakeOllama(GOOD_JSON)
        candidate, model = LocalAssistant(model="qwen2.5:7b", transport=ollama).simplify(
            translation="We will perform an endoscopy.", language="eng_Latn", level="plain")
        payload = ollama.payload
        self.assertEqual(ollama.requests[-1]["path"], "/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["format"]["type"], "object")
        self.assertIn("rendered", payload["format"]["properties"])
        self.assertIsNone(validate_simplification(candidate))
        self.assertEqual(model, "qwen2.5:7b")

    def test_a_malformed_glossary_row_is_dropped_not_fatal(self):
        """A small model gets rows slightly wrong; the rewrite is still usable."""
        ollama = FakeOllama(json.dumps({
            "rendered": "A camera looks inside.",
            "glossary": ["endoscopy", {"term": "scope", "meaning": "a tube"}, {"term": "  "}],
            "omitted": "", "abstain": "false",
        }))
        candidate, _ = LocalAssistant(model="m", transport=ollama).simplify(
            translation="An endoscopy.", language="eng_Latn", level="simple")
        self.assertEqual(candidate["glossary"], [{"term": "scope", "meaning": "a tube"}])
        self.assertIs(candidate["abstain"], False)
        self.assertIsNone(validate_simplification(candidate))

    def test_unusable_output_is_reported_as_a_reason_code(self):
        for content, reason in ((" ", "empty_answer"), ("not json", "invalid_response")):
            with self.assertRaises(AssistantError) as caught:
                LocalAssistant(model="m", transport=FakeOllama(content)).simplify(
                    translation="Hello.", language="eng_Latn", level="plain")
            self.assertEqual(str(caught.exception), reason)

    def test_a_stopped_ollama_never_breaks_the_translation(self):
        ollama = FakeOllama("", error=AssistantError("local_model_unreachable"))
        with self.assertRaises(AssistantError) as caught:
            LocalAssistant(model="m", transport=ollama).simplify(
                translation="Hello.", language="eng_Latn", level="plain")
        self.assertEqual(str(caught.exception), "local_model_unreachable")

    def test_the_level_and_language_are_checked_before_the_model_runs(self):
        ollama = FakeOllama(GOOD_JSON)
        assistant = LocalAssistant(model="m", transport=ollama)
        for kwargs, reason in (({"level": "as_spoken", "language": "eng_Latn"}, "unsupported_level"),
                               ({"level": "plain", "language": "xx_Xx"}, "unsupported_language")):
            with self.assertRaises(AssistantError) as caught:
                assistant.simplify(translation="Hello.", **kwargs)
            self.assertEqual(str(caught.exception), reason)
        self.assertEqual(ollama.requests, [])


class LocalLookupTests(TestCase):
    def test_the_answer_is_grounded_in_the_passages_it_cites(self):
        ollama = FakeOllama("Buy a ticket at a kiosk and validate it on board.")
        answer, _ = LocalAssistant(model="m", transport=ollama,
                                   searcher=lambda q, lang: PASSAGES).look_up(
            question="How do I pay for a bus?", language="eng_Latn", place="Athens")
        fields = json.loads(ollama.payload["messages"][1]["content"])
        self.assertEqual([p["title"] for p in fields["passages"]],
                         ["Public transport in Athens", "Athens Metro"])
        self.assertNotIn("format", ollama.payload, "a prose answer needs no schema")
        self.assertEqual(answer["answer"], "Buy a ticket at a kiosk and validate it on board.")
        self.assertEqual(answer["sources"], [{"url": p["url"], "title": p["title"]} for p in PASSAGES])
        self.assertTrue(answer["searched"])

    def test_the_place_sharpens_the_search_query(self):
        seen = []

        def searcher(query, language):
            seen.append((query, language))
            return PASSAGES

        LocalAssistant(model="m", transport=FakeOllama("ok"), searcher=searcher).look_up(
            question="How do I pay for a bus?", language="spa_Latn", place="Athens")
        self.assertEqual(seen, [("How do I pay for a bus? Athens", "spa_Latn")])

    def test_no_reference_means_no_invented_answer(self):
        ollama = FakeOllama("made up")
        with self.assertRaises(AssistantError) as caught:
            LocalAssistant(model="m", transport=ollama, searcher=lambda q, lang: []).look_up(
                question="Anything?", language="eng_Latn")
        self.assertEqual(str(caught.exception), "no_reference_found")
        self.assertEqual(ollama.requests, [], "the model must not run without references")

    def test_an_unreachable_reference_source_is_reported(self):
        def searcher(query, language):
            raise ResearchError("research_unreachable")

        with self.assertRaises(AssistantError) as caught:
            LocalAssistant(model="m", transport=FakeOllama("x"), searcher=searcher).look_up(
                question="Anything?", language="eng_Latn")
        self.assertEqual(str(caught.exception), "research_unreachable")


class SubstitutionTests(TestCase):
    """The transport and reference source must stay patchable at the module level."""

    def test_module_level_patches_reach_an_instance_built_with_no_arguments(self):
        calls = []

        def transport(host, path, payload, timeout):
            calls.append(path)
            return {"model": "m", "message": {"content": "an answer"}, "done": True}

        with patch("pragmatics.local_assistant._post", transport), \
                patch("pragmatics.local_assistant.search", lambda q, lang: PASSAGES):
            answer, _ = LocalAssistant(model="m").look_up(question="How?", language="eng_Latn")
        self.assertEqual(answer["answer"], "an answer")
        self.assertEqual(calls, ["/api/chat"])


class LocalSayInTests(TestCase):
    def test_an_answer_is_translated_without_a_second_search(self):
        ollama = FakeOllama("Compra el billete antes de subir.")
        restated, _ = LocalAssistant(model="m", transport=ollama).say_in(
            answer="Buy a ticket before you board.", language="spa_Latn")
        fields = json.loads(ollama.payload["messages"][1]["content"])
        self.assertEqual(fields, {"answer": "Buy a ticket before you board.", "language": "Spanish"})
        self.assertEqual(restated, "Compra el billete antes de subir.")


class ResearchTests(TestCase):
    def test_passages_are_built_from_the_api_payload(self):
        payload = {"query": {"pages": [
            {"title": "Athens Metro", "extract": "  The metro  serves the city. " * 6, "index": 2},
            {"title": "Too short", "extract": "Short."},
            {"title": "Missing page", "missing": True},
            {"title": "Public transport", "extract": "Buses and trams run across the city. " * 4, "index": 1},
        ]}}
        passages = _passages(payload, "en.wikipedia.org")
        self.assertEqual([p["title"] for p in passages], ["Public transport", "Athens Metro"])
        self.assertEqual(passages[0]["url"], "https://en.wikipedia.org/wiki/Public_transport")
        self.assertNotIn("  ", passages[1]["extract"], "whitespace is collapsed")

    def test_either_api_response_shape_is_accepted(self):
        """formatversion 1 keys pages by id; formatversion 2 returns a list."""
        entry = {"title": "Athens Metro", "extract": "The metro serves the city. " * 6, "index": 1}
        for pages in (([entry]), ({"12345": entry})):
            passages = _passages({"query": {"pages": pages}}, "en.wikipedia.org")
            self.assertEqual([p["title"] for p in passages], ["Athens Metro"])

    def test_a_malformed_payload_yields_no_passages_rather_than_an_error(self):
        for payload in (None, {}, {"query": None}, {"query": {"pages": "nope"}}, {"query": {}}):
            self.assertEqual(_passages(payload, "en.wikipedia.org"), [])

    def test_a_blank_question_is_refused_before_any_request(self):
        with patch("pragmatics.research.urlopen") as opened:
            with self.assertRaises(ResearchError):
                search("   ")
            opened.assert_not_called()

    def test_a_non_english_language_falls_back_to_english(self):
        hosts = []

        def fake(host, parameters):
            hosts.append(host)
            return {"query": {"pages": []}}

        with patch("pragmatics.research._request", side_effect=fake):
            self.assertEqual(search("bus tickets", "spa_Latn"), [])
        self.assertEqual(hosts, ["es.wikipedia.org", "en.wikipedia.org"])

    def test_no_personal_detail_is_sent_in_the_user_agent(self):
        from pragmatics.research import USER_AGENT

        self.assertNotIn("@", USER_AGENT)
        self.assertTrue(USER_AGENT.startswith("ClickResearchPrototype/"))
