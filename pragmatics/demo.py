"""Fixed examples for offline interface verification only."""

from .adaptation import RemoteError

SOURCE = "Could we move our meeting to Friday?"
TARGETS = {
    "spa_Latn": ("¿Podríamos mover nuestra reunión al viernes?",
                 "¿Podríamos pasar nuestra reunión al viernes?"),
    "ell_Grek": ("Θα μπορούσαμε να μεταφέρουμε τη συνάντησή μας για την Παρασκευή;",
                 "Θα ήταν δυνατό να μεταφέρουμε τη συνάντησή μας για την Παρασκευή;"),
    "jpn_Jpan": ("会議を金曜日に移せますか？", "会議を金曜日に変更していただけますか。"),
}


class DemoTranslator:
    def translate(self, text, source, target):
        if text != SOURCE or source != "eng_Latn" or target not in TARGETS:
            raise ValueError("Fixture mode requires the preset English sentence and a non-English target.")
        return TARGETS[target][0]


class DemoAdapter:
    def adapt(self, fields):
        if any(fields[key] for key in ("before", "after", "locale", "intended_emotion")) or (
            fields["relationship"] != "unspecified" or fields["tone"] != "preserve"
        ):
            raise RemoteError("fixture_context_unsupported")
        return {
            "translation": TARGETS[fields["target_language"]][1],
            "explanation": "This fixed example illustrates an alternative phrasing of the request.",
            "ambiguity": "",
            "abstain": False,
            "adequacy_probability": 0.75,
        }, "fixture"
