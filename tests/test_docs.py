"""Cheap structural guards for the reader-facing documents.

These exist because an editing slip once wrote one document's contents over
another's, which no test noticed and no build step would have caught.
"""

from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


class DocumentTests(TestCase):
    def test_each_document_keeps_its_own_opening_heading(self):
        for name, heading in (("README.md", "# Click"),
                              ("START_HERE.md", "# Run Click on your computer"),
                              ("PUBLISH.md", "# Publish Click for review"),
                              ("HOSTING.md", "# ")):
            self.assertIn(heading, read(name), f"{name} lost its own heading")

    def test_no_document_is_a_copy_of_another(self):
        names = ["README.md", "START_HERE.md", "PUBLISH.md", "HOSTING.md"]
        bodies = {name: read(name).strip() for name in names}
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                self.assertNotEqual(bodies[first], bodies[second],
                                    f"{first} and {second} have identical contents")

    def test_the_reference_keeps_every_section_a_reader_is_sent_to(self):
        readme = read("README.md")
        for heading in ("## Run locally", "## Architecture", "## Verify and evaluate",
                        "## Live conversation and private hosting",
                        "## Match the speaker’s voice range",
                        "## Explain at another level", "## Ask about where you are"):
            self.assertIn(heading, readme, f"README lost {heading!r}")

    def test_both_engines_are_documented_where_a_reader_looks_for_them(self):
        for name in ("README.md", "START_HERE.md"):
            body = read(name)
            self.assertIn("Ollama", body, f"{name} does not mention the free engine")
            self.assertIn("ANTHROPIC_API_KEY", body, f"{name} does not mention the paid engine")

    def test_the_product_name_is_used_throughout(self):
        for name in ("README.md", "START_HERE.md", "PUBLISH.md", "app.py",
                     "pragmatics/live.html", "docs/index.html"):
            self.assertNotIn("Say Across", read(name), f"{name} still says the old name")

    def test_no_credential_is_committed_in_a_document(self):
        for name in ("README.md", "START_HERE.md", "PUBLISH.md", "HOSTING.md"):
            body = read(name)
            self.assertNotIn("sk-ant-", body, f"{name} may contain a real key")
            self.assertNotIn("sk-proj-", body, f"{name} may contain a real key")
