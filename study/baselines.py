"""Google Cloud Translation Basic v2, general NMT, plain text."""

from html import unescape
import os

from pragmatics.adaptation import post_json
from pragmatics.config import LANGUAGES


class GoogleNMT:
    def __init__(self, transport=post_json):
        self.transport = transport

    def translate(self, text, source, target):
        key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        if not key:
            raise RuntimeError("missing_google_api_key")
        try:
            response = self.transport(
                "https://translation.googleapis.com/language/translate/v2",
                {"X-Goog-Api-Key": key},
                {"q": text, "source": LANGUAGES[source][1], "target": LANGUAGES[target][1],
                 "format": "text", "model": "nmt"},
            )
            translation = unescape(response["data"]["translations"][0]["translatedText"]).strip()
            if not translation:
                raise ValueError("Empty Google output")
            return translation
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            raise RuntimeError("google_translation_failed") from None
