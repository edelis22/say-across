"""Conservative surface checks. Human review measures semantic fidelity."""

from collections import Counter
import re
import unicodedata

from .adaptation import FIELDS


def validate_candidate(candidate, source):
    if not isinstance(candidate, dict) or set(candidate) != set(FIELDS):
        return "invalid_schema"
    for field, kind in FIELDS.items():
        value = candidate[field]
        if kind == "string" and (type(value) is not str or len(value) > 6000):
            return "invalid_schema"
        if kind == "boolean" and type(value) is not bool:
            return "invalid_schema"
    probability = candidate["adequacy_probability"]
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        return "invalid_probability"
    if candidate["abstain"]:
        return "model_abstained"
    translation = candidate["translation"].strip()
    if not translation or not candidate["explanation"].strip():
        return "empty_candidate"
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in translation):
        return "control_character"
    if numbers(translation) != numbers(source):
        return "numeric_surface_mismatch"
    if urls(translation) != urls(source):
        return "url_surface_mismatch"
    return None


def numbers(text):
    normalized = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                         for c in unicodedata.normalize("NFKC", text))
    return Counter(re.findall(r"[0-9]+", normalized))


def urls(text):
    return Counter(url.rstrip(".,!?;:。！？") for url in re.findall(r'https?://[^\s<>"\u3000]+', text))
