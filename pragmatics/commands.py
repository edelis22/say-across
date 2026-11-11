"""Spoken commands addressed to Click itself.

A live turn is normally translated. A turn that opens by naming the app is
treated as a request to Click instead: look something up, or restate the last
turn in simpler words. Recognition is deliberately narrow, because a false
positive silently swallows a sentence the speaker wanted translated.
"""

import re
import unicodedata

# What recognizers actually produce for "Click", including common mishearings.
WAKE_WORDS = ("click", "clicks", "clique", "cliq", "clic", "klick", "klik", "quick click")
# An address only counts at the very start of a turn, optionally after a greeting.
GREETINGS = ("hey", "hi", "hello", "ok", "okay", "yo", "a", "um", "uh")

LOOKUP_HINTS = (
    "look up", "lookup", "look it up", "search", "find out", "find me", "tell me about",
    "what is", "what's", "whats", "what are", "who is", "who's", "where is", "where can",
    "how do", "how does", "how can", "when is", "why is", "why does", "which", "google",
    "current policy", "policy", "news", "rules", "regulation", "requirement",
)
SIMPLIFY_HINTS = (
    "simpler", "simply", "simplify", "plain", "plainer", "easier", "easy words",
    "dumb it down", "break it down", "explain that", "explain this", "explain it",
    "what does that mean", "what did that mean", "what does this mean", "i don't understand",
    "i do not understand", "say that again", "in other words", "clarify",
)
# Said alone, these are an instruction rather than a question to research.
BARE_SIMPLIFY = ("explain", "again", "repeat", "what", "huh")

MAX_QUERY_CHARS = 400


def _normalize(text):
    folded = unicodedata.normalize("NFKC", text or "").casefold().replace("’", "'")
    # Keep apostrophes so "what's" survives; drop the rest of the punctuation.
    return " ".join(re.sub(r"[^\w\s']+", " ", folded).split())


def _strip_address(words):
    """Remove a leading greeting and the app's name, or return None.

    Returns the words that follow the address, so "hey click look up X" yields
    "look up X". The name has to appear in the first three words: a mention
    later in a sentence is almost always someone talking about clicking.
    """
    for start in range(min(3, len(words))):
        if start and words[start - 1] not in GREETINGS:
            break
        # "quick click" and the like arrive as two tokens.
        for span in (2, 1):
            if words[start:start + span] and " ".join(words[start:start + span]) in WAKE_WORDS:
                return words[start + span:]
    return None


def parse_command(transcript):
    """Classify a turn addressed to Click.

    Returns (intent, query) with intent "lookup" or "simplify", or None when the
    turn is ordinary speech to be translated as usual.
    """
    words = _normalize(transcript).split()
    remainder = _strip_address(words)
    if remainder is None:
        return None
    # "please" and a comma-ish filler carry no intent.
    while remainder and remainder[0] in ("please", "can", "could", "you", "would", "will", "and"):
        remainder = remainder[1:]
    rest = " ".join(remainder)
    if not rest:
        # The name alone is an address with nothing asked.
        return None
    if any(hint in rest for hint in SIMPLIFY_HINTS) or rest in BARE_SIMPLIFY:
        return "simplify", ""
    if any(hint in rest for hint in LOOKUP_HINTS):
        return "lookup", _query(rest)
    # Anything else addressed to Click is taken as something to look up.
    return "lookup", _query(rest)


def _query(rest):
    """Trim the command wording so the search sees the question itself."""
    for opener in ("look up", "lookup", "look it up", "search for", "search", "find out about",
                   "find out", "find me", "tell me about", "tell me"):
        if rest.startswith(opener):
            rest = rest[len(opener):].strip()
            break
    for filler in ("what the", "what is the", "what's the"):
        if rest.startswith(filler):
            rest = rest[len(filler):].strip()
            break
    return rest[:MAX_QUERY_CHARS].strip()
