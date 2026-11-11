"""Free, key-free grounding for the lookup feature.

Wikipedia's API needs no account, no key and no quota, so the lookup works on a
machine that never talks to a paid service. It is narrower than a web search:
it is good for how a place, an institution or a procedure generally works, and
poor for prices, opening times and this week's news. Callers say so to the
reader rather than presenting an extract as current fact.
"""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import LANGUAGES

# A generic client name. No user, project or machine detail is ever sent.
USER_AGENT = "ClickResearchPrototype/1.0 (noncommercial research; local use)"
MAX_EXTRACT_CHARS = 1200
RESULT_LIMIT = 4
TIMEOUT_SECONDS = 12


class ResearchError(RuntimeError):
    pass


def _wikipedia_host(language):
    """Map a configured language to its Wikipedia subdomain."""
    code = LANGUAGES[language][1] if language in LANGUAGES else "en"
    return f"{code}.wikipedia.org"


def _request(host, parameters):
    url = f"https://{host}/w/api.php?{urlencode(parameters)}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        raise ResearchError("research_unreachable") from None


def search(query, language="eng_Latn", *, limit=RESULT_LIMIT):
    """Return encyclopedia passages for a question, newest-known-good first.

    Searching in the reader's own language gives better passages when that
    Wikipedia covers the topic, so English is only a fallback.
    """
    query = (query or "").strip()
    if not query:
        raise ResearchError("empty_question")
    hosts = [_wikipedia_host(language)]
    if hosts[0] != "en.wikipedia.org":
        hosts.append("en.wikipedia.org")
    for host in hosts:
        try:
            payload = _request(host, {
                "action": "query", "format": "json", "formatversion": "2",
                "generator": "search", "gsrsearch": query[:300], "gsrlimit": str(limit),
                "gsrnamespace": "0", "prop": "extracts", "exintro": "1", "explaintext": "1",
                "exlimit": str(limit), "redirects": "1",
            })
        except ResearchError:
            continue
        passages = _passages(payload, host)
        if passages:
            return passages
    return []


def _passages(payload, host):
    query = payload.get("query") if isinstance(payload, dict) else None
    pages = query.get("pages") if isinstance(query, dict) else None
    # formatversion=2 returns a list; formatversion=1 returns a dict keyed by page
    # id. Accept either, so a server-side default cannot empty the results.
    if isinstance(pages, dict):
        pages = list(pages.values())
    elif not isinstance(pages, list):
        pages = []
    passages = []
    for page in pages:
        if not isinstance(page, dict) or page.get("missing"):
            continue
        title = str(page.get("title", "")).strip()
        extract = " ".join(str(page.get("extract", "")).split())
        if not title or len(extract) < 80:
            continue
        # The generator reports search rank in "index"; keep it for ordering only.
        rank = page.get("index")
        passages.append((rank if isinstance(rank, int) else len(passages), {
            "title": title,
            "url": f"https://{host}/wiki/{title.replace(' ', '_')}",
            "extract": extract[:MAX_EXTRACT_CHARS],
            "wiki": host.split(".", 1)[0],
        }))
    return [passage for _, passage in sorted(passages, key=lambda item: item[0])][:RESULT_LIMIT]
