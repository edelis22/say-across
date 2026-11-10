"""Translate locally, optionally adapt, and expose fallbacks with machine-readable reasons."""

from dataclasses import asdict, dataclass
from time import perf_counter

from .adaptation import RemoteError
from .config import LANGUAGES, RELATIONSHIPS, TONES
from .emotion import EmotionReading
from .validation import validate_candidate


def validate_text(source, before, after, *, source_limit=2500):
    for value, limit in ((source, source_limit), (before, 1000), (after, 1000)):
        if type(value) is not str or len(value) > limit:
            raise ValueError(f"A text field exceeded its {limit}-character limit.")
    if not source.strip():
        raise ValueError("Enter a sentence.")


@dataclass(frozen=True)
class Sentence:
    source: str
    source_language: str
    target_language: str
    before: str = ""
    after: str = ""
    relationship: str = "unspecified"
    locale: str = ""
    tone: str = "preserve"
    intended_emotion: str = ""

    def validate(self, *, source_limit=2500):
        validate_text(self.source, self.before, self.after, source_limit=source_limit)
        if type(self.intended_emotion) is not str or len(self.intended_emotion) > 100:
            raise ValueError("Intended emotion must be at most 100 characters.")
        if self.source_language not in LANGUAGES or self.target_language not in LANGUAGES:
            raise ValueError("Select a configured language.")
        if self.source_language == self.target_language:
            raise ValueError("Select two different languages.")
        if self.relationship not in RELATIONSHIPS or self.tone not in TONES:
            raise ValueError("Select a supported relationship and tone.")
        if self.locale and self.locale not in LANGUAGES[self.target_language][2]:
            raise ValueError("Select a locale for the target language.")


@dataclass
class Result:
    baseline: str
    output: str
    status: str = "baseline_only"
    reason: str = ""
    explanation: str = ""
    intended_emotion: str = ""
    emotion: EmotionReading | None = None
    emotion_hints: tuple[str, ...] = ()
    ambiguity: str = ""
    adequacy_probability: float | None = None
    candidate: str | None = None
    response_model: str | None = None
    nllb_ms: float = 0
    adaptation_ms: float = 0
    emotion_ms: float = 0


def run(sentence, translator, adapter=None, *, guarded=True, baseline=None,
        emotion_detector=None, emotion_reading=None, spoken_turn=False):
    sentence.validate(source_limit=12000 if spoken_turn else 2500)
    start = perf_counter()
    if baseline is None:
        translate = translator.translate_turn if spoken_turn else translator.translate
        baseline = translate(sentence.source.strip(), sentence.source_language, sentence.target_language)
        nllb_ms = (perf_counter() - start) * 1000
    else:
        nllb_ms = 0  # Shared baseline measured separately by the study runner.
    result = Result(baseline, baseline, intended_emotion=sentence.intended_emotion.strip(),
                    nllb_ms=nllb_ms, emotion=emotion_reading)
    if emotion_detector is not None and result.emotion is None:
        start = perf_counter()
        result.emotion = emotion_detector.detect(sentence.source, sentence.before, sentence.after)
        result.emotion_ms = (perf_counter() - start) * 1000
    if adapter is None:
        return result

    start = perf_counter()
    try:
        if result.emotion is not None and not result.intended_emotion:
            result.emotion_hints = result.emotion.adaptation_hints
        candidate, result.response_model = adapter.adapt({
            **asdict(sentence), "baseline": baseline, "detected_emotions": list(result.emotion_hints),
        })
        reason = validate_candidate(candidate, sentence.source)
        if isinstance(candidate, dict) and isinstance(candidate.get("translation"), str):
            result.candidate = candidate["translation"]
        if reason and (guarded or reason not in {
            "numeric_surface_mismatch", "url_surface_mismatch"
        }):
            result.status, result.reason = "fallback", reason
        else:
            result.status, result.reason = "adapted", reason or ""
            result.output = candidate["translation"].strip()
            result.explanation = candidate["explanation"].strip()
            result.ambiguity = candidate["ambiguity"].strip()
            result.adequacy_probability = candidate["adequacy_probability"]
    except RemoteError as error:
        result.status, result.reason = "fallback", str(error)
    finally:
        result.adaptation_ms = (perf_counter() - start) * 1000
    return result
