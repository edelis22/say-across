"""Explicit delivery intent and a small local catalogue for live interpretation."""

from dataclasses import dataclass, replace
import json
import re
import unicodedata

from .config import LANGUAGES, RELATIONSHIPS

VOICE_TONES = {
    "auto": ("Match delivery", None),
    "warm": ("Warm", (158, 1, 50, 0.44, 0.9)),
    "calm": ("Calm", (143, -2, 32, 0.16, 0.55)),
    "excited": ("Excited", (194, 4, 67, 0.69, 1.0)),
    "firm": ("Firm", (168, -2, 41, 0.48, 0.8)),
    "formal": ("Formal", (153, 0, 37, 0.27, 0.65)),
}

# Complete turns only. Each entry supplies a literal-language paraphrase for NLLB.
IDIOMS = {
    ("eng_Latn", "break a leg"): ("Good luck!", "performance, audition, theater, theatre, rehearsal"),
    ("eng_Latn", "it's a piece of cake"): ("It's very easy.", "task, exam, homework, project, easy"),
    ("eng_Latn", "it's raining cats and dogs"): ("It's raining very heavily.", ""),
    ("eng_Latn", "i'm pulling your leg"): ("I'm joking with you.", ""),
    ("eng_Latn", "let's call it a day"): ("Let's stop for today.", ""),
    ("eng_Latn", "we're on the same page"): ("We have the same understanding.", "plan, project, agree, agreement, meeting"),
    ("spa_Latn", "no me tomes el pelo"): ("No intentes engañarme.", ""),
    ("spa_Latn", "no me tome el pelo"): ("No intente engañarme.", ""),
    ("spa_Latn", "está chupado"): ("Es muy fácil.", "tarea, examen, trabajo, fácil, proyecto"),
    ("ell_Grek", "έβρεχε καρεκλοπόδαρα"): ("Έβρεχε πάρα πολύ.", ""),
    ("jpn_Jpan", "猫の手も借りたい"): ("とても忙しくて、誰かに手伝ってほしいです。", ""),
    ("fra_Latn", "il pleut des cordes"): ("Il pleut très fort.", ""),
    ("deu_Latn", "das ist nicht mein bier"): ("Das geht mich nichts an.", ""),
    ("ita_Latn", "in bocca al lupo"): ("Ti auguro buona fortuna!", ""),
    ("por_Latn", "estou com a pulga atrás da orelha"): ("Tenho minhas dúvidas.", ""),
    ("nld_Latn", "nu komt de aap uit de mouw"): ("Nu wordt de verborgen bedoeling duidelijk.", ""),
    ("zho_Hans", "这对我来说是小菜一碟"): ("这对我来说非常容易。", ""),
    ("kor_Hang", "식은 죽 먹기예요"): ("아주 쉬워요.", ""),
    ("arb_Arab", "الصبر مفتاح الفرج"): ("بالصبر نتجاوز الشدائد.", ""),
}

# Wording changes are restricted to exact phrases and an explicit relationship.
THANKS = {"eng_Latn": {"thank you for helping me", "thank you for your help"},
          "spa_Latn": {"gracias por ayudarme", "gracias por tu ayuda", "gracias por su ayuda"}}
REGISTER = {
    ("spa_Latn", "close"): "Gracias por ayudarme.",
    ("spa_Latn", "professional"): "Gracias por su ayuda.",
    ("ell_Grek", "close"): "Ευχαριστώ που με βοήθησες.",
    ("ell_Grek", "professional"): "Σας ευχαριστώ για τη βοήθειά σας.",
    ("jpn_Jpan", "close"): "手伝ってくれてありがとう。",
    ("jpn_Jpan", "professional"): "手伝ってくださってありがとうございます。",
    ("fra_Latn", "close"): "Merci pour ton aide.",
    ("fra_Latn", "professional"): "Merci pour votre aide.",
    ("deu_Latn", "close"): "Danke für deine Hilfe.",
    ("deu_Latn", "professional"): "Vielen Dank für Ihre Hilfe.",
}


@dataclass(frozen=True)
class InterpreterSettings:
    two_way: bool = False
    tone: str = "auto"
    relationship: str = "unspecified"
    situation: str = ""
    remote: bool = False
    profile_language: str = ""

    @classmethod
    def parse(cls, payload):
        if not payload:
            return cls()
        if len(payload) > 4096:
            raise ValueError("Conversation settings are too long.")
        try:
            settings = cls(**json.loads(payload))
        except (TypeError, ValueError):
            raise ValueError("Invalid conversation settings.") from None
        if (type(settings.two_way) is not bool or type(settings.remote) is not bool
                or type(settings.situation) is not str or len(settings.situation) > 300
                or type(settings.tone) is not str or settings.tone not in VOICE_TONES
                or type(settings.profile_language) is not str
                or (settings.profile_language and settings.profile_language not in LANGUAGES)
                or settings.relationship not in RELATIONSHIPS):
            raise ValueError("Invalid conversation settings.")
        return settings


def _normalize(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return " ".join(normalized.strip(" .?!¿¡؟。！？，،…\t\n").split())


def reviewed_spoken_translation(text, source, target):
    """Protect exact complete turns with documented errors in the live model."""
    if target != "eng_Latn" or text.rstrip().endswith(("?", ";", "\u037e")):
        return None
    words = _normalize(text)
    if source == "ell_Grek":
        # Include the reported ASR spelling and the spellings reproduced with Whisper Base.
        if words in {"ευχαριστώ πάρα πολύ", "ευχαριστώ πάρω πολύ", "ευχαριστώ παραπολή",
                     "ευχαριστώ, παραπολή"}:
            return "Thank you very much."
        if words == "ευχαριστώ":
            return "Thank you."
    if source == "spa_Latn" and words == "muchas gracias":
        return "Thank you very much."
    return None


def local_interpretation(text, source, target, settings):
    """Return an optional target sentence or a source paraphrase with a visible note."""
    words = _normalize(text)
    if words in THANKS.get(source, ()):
        group = {"friend": "close", "family": "close", "supervisor": "professional",
                 "customer": "professional"}.get(settings.relationship)
        candidate = REGISTER.get((target, group))
        if candidate:
            return candidate, text, "Wording adjusted for the selected relationship."
    context = _normalize(settings.situation)
    if source == "eng_Latn" and words == "let's table the motion":
        meanings = [plain for place, plain in (
            ("us senate", "Let's set the motion aside."),
            ("uk parliament", "Let's formally put forward the motion."),
        ) if re.search(rf"\b{place}\b", context)]
        if len(meanings) == 1:
            return None, meanings[0], "Interpreted the wording using the supplied meeting context."
    entry = IDIOMS.get((source, words))
    if entry:
        plain, triggers = entry
        if not triggers or any(re.search(rf"\b{re.escape(term)}\b", context) for term in triggers.split(", ")):
            return None, plain, "Interpreted an idiom from the original speech."
    return None, text, ""


def intended_delivery(measured, tone):
    preset = VOICE_TONES[tone][1]
    if preset is None:
        return measured
    rate, shift, pitch_range, expressiveness, emphasis = preset
    return replace(measured, rate=rate, pitch_offset=shift, pitch_range=pitch_range,
                   exaggeration=expressiveness, emphasis_db=tuple(value * emphasis for value in measured.emphasis_db))
