"""Short, localized descriptions of measured vocal changes."""

from .config import LANGUAGES

_KINDS = ("pitch_up", "pitch_down", "louder", "softer", "faster", "slower")
_LABELS = {
    "en": ("Pitch rises", "Pitch falls", "Voice gets louder", "Voice gets softer", "Rhythm quickens", "Rhythm slows"),
    "es": ("Sube el tono", "Baja el tono", "Sube el volumen de la voz", "Baja el volumen de la voz",
           "El ritmo se acelera", "El ritmo se ralentiza"),
    "el": ("Ο τόνος ανεβαίνει", "Ο τόνος κατεβαίνει", "Η φωνή δυναμώνει", "Η φωνή χαμηλώνει",
           "Ο ρυθμός επιταχύνεται", "Ο ρυθμός επιβραδύνεται"),
    "ja": ("声の高さが上がる", "声の高さが下がる", "声が大きくなる", "声が小さくなる", "話すリズムが速くなる", "話すリズムが遅くなる"),
    "fr": ("La voix devient plus aiguë", "La voix devient plus grave", "La voix se fait plus forte",
           "La voix se fait plus douce", "Le rythme s’accélère", "Le rythme ralentit"),
    "de": ("Die Tonhöhe steigt", "Die Tonhöhe sinkt", "Die Stimme wird lauter", "Die Stimme wird leiser",
           "Der Sprechrhythmus wird schneller", "Der Sprechrhythmus wird langsamer"),
    "it": ("La voce diventa più acuta", "La voce diventa più grave", "Il volume della voce aumenta",
           "Il volume della voce diminuisce", "Il ritmo accelera", "Il ritmo rallenta"),
    "pt": ("O tom sobe", "O tom desce", "O volume da voz aumenta", "O volume da voz diminui",
           "O ritmo acelera", "O ritmo desacelera"),
    "nl": ("De toonhoogte stijgt", "De toonhoogte daalt", "De stem wordt luider", "De stem wordt zachter",
           "Het spreekritme versnelt", "Het spreekritme vertraagt"),
    "zh": ("音调升高", "音调降低", "声音变大", "声音变轻", "说话节奏加快", "说话节奏放慢"),
    "ko": ("음높이가 올라감", "음높이가 내려감", "목소리가 커짐", "목소리가 작아짐", "말하는 리듬이 빨라짐", "말하는 리듬이 느려짐"),
    "ar": ("ترتفع نبرة الصوت", "تنخفض نبرة الصوت", "يزداد علو الصوت", "يخفت الصوت",
           "تتسارع وتيرة الكلام", "تتباطأ وتيرة الكلام"),
}


def caption_cues(changes, target):
    labels = dict(zip(_KINDS, _LABELS[LANGUAGES[target][1]], strict=True))
    return [{"second": second, "kind": kind, "text": labels[kind]} for second, kind in changes]
