"""Frozen research defaults. Credentials are read only when a remote call is made."""

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
LLM_MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "2.0"
EMOTION_MODEL = "Horizon-Labs/multilingual-emotions-small"
EMOTION_REVISION = "a95c3a7bba5ab9cc01d8666aa690c51cc3d42b16"
MAX_SOURCE_TOKENS = 512
MAX_NEW_TOKENS = 512
LANGUAGES = {
    "eng_Latn": ("English", "en", ("en-US", "en-GB")),
    "spa_Latn": ("Spanish", "es", ("es-MX", "es-ES")),
    "ell_Grek": ("Greek", "el", ("el-GR", "el-CY")),
    "jpn_Jpan": ("Japanese", "ja", ("ja-JP",)),
    "fra_Latn": ("French", "fr", ("fr-FR", "fr-CA")),
    "deu_Latn": ("German", "de", ("de-DE",)),
    "ita_Latn": ("Italian", "it", ("it-IT",)),
    "por_Latn": ("Portuguese", "pt", ("pt-BR", "pt-PT")),
    "nld_Latn": ("Dutch", "nl", ("nl-NL",)),
    "zho_Hans": ("Mandarin Chinese (Simplified)", "zh", ("zh-CN",)),
    "kor_Hang": ("Korean", "ko", ("ko-KR",)),
    "arb_Arab": ("Arabic (Standard)", "ar", ("ar-001",)),
}
RELATIONSHIPS = ("unspecified", "friend", "family", "peer", "supervisor", "customer", "stranger")
TONES = ("preserve", "neutral", "warm", "formal", "informal", "firm")
