"""Local sentence-level NLLB inference."""

import re
from threading import Lock

from .config import MAX_NEW_TOKENS, MAX_SOURCE_TOKENS, NLLB_MODEL, NLLB_REVISION


class SourceTokenLimit(ValueError):
    pass


def split_turn(text):
    """Split a long turn at sentence or word boundaries when available."""
    text = text.strip()
    if len(text) <= 500:
        yield text
        return

    for sentence in re.split(r"(?<=[。！？])|(?<=[.!?])\s+", text):
        remaining = sentence.strip()
        while len(remaining) > 500:
            end = max(remaining.rfind(" ", 200, 501), remaining.rfind("\n", 200, 501))
            if end < 0:
                end = 500
            yield remaining[:end].strip()
            remaining = remaining[end:].lstrip()
        if remaining:
            yield remaining


class NLLB:
    def __init__(self, device="cpu", *, turn_beams=4, fast_model=None):
        self.device = device
        self.turn_beams = turn_beams
        self.fast_model = fast_model
        self._lock = Lock()
        self._backend = None
        self._fast_backend = None

    def translate_turn(self, text, source, target):
        def translate_part(part):
            try:
                if self.fast_model is not None:
                    return self._translate_fast(part, source, target)
                if self.turn_beams == 4:
                    return self.translate(part, source, target)
                return self.translate(part, source, target, num_beams=self.turn_beams)
            except SourceTokenLimit:
                if len(part) < 2:
                    raise
                midpoint = len(part) // 2
                split = part.rfind(" ", len(part) // 4, midpoint + 1)
                split = midpoint if split <= 0 else split
                return join(translate_part(piece.strip()) for piece in (part[:split], part[split:]))

        def join(parts):
            return ("" if target in {"jpn_Jpan", "zho_Hans"} else " ").join(parts)

        return join(translate_part(part) for part in split_turn(text))

    @staticmethod
    def _load_tokenizer():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(NLLB_MODEL, revision=NLLB_REVISION,
                                             trust_remote_code=False, token=False)

    @staticmethod
    def _select_direction(tokenizer, source, target):
        for code in (source, target):
            if (not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", code)
                    or tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id):
                raise ValueError("Unknown NLLB language code.")
        tokenizer.src_lang = source

    def _translate_fast(self, text, source, target):
        with self._lock:
            if self._fast_backend is None:
                from ctranslate2 import Translator

                tokenizer = self._load_tokenizer()
                model = Translator(str(self.fast_model), device="cpu", compute_type="int8", intra_threads=4)
                self._fast_backend = tokenizer, model
            tokenizer, model = self._fast_backend
            self._select_direction(tokenizer, source, target)
            ids = tokenizer.encode(text, truncation=False)
            if len(ids) > MAX_SOURCE_TOKENS:
                raise SourceTokenLimit("Source exceeds 512 model tokens. Submit a shorter sentence.")
            tokens = tokenizer.convert_ids_to_tokens(ids)
            output = model.translate_batch(
                [tokens], target_prefix=[[target]], beam_size=self.turn_beams, max_input_length=0,
                max_decoding_length=MAX_NEW_TOKENS, return_end_token=True,
            )[0].hypotheses[0]
            if not output or output[0] != target or output[-1] != tokenizer.eos_token:
                raise RuntimeError("NLLB did not complete the requested translation.")
            result = tokenizer.decode(tokenizer.convert_tokens_to_ids(output), skip_special_tokens=True).strip()
            if not result:
                raise RuntimeError("NLLB returned an empty translation.")
            return result

    def translate(self, text, source, target, *, num_beams=4):
        # The tokenizer's source language is mutable. Serialize initialization and inference.
        with self._lock:
            import torch
            from transformers import AutoModelForSeq2SeqLM

            if self._backend is None:
                tokenizer = self._load_tokenizer()
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    NLLB_MODEL, revision=NLLB_REVISION, trust_remote_code=False, token=False,
                    use_safetensors=False, weights_only=True,
                ).to(self.device).eval()
                self._backend = tokenizer, model
            tokenizer, model = self._backend

            self._select_direction(tokenizer, source, target)
            inputs = tokenizer(text, return_tensors="pt", truncation=False)
            if inputs.input_ids.shape[-1] > MAX_SOURCE_TOKENS:
                raise SourceTokenLimit("Source exceeds 512 model tokens. Submit a shorter sentence.")
            with torch.inference_mode():
                output = model.generate(
                    **inputs.to(self.device),
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
                    forced_eos_token_id=None,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=num_beams,
                    do_sample=False,
                )[0]
            if output[-1].item() != tokenizer.eos_token_id:
                raise RuntimeError("NLLB hit the generation limit before completing the sentence.")
            result = tokenizer.decode(output, skip_special_tokens=True).strip()
            if not result:
                raise RuntimeError("NLLB returned an empty translation.")
            return result
