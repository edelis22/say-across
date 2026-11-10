"""Local, multi-label classification of emotion expressed in a source sentence."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock

from .config import EMOTION_MODEL, EMOTION_REVISION

MAX_TOKENS = 256
HINT_SCORE_FLOOR = 0.55
CONTEXT_GAIN = 0.15
NEUTRAL_MARGIN = 0.10
POLICY_VERSION = "1.0"

LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
)


@dataclass(frozen=True)
class EmotionReading:
    status: str
    labels: tuple[str, ...] = ()
    reason: str = ""
    mode: str = "sentence_only"
    scores: dict[str, float] = field(default_factory=dict)
    sentence_scores: dict[str, float] | None = None
    context_only_scores: dict[str, float] | None = None

    @property
    def adaptation_hints(self):
        if self.status != "detected":
            return ()
        return tuple(label for label in self.labels if self.scores[label] >= HINT_SCORE_FLOOR)


def decide(sentence_scores, thresholds, joined_scores=None, context_scores=None, *, reason=""):
    """Apply the same provisional decision policy in the UI and the evaluation."""

    scores = sentence_scores if joined_scores is None else joined_scores
    candidates = [label for label in LABELS if label != "neutral"
                  and scores[label] >= thresholds[label]]
    chosen = [label for label in candidates if context_scores is None
              or sentence_scores[label] >= thresholds[label]
              or scores[label] - context_scores[label] >= CONTEXT_GAIN]
    chosen.sort(key=scores.__getitem__, reverse=True)
    neutral = scores["neutral"] >= thresholds["neutral"]

    if chosen and neutral and scores["neutral"] >= scores[chosen[0]] - NEUTRAL_MARGIN:
        status, labels, decision_reason = "uncertain", (), "competing_neutral_score"
    elif chosen:
        status, labels, decision_reason = "detected", tuple(chosen), ""
    elif candidates:
        status, labels, decision_reason = "uncertain", (), "emotion_also_in_context_alone"
    elif neutral:
        status, labels, decision_reason = "neutral", ("neutral",), ""
    else:
        status, labels, decision_reason = "uncertain", (), "weak_scores"

    with_context = joined_scores is not None
    return EmotionReading(status, labels, reason or decision_reason,
                          "with_context" if with_context else "sentence_only",
                          scores, sentence_scores if with_context else None, context_scores)


class _OnnxEmotionModel:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        import numpy as np
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer

        def checkpoint_file(filename):
            return hf_hub_download(EMOTION_MODEL, filename, revision=EMOTION_REVISION, token=False)

        config = json.loads(Path(checkpoint_file("config.json")).read_text(encoding="utf-8"))
        labels = tuple(config["id2label"][str(index)] for index in range(len(LABELS)))
        if labels != LABELS or len(config["id2label"]) != len(LABELS):
            raise ValueError("Unexpected emotion classifier label order.")
        thresholds = json.loads(Path(checkpoint_file("thresholds.json")).read_text(encoding="utf-8"))
        if set(thresholds) != set(LABELS) or any(not 0 < value <= 1 for value in thresholds.values()):
            raise ValueError("Unexpected emotion classifier thresholds.")
        self.thresholds = thresholds
        self.tokenizer = Tokenizer.from_file(checkpoint_file("tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = SessionOptions()
        options.intra_op_num_threads = 2
        self.session = InferenceSession(checkpoint_file("onnx/model_quantized.onnx"), options,
                                        providers=["CPUExecutionProvider"])
        self.np = np

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def score(self, tokens):
        np = self.np
        inputs = {
            "input_ids": np.asarray([tokens.ids], dtype=np.int64),
            "attention_mask": np.asarray([tokens.attention_mask], dtype=np.int64),
        }
        logits = self.session.run(["logits"], inputs)[0]
        if logits.shape != (1, len(LABELS)) or not np.isfinite(logits).all():
            raise ValueError("Unexpected emotion classifier output.")
        values = 1 / (1 + np.exp(-np.clip(logits[0], -60, 60)))
        return dict(zip(LABELS, map(float, values)))


class EmotionDetector:
    """Classify text without a translation, an emotion annotation, or an API key."""

    def __init__(self):
        self._lock = Lock()
        self._model = None

    def detect(self, source, before="", after=""):
        if not source.strip():
            raise ValueError("Enter a sentence to analyze.")
        before, source, after = before.strip(), source.strip(), after.strip()
        with self._lock:
            if self._model is None:
                try:
                    self._model = _OnnxEmotionModel()
                except Exception:
                    # Optional runtime and Hub failures should leave translation usable.
                    return EmotionReading("unavailable", reason="model_load_failed")
            try:
                return self._classify(before, source, after)
            except Exception:
                return EmotionReading("unavailable", reason="model_inference_failed")

    def _classify(self, before, source, after):
        model = self._model
        source_tokens = model.encode(source)
        if len(source_tokens.ids) > MAX_TOKENS:
            return EmotionReading("unavailable", reason="sentence_over_token_limit")
        sentence_scores = model.score(source_tokens)
        if before or after:
            joined = model.encode("\n".join(filter(None, (before, source, after))))
            context = model.encode("\n".join(filter(None, (before, after))))
            if max(len(joined.ids), len(context.ids)) <= MAX_TOKENS:
                return decide(sentence_scores, model.thresholds, model.score(joined),
                              model.score(context))
            return decide(sentence_scores, model.thresholds, reason="context_over_token_limit")
        return decide(sentence_scores, model.thresholds)
