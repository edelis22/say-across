"""Voice affect from pitch, relative energy, and timing features at 16 kHz."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .config import HIGHER_PITCH_FLOOR_HZ, LOWER_PITCH_CEILING_HZ
from .emotion import EmotionReading

LABELS = ("neutral", "calm", "joy", "sadness", "anger", "fear", "disgust", "surprise")
MODEL_PATH = Path(__file__).with_name("prosody_model.json")
FEATURE_VERSION = 1
MARGIN = 0.10
LABEL_MARGIN = 0.15
FRAME_SECONDS = 0.02
SAMPLE_RATE = 16000
MIN_VOICED_FRAMES_FOR_RANGE = 15


@dataclass(frozen=True)
class _Track:
    onset: int
    rms: np.ndarray
    active: np.ndarray
    lags: np.ndarray
    voiced: np.ndarray


def _acoustic_track(audio):
    # No transcript, phoneme, speaker embedding, or spectral features enter the classifier.
    if len(audio) < 8000:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio, 640)[::320]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.max() < 0.002:
        return None
    active = rms >= max(rms.max() * 0.06, 0.001)
    bounds = np.flatnonzero(active)
    if len(bounds) < 14:
        return None
    span = slice(bounds[0], bounds[-1] + 1)
    rms, active = rms[span], active[span]
    centered = frames[span] - frames[span].mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, n=2048, axis=1)) ** 2
    ac = np.fft.irfft(power, axis=1)
    ac /= np.maximum(ac[:, :1], 1e-12)
    band = ac[:, 40:269]
    peaks = (band[:, 1:-1] > band[:, :-2]) & (band[:, 1:-1] >= band[:, 2:])
    peak_strength = np.where(peaks, band[:, 1:-1], 0)
    best = peak_strength.max(axis=1)
    # Earliest nearly strongest peak avoids choosing a multiple of the pitch period.
    lags = (peak_strength >= 0.90 * best[:, None]).argmax(axis=1) + 41
    voiced = active & (best >= 0.52)
    if voiced.sum() < 8:
        return None
    return _Track(int(bounds[0]), rms, active, lags, voiced)


def _summarize(track):
    if track is None:
        return None
    rms, active, lags, voiced = track.rms, track.active, track.lags, track.voiced
    envelope = 20 * np.log10(np.maximum(rms / rms.max(), 0.001))
    level = envelope[active]
    pitch = 12 * np.log2(16000 / lags[voiced])
    # Semitone differences are relative to each clip; absolute pitch is excluded.
    pitch -= np.median(pitch)
    successive = voiced[1:] & voiced[:-1]
    steps = np.abs(12 * np.log2(lags[1:][successive] / lags[:-1][successive]))
    pauses = int(np.count_nonzero(active[:-1] & ~active[1:]))
    smooth = np.convolve(envelope, np.ones(5) / 5, mode="same")
    energy_peaks = np.count_nonzero(
        (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]) & (smooth[1:-1] > -20)
    )
    seconds = len(rms) * FRAME_SECONDS
    return np.array([
        np.percentile(pitch, 90) - np.percentile(pitch, 10), np.std(pitch),
        np.percentile(pitch, 75) - np.percentile(pitch, 25),
        np.median(steps) if len(steps) else 0,
        np.percentile(steps, 85) if len(steps) else 0,
        np.median(pitch[-len(pitch) // 3:]) - np.median(pitch[:max(1, len(pitch) // 3)]),
        np.median(level), np.std(level), np.percentile(level, 25),
        np.mean(np.abs(np.diff(envelope))),
        active.mean(), voiced.mean(), pauses / seconds, energy_peaks / seconds,
    ])


def features(audio):
    return _summarize(_acoustic_track(audio))


def _median_pitch(track):
    """Median voiced f0 in Hz.

    This is the one place absolute pitch is read. It never enters _summarize,
    so the emotion classifier still sees only clip-relative pitch and the
    published voice model stays unchanged. It is used to choose an output
    voice whose range is closer to the speaker's.
    """
    if track is None or track.voiced.sum() < MIN_VOICED_FRAMES_FOR_RANGE:
        return None
    return float(np.median(SAMPLE_RATE / track.lags[track.voiced]))


def pitch_voice_range(pitch_hz):
    """Place a measured pitch in the lower or higher band, or abstain.

    The gap between the two thresholds is the range where adult speakers
    overlap. Measurements that land inside it return "" so the configured
    default voice for the locale is kept rather than guessed at. This reports
    a pitch band only and makes no claim about a speaker's gender or identity.
    """
    if pitch_hz is None:
        return ""
    if pitch_hz <= LOWER_PITCH_CEILING_HZ:
        return "lower"
    if pitch_hz >= HIGHER_PITCH_FLOOR_HZ:
        return "higher"
    return ""


def voice_pitch(audio):
    """Measure the speaker's pitch band from a recording."""
    pitch = _median_pitch(_acoustic_track(audio))
    return pitch, pitch_voice_range(pitch)


def _delivery_changes(track):
    """Report the strongest local changes using adjacent voiced windows in this turn."""
    if track is None:
        return []
    smooth = np.convolve(track.rms, np.ones(3) / 3, mode="same")
    peaks = np.r_[False, (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]), False]
    axes = (("pitch_up", "pitch_down", 3.0), ("louder", "softer", 6.0), ("faster", "slower", 2.5))
    events = []
    previous = None
    for start in range(0, len(track.rms), 40):
        window = slice(start, start + 40)
        voiced = track.voiced[window]
        if len(voiced) < 25 or voiced.sum() < 10:
            previous = None
            continue
        rms = track.rms[window][voiced]
        pitch = np.median(12 * np.log2(16000 / track.lags[window][voiced]))
        level = 20 * np.log10(np.median(rms))
        energy = smooth[window]
        pulses = np.count_nonzero(peaks[window] & (energy >= np.median(energy) * 1.15))
        rhythm = pulses / (len(voiced) * FRAME_SECONDS) if pulses >= 2 else np.nan
        current = np.array([pitch, level, rhythm])
        if previous is not None:
            second = round((track.onset + start) * FRAME_SECONDS, 1)
            for change, (up, down, threshold) in zip(current - previous, axes):
                if abs(change) >= threshold:
                    events.append((abs(change) / threshold, second, up if change > 0 else down))
        previous = current
    strongest = sorted(events, reverse=True)[:8]
    return sorted((second, cue) for _, second, cue in strongest)


@lru_cache(maxsize=1)
def load_model():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model["features_version"] != FEATURE_VERSION or tuple(model["labels"]) != LABELS:
        raise ValueError("Unexpected voice model version.")
    return model


def normalize_features(rows, model):
    return np.clip((rows - model["center"]) / model["scale"], -5, 5)


def probabilities(rows, model):
    logits = normalize_features(rows, model) @ np.asarray(model["weights"]) + model["bias"]
    scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def detect_voice(audio, enabled=True):
    return _classify(features(audio) if enabled else None, enabled)


def choose_label(scores, thresholds, margin):
    second, first = scores.argsort()[-2:]
    label = LABELS[first]
    floor = thresholds.get(label)
    if floor is not None and scores[first] >= floor and scores[first] - scores[second] >= margin:
        return label
    return None


def _classify(row, enabled):
    mode = "voice_prosody"
    if not enabled:
        return EmotionReading("unavailable", reason="disabled", mode=mode)
    if row is None:
        return EmotionReading("uncertain", reason="insufficient_voiced_audio", mode=mode)
    try:
        model = load_model()
        scores = probabilities(row, model)
    except (OSError, KeyError, ValueError):
        return EmotionReading("unavailable", reason="voice_model_unavailable", mode=mode)
    raw = dict(zip(LABELS, map(float, scores)))
    label = choose_label(scores, model.get("label_thresholds", {}), LABEL_MARGIN)
    if label is None:
        return EmotionReading("uncertain", reason="no_supported_voice_label", mode=mode, scores=raw)
    return EmotionReading("neutral" if label == "neutral" else "detected", (label,), mode=mode, scores=raw)


@dataclass(frozen=True)
class Delivery:
    rate: int
    pitch_offset: int
    pitch_range: int
    exaggeration: float
    emphasis_db: tuple[float, ...] = ()
    voice_range: str = ""
    pitch_hz: float | None = None


def delivery(audio, *, source_language="eng_Latn"):
    """Map measured vocal variation to bounded controls for the output voice."""
    track = _acoustic_track(audio)
    vector = _summarize(track)
    return _delivery(vector, _phrase_emphasis(track), _source_intensity(vector, source_language),
                     _median_pitch(track))


def intensity_level(vector, model):
    settings = model.get("intensity")
    if vector is None or settings is None:
        return None
    score = normalize_features(vector, model) @ np.asarray(settings["weights"])
    return float(np.tanh((score - settings["center"]) / settings["scale"]))


def _source_intensity(vector, source_language):
    if vector is None or source_language != "eng_Latn":
        return None
    try:
        return intensity_level(vector, load_model())
    except (OSError, KeyError, ValueError):
        return None


def _delivery(vector, emphasis=(), intensity=None, pitch_hz=None):
    span, drift, pace = (vector[0], vector[5], vector[13]) if vector is not None else (5, 0, 4)
    pitch_activity = float(np.clip((span - 5) / 7, -1, 1))
    rhythm = float(np.clip((pace - 4) / 3, -1, 1))
    rate = round(168 + 19 * rhythm + 8 * pitch_activity)
    shift = round(np.clip(0.55 * drift + 1.2 * pitch_activity, -6, 6))
    pitch_range = round(45 + 17 * pitch_activity)
    expressiveness = 0.38 + 0.21 * pitch_activity + 0.11 * rhythm
    if intensity is not None:
        expressiveness = 0.40 * expressiveness + 0.60 * (0.39 + 0.27 * intensity)
        emphasis = tuple(round(value * (0.85 + 0.15 * intensity), 2) for value in emphasis)
    exaggeration = round(float(np.clip(expressiveness, 0.12, 0.70)), 2)
    return Delivery(rate, shift, pitch_range, exaggeration, emphasis,
                    pitch_voice_range(pitch_hz), pitch_hz)


def _phrase_emphasis(track):
    if track is None or len(track.rms) < 40:
        return ()
    sections = min(8, len(track.rms) // 20)
    frames = np.array_split(np.arange(len(track.rms)), sections)
    voiced = [track.rms[block[track.voiced[block]]] for block in frames]
    levels = np.array([20 * np.log10(np.median(block)) if len(block) >= 4 else np.nan for block in voiced])
    usable = np.flatnonzero(np.isfinite(levels))
    if len(usable) < 2:
        return ()
    relative = levels[usable] - np.median(levels[usable])
    curve = np.interp(np.arange(sections), usable, relative)
    return tuple(round(float(value), 2) for value in np.clip(curve * 0.45, -3, 3))


def analyze_turn(audio, *, enabled=True, source_language="eng_Latn"):
    """Measure local delivery changes and playback controls from the same waveform."""
    track = _acoustic_track(audio)
    vector = _summarize(track)
    reading = _classify(vector, enabled and source_language == "eng_Latn")
    style = _delivery(vector, _phrase_emphasis(track), _source_intensity(vector, source_language),
                      _median_pitch(track))
    return reading, style, _delivery_changes(track) if enabled else []
