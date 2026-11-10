"""Keep one optional Apple Silicon voice engine warm across live turns."""

import base64
from pathlib import Path

from .config import LANGUAGES
from .speech import SAMPLE_RATE, SpeechError, SpeechWorker, decode_live_phrase

VOICE_PROFILE_SECONDS = 9
MIN_PROFILE_SPEECH_SECONDS = 3


def _encode_reference(signal):
    import numpy as np

    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _decode_profile(encoded):
    import numpy as np

    signal = decode_live_phrase(encoded, max_seconds=VOICE_PROFILE_SECONDS)
    frame = SAMPLE_RATE // 50
    usable = len(signal) // frame * frame
    rms = np.sqrt(np.mean(signal[:usable].reshape(-1, frame) ** 2, axis=1))
    active = np.flatnonzero(rms >= max(0.004, rms.max(initial=0) * 0.08))
    if len(active) * frame < MIN_PROFILE_SPEECH_SECONDS * SAMPLE_RATE:
        raise SpeechError("Record at least three seconds of clear speech for your profile.")
    start = max(0, (active[0] - 5) * frame)
    end = min(len(signal), (active[-1] + 6) * frame)
    return signal[start:end]


class VoiceMatch(SpeechWorker):
    def __init__(self, executable):
        super().__init__([executable, "-u", str(Path(__file__).with_name("clone_worker.py"))],
                         "The matching voice engine")

    def warmup(self):
        self.request({"warm": True})

    @staticmethod
    def _synthesis_request(text, target, signal, style):
        if len(signal) < SAMPLE_RATE:
            raise SpeechError("Voice matching needs at least a second of reference speech.")
        if len(text) > 500:
            raise SpeechError("This turn is too long for voice matching.")
        return {
            "text": text, "language": LANGUAGES[target][1], "exaggeration": style.exaggeration,
            "reference": _encode_reference(signal[:SAMPLE_RATE * 10]),
        }

    def synthesize(self, text, target, signal, style):
        return self.audio(self._synthesis_request(text, target, signal, style))

    def synthesize_profiled(self, text, target, signal, style, profile):
        request = self._synthesis_request(text, target, signal, style)
        request["profile"] = _encode_reference(_decode_profile(profile))
        reply = self.request(request)
        return self.decode_audio(reply), reply.get("profile_used", False)
