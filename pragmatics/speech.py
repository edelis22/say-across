"""Local speech transcription and voice playback."""

import atexit
import base64
import binascii
from contextlib import suppress
from io import BytesIO
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import wave

from .config import LANGUAGES

ASR_MODEL = "openai/whisper-base"
ASR_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
MLX_ASR_MODEL = "mlx-community/whisper-base-mlx"
MLX_ASR_REVISION = "1e3e249fb8d01c655324bd6841b1deadffd6d04c"
MLX_TURBO_MODEL = "mlx-community/whisper-large-v3-turbo-4bit"
MLX_TURBO_REVISION = "0f058d38170d183f9fdee07908f5b515d91793a8"
MAX_AUDIO_SECONDS = 120
CAPTION_WINDOW_SECONDS = 8
MAX_AUDIO_BYTES = 32_000_000
SAMPLE_RATE = 16_000
VOICES = {
    "en-US": "Samantha",
    "en-GB": "Daniel",
    "es-MX": "Paulina",
    "es-ES": "Mónica",
    "el-GR": "Melina",
    "el-CY": "Melina",
    "ja-JP": "Kyoko",
    "fr-FR": "Thomas",
    "fr-CA": "Amélie",
    "de-DE": "Anna",
    "it-IT": "Alice",
    "pt-BR": "Luciana",
    "pt-PT": "Joana",
    "nl-NL": "Xander",
    "zh-CN": "Tingting",
    "ko-KR": "Yuna",
    "ar-001": "Majed",
}


LINUX_VOICES = {
    "en-US": "en-us", "en-GB": "en-gb", "es-MX": "es-419", "es-ES": "es",
    "el-GR": "el", "el-CY": "el", "ja-JP": "ja",
    "fr-FR": "fr-fr", "fr-CA": "fr", "de-DE": "de", "it-IT": "it",
    "pt-BR": "pt-br", "pt-PT": "pt-pt", "nl-NL": "nl", "zh-CN": "cmn",
    "ko-KR": "ko", "ar-001": "ar",
}


def speech_available():
    return sys.platform == "darwin" or shutil.which("espeak-ng") is not None


class SpeechError(RuntimeError):
    pass


def choose_spoken_language(scores, first, second):
    choices = sorted((scores.get(LANGUAGES[code][1], 0.0), code) for code in (first, second))
    (runner_up, _), (best, source) = choices
    if best < 0.25 or best - runner_up < 0.12 or max(scores.values(), default=0) - best > 0.20:
        raise SpeechError("Language unclear. Try a longer phrase in one of the selected languages.")
    return source


class SpeechWorker:
    """Exchange one JSON request at a time with a resident local synthesizer."""

    def __init__(self, command, label):
        self.command = command
        self.label = label
        self.process = None
        self.lock = Lock()
        atexit.register(self.close)

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        process.kill()
        process.wait()
        with suppress(OSError):
            process.stdin.close()
        process.stdout.close()

    def request(self, message, *, timeout=120):
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.process = subprocess.Popen(
                        self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"},
                    )
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
                if not select.select([self.process.stdout], [], [], timeout)[0]:
                    raise TimeoutError
                reply = json.loads(self.process.stdout.readline())
            except (OSError, ValueError):
                self.close()
                raise SpeechError(f"{self.label} did not respond.") from None
            if "error" in reply:
                raise SpeechError(f"{self.label} could not generate this turn.")
            return reply

    def audio(self, message, *, timeout=120):
        return self.decode_audio(self.request(message, timeout=timeout))

    @staticmethod
    def decode_audio(reply):
        import numpy as np

        return reply["rate"], np.frombuffer(base64.b64decode(reply["pcm"]), dtype="<i2")


class MacVoice(SpeechWorker):
    def __init__(self, executable):
        super().__init__([executable], "The native Mac voice")

    def warmup(self):
        for language in ("spa_Latn", "eng_Latn"):
            try:
                self.synthesize(LANGUAGES[language][0], language)
            except SpeechError:
                pass

    def synthesize(self, text, target, locale="", *, style=None):
        locale = locale or LANGUAGES[target][2][0]
        return self.audio({
            "text": text, "locale": "el-GR" if locale == "el-CY" else locale,
            "rate": style.rate if style else 168,
            "pitch_offset": style.pitch_offset if style else 0,
        }, timeout=20)


def read_pcm(path, *, max_seconds):
    import numpy as np

    with wave.open(str(path), "rb") as recording:
        frames = recording.getnframes()
        rate = recording.getframerate()
        if recording.getnchannels() != 1 or recording.getsampwidth() != 2 or rate <= 0:
            raise SpeechError("The audio could not be converted to mono speech.")
        if frames > max_seconds * rate:
            raise SpeechError(f"Record up to {max_seconds} seconds at a time.")
        samples = np.frombuffer(recording.readframes(frames), dtype="<i2").copy()
        if len(samples) != frames or not frames:
            raise SpeechError("The audio file is empty or incomplete.")
    return rate, samples


def load_recording(path):
    import numpy as np

    from imageio_ffmpeg import get_ffmpeg_exe

    if not path:
        raise SpeechError("Record or upload a sentence first.")
    try:
        with Path(path).open("rb") as file:
            data = file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise SpeechError("Use an audio file under 32 MB.")
        decoded = subprocess.run(
            [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-protocol_whitelist", "pipe", "-i", "pipe:0", "-map", "0:a:0",
             "-t", str(MAX_AUDIO_SECONDS + 0.1), "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "s16le", "pipe:1"],
            input=data, check=True, capture_output=True, timeout=15,
        ).stdout
        samples = np.frombuffer(decoded, dtype="<i2")
    except (OSError, subprocess.SubprocessError, ValueError):
        raise SpeechError("Could not read the recording. Try the microphone or a WAV file.") from None

    if len(samples) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        raise SpeechError(f"Record up to {MAX_AUDIO_SECONDS} seconds at a time.")
    if len(samples) < SAMPLE_RATE // 4:
        raise SpeechError("Record at least a quarter second of speech.")
    audio = samples.astype(np.float32) / 32768
    if np.sqrt(np.mean(audio ** 2)) < 0.001:
        raise SpeechError("The recording is very quiet. Check your microphone and try again.")
    return audio


class WhisperTranscriber:
    def __init__(self, device="cpu", *, fast=False, turbo=False):
        self.device = device
        self.fast = fast or turbo
        self.turbo = turbo
        self._mlx_path = None
        self._mlx_staging = None
        self._lock = Lock()
        self._backend = None

    def transcribe(self, path, source_language):
        return self.transcribe_audio(load_recording(path), source_language)

    def transcribe_audio(self, audio, source_language):
        return self._recognize(audio, source_language)[1]

    def transcribe_pair(self, audio, first, second):
        return self._recognize(audio, first, second)

    def _recognize(self, audio, source_language, alternate=None):
        if self.fast:
            with self._lock:
                try:
                    import mlx_whisper

                    if self._mlx_path is None:
                        self._mlx_path = self._cached_mlx_model()
                    if self.turbo and len(audio) <= 30 * SAMPLE_RATE:
                        return self._mlx_short_turn(audio, source_language, alternate)
                    chosen = self._mlx_language(audio, source_language, alternate) if alternate else source_language
                    result = mlx_whisper.transcribe(
                        audio, path_or_hf_repo=self._mlx_path, language=LANGUAGES[chosen][1],
                        temperature=0.0, verbose=None,
                    )
                except SpeechError:
                    raise
                except (ImportError, OSError, RuntimeError, ValueError):
                    if self.turbo:
                        raise SpeechError(
                            "Whisper Turbo is unavailable. Check the Apple Silicon setup and cached model."
                        ) from None
                    self.fast = False
                else:
                    return chosen, self._validate_transcript(result["text"].strip())
        return self._transcribe_torch(audio, source_language, alternate)

    def _cached_mlx_model(self):
        from huggingface_hub import snapshot_download

        if not self.turbo:
            return snapshot_download(
                MLX_ASR_MODEL, revision=MLX_ASR_REVISION, token=False,
                local_files_only=True, allow_patterns=["config.json", "weights.npz"],
            )
        snapshot = Path(snapshot_download(
            MLX_TURBO_MODEL, revision=MLX_TURBO_REVISION, token=False,
            local_files_only=True, allow_patterns=["config.json", "model.safetensors"],
        ))
        config = (snapshot / "config.json").resolve(strict=True)
        weights = (snapshot / "model.safetensors").resolve(strict=True)
        # mlx-whisper 0.4.3 reads the same tensors under a different filename.
        self._mlx_staging = TemporaryDirectory(prefix="say-across-whisper-")
        directory = Path(self._mlx_staging.name)
        (directory / "config.json").symlink_to(config)
        (directory / "weights.safetensors").symlink_to(weights)
        return str(directory)

    def _mlx_short_turn(self, audio, source, alternate):
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self._mlx_path, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        frames = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        # Share one encoder pass across language detection and decoding.
        features = model.embed_audio(frames[None])[0]
        if alternate:
            _, scores = model.detect_language(features)
            source = choose_spoken_language(scores, source, alternate)
        options = DecodingOptions(language=LANGUAGES[source][1], without_timestamps=True)
        result = model.decode(features, options)
        text = result.text.strip()
        if ((result.no_speech_prob > 0.6 and result.avg_logprob < -1.0)
                or result.compression_ratio > 2.4):
            text = ""
        return source, self._validate_transcript(text)

    def _mlx_language(self, audio, first, second):
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self._mlx_path, mx.float16)
        mel = log_mel_spectrogram(audio[:8 * SAMPLE_RATE], n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, scores = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        return choose_spoken_language(scores, first, second)

    def _transcribe_torch(self, audio, source_language, alternate=None):
        with self._lock:
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                if self._backend is None:
                    options = {"revision": ASR_REVISION, "token": False, "trust_remote_code": False}
                    processor = WhisperProcessor.from_pretrained(ASR_MODEL, **options)
                    model = WhisperForConditionalGeneration.from_pretrained(
                        ASR_MODEL, use_safetensors=True, **options
                    ).to(self.device).eval()
                    self._backend = processor, model
                processor, model = self._backend
                long_form = len(audio) > 30 * SAMPLE_RATE
                inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                   return_attention_mask=True, truncation=not long_form,
                                   padding="longest" if long_form else "max_length").to(self.device)
                with torch.inference_mode():
                    if alternate:
                        config = model.generation_config
                        start = torch.tensor([[config.decoder_start_token_id]], device=self.device)
                        segment = inputs.input_features[:, :, :3000]
                        detection_input = {"input_features": segment}
                        if not long_form:
                            encoder_outputs = model.get_encoder()(segment)
                            inputs["encoder_outputs"] = encoder_outputs
                            detection_input = {"encoder_outputs": encoder_outputs}
                        logits = model(**detection_input, decoder_input_ids=start, use_cache=False).logits[0, -1]
                        tokens = list(config.lang_to_id)
                        values = logits[[config.lang_to_id[token] for token in tokens]].softmax(-1).tolist()
                        scores = dict(zip((token[2:-2] for token in tokens), values))
                        source_language = choose_spoken_language(scores, source_language, alternate)
                    language = LANGUAGES[source_language][1]
                    tokens = model.generate(**inputs, language=language, task="transcribe",
                                            return_timestamps=long_form, do_sample=False)
                text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            except SpeechError:
                raise
            except (ImportError, OSError, RuntimeError, ValueError):
                raise SpeechError("Speech recognition failed. Check the Whisper download and try again.") from None
        return source_language, self._validate_transcript(text)

    @staticmethod
    def _validate_transcript(text):
        if not text or not any(char.isalnum() for char in text):
            raise SpeechError("I couldn't make out a sentence. Try speaking clearly and close to the mic.")
        return text


def speak(text, target_language, locale="", *, style=None):
    locale = locale or LANGUAGES[target_language][2][0]
    mac = sys.platform == "darwin"
    voice = (VOICES if mac else LINUX_VOICES).get(locale)
    if not voice:
        raise SpeechError("Playback unavailable for the selected locale.")
    try:
        with TemporaryDirectory(prefix="polyglot-output-") as directory:
            path = Path(directory) / "translation.wav"
            if mac:
                command = ["/usr/bin/say", "-v", voice, "-o", str(path), "--file-format=WAVE",
                           "--data-format=LEI16@22050"]
                text = text.replace("[[", "[ [")
                if style is not None:
                    command.extend(["-r", str(style.rate)])
                    text = f"[[pmod {style.pitch_range}]] [[pbas {style.pitch_offset:+d}]] {text}"
            else:
                command = ["espeak-ng", "--stdin", "-b", "1", "-v", voice, "-w", str(path)]
                if style is not None:
                    command.extend(["-s", str(style.rate), "-p", str(50 + style.pitch_offset)])
            subprocess.run(command, input=text.encode("utf-8"),
                           check=True, capture_output=True, timeout=25)
            return read_pcm(path, max_seconds=300)
    except (OSError, subprocess.SubprocessError, wave.Error, EOFError, SpeechError):
        raise SpeechError(f"Playback unavailable. Check that voice {voice} is installed.") from None


def decode_live_phrase(encoded, *, max_seconds=MAX_AUDIO_SECONDS):
    """Accept only bounded 16 kHz mono PCM from the live browser recorder."""
    import numpy as np

    max_bytes = 2 * SAMPLE_RATE * max_seconds
    if not encoded or len(encoded) > 4 * ((max_bytes + 2) // 3):
        raise SpeechError("The live phrase was too long or empty.")
    try:
        pcm = base64.b64decode(encoded, validate=True)
        if not 2 * SAMPLE_RATE // 5 <= len(pcm) <= max_bytes:
            raise ValueError
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    except (ValueError, binascii.Error):
        raise SpeechError("Could not read the live phrase.") from None


def apply_vocal_emphasis(audio, style):
    """Transfer the broad loudness contour onto the translated speech."""
    import numpy as np

    if style is None or not any(style.emphasis_db):
        return audio
    rate, samples = audio
    signal = samples.astype(np.float32)
    amplitude = np.abs(signal)
    audible = np.flatnonzero(amplitude > max(128, amplitude.max(initial=0) * 0.06))
    if len(audible) < 2:
        return audio
    anchors = (np.arange(len(style.emphasis_db)) + 0.5) / len(style.emphasis_db)
    progress = (np.arange(len(signal)) - audible[0]) / (audible[-1] - audible[0])
    gains = 10 ** (np.interp(progress, anchors, style.emphasis_db) / 20)
    shaped = signal * gains
    peak = np.abs(shaped).max()
    if peak > 32767:
        shaped *= 32767 / peak
    return rate, np.rint(shaped).astype("<i2")


def adjust_voice_pace(audio, rate):
    """Apply an explicit pace to matched speech while retaining its pitch."""
    import numpy as np
    from imageio_ffmpeg import get_ffmpeg_exe

    sample_rate, pcm = audio
    format_args = ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"]
    try:
        result = subprocess.run([get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                                 *format_args, "-i", "pipe:0", "-af", f"atempo={rate / 168:.3f}",
                                 *format_args, "pipe:1"], input=pcm.tobytes(), capture_output=True,
                                timeout=5, check=True)
        if result.stdout:
            return sample_rate, np.frombuffer(result.stdout, dtype="<i2")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return audio


def wav_base64(audio):
    rate, samples = audio
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())
    return base64.b64encode(output.getvalue()).decode("ascii")
