"""Isolated MLX voice-matching engine. Receives local JSON lines on standard input."""

import base64
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path

REPO = "mlx-community/chatterbox-4bit"
REVISION = "f1d7b9696e1b6242e64eb8c4a823b6d1a50425a8"
TOKENIZER_REPO = "mlx-community/S3TokenizerV2"
TOKENIZER_REVISION = "e0c9886f0e1c35ae85b1f27277416fb19fc72bec"
PROFILE_SIMILARITY_FLOOR = 0.75


def load():
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model

    path = snapshot_download(
        REPO, revision=REVISION, token=False, local_files_only=True,
        allow_patterns=["config.json", "model.safetensors*", "conds.safetensors", "tokenizer.json",
                        "Cangjie5_TC.json"],
    )
    # The MLX Audio loader looks up this auxiliary snapshot by its cached main ref.
    tokenizer = snapshot_download(TOKENIZER_REPO, token=False, local_files_only=True,
                                  allow_patterns=["model.safetensors", "config.json"])
    if Path(tokenizer).name != TOKENIZER_REVISION:
        raise ValueError("The cached speech tokenizer does not match the tested revision.")
    return load_model(path)


def _prompt(encoded):
    import mlx.core as mx
    import numpy as np

    reference = np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<i2")
    return mx.array(reference.astype(np.float32) / 32768)


def _profile_conditionals(model, prompt, profile, exaggeration):
    import mlx.core as mx

    conds = model.prepare_conditionals(prompt, 16000, exaggeration)
    identity = mx.mean(model.ve.embeds_from_wavs([profile], sample_rate=16000), axis=0, keepdims=True)
    live = conds.t3.speaker_emb
    cosine = mx.sum(identity * live) / (mx.linalg.norm(identity) * mx.linalg.norm(live) + 1e-8)
    use_profile = float(cosine) >= PROFILE_SIMILARITY_FLOOR
    if use_profile:
        # Keep the current turn's prompt tokens and acoustic features for delivery.
        conds.t3.speaker_emb = identity
        conds.gen["embedding"] = model.s3gen.speaker_encoder.inference(profile[None])
    return conds, use_profile


def render(model, request):
    import numpy as np

    prompt = _prompt(request["reference"])
    options = {"audio_prompt": prompt, "audio_prompt_sr": 16000}
    used_profile = False
    if request.get("profile"):
        conds, used_profile = _profile_conditionals(model, prompt, _prompt(request["profile"]), request["exaggeration"])
        options = {"conds": conds}
    audio = [np.asarray(item.audio, dtype=np.float32).reshape(-1) for item in model.generate(
        text=request["text"], lang_code=request["language"], exaggeration=request["exaggeration"],
        verbose=False, **options,
    )]
    if not audio:
        raise ValueError("No generated speech")
    samples = np.concatenate(audio)
    if not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError("Invalid generated audio")
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    return {"rate": model.sample_rate, "pcm": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "profile_used": used_profile}


def main():
    model = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if model is None:
                    model = load()
                if request.get("warm"):
                    import mlx.core as mx

                    for chunk in model.generate("Ready to translate.", lang_code="en", verbose=False):
                        mx.eval(chunk.audio)
                    result = {"ready": True}
                else:
                    result = render(model, request)
        except Exception as error:
            result = {"error": type(error).__name__}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
