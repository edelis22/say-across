"""Launch with python app.py or use --demo for fixed offline examples."""

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import nullcontext
from functools import partial
from html import escape
import os
from pathlib import Path
from threading import Thread
from time import perf_counter

from pragmatics.adaptation import OpenAIAdapter, post_json
from pragmatics.assistant import (
    AssistantError, ClaudeAssistant, MAX_PASSAGE_CHARS, MAX_QUESTION_CHARS, assistant_available,
    validate_simplification,
)
from pragmatics.captions import caption_cues
from pragmatics.config import EXPLANATION_LEVELS, LANGUAGES, RELATIONSHIPS, TONES, VOICE_RANGES
from pragmatics.local_assistant import LocalAssistant, local_assistant_available
from pragmatics.demo import DemoAdapter, DemoTranslator, SOURCE, TARGETS
from pragmatics.emotion import EmotionDetector
from pragmatics.interpreter import (
    InterpreterSettings, VOICE_TONES, intended_delivery, local_interpretation, reviewed_spoken_translation,
)
from pragmatics.pipeline import Sentence, run, validate_text
from pragmatics.prosody import analyze_turn
from pragmatics.speech import (
    CAPTION_WINDOW_SECONDS, MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS, MacVoice, SpeechError, WhisperTranscriber,
    adjust_voice_pace, apply_vocal_emphasis, decode_live_phrase, load_recording, resolve_voice_range, speak,
    speech_available, wav_base64,
)
from pragmatics.translation import NLLB
from pragmatics.validation import numbers
from pragmatics.voice_match import MIN_PROFILE_SPEECH_SECONDS, VOICE_PROFILE_SECONDS, VoiceMatch


def describe_emotion(reading):
    if reading is None:
        return "Emotion detection is off.", None
    if reading.mode == "voice_prosody":
        messages = {
            "detected": "Possible vocal expression · " + ", ".join(reading.labels),
            "neutral": "Vocal expression · neutral",
            "uncertain": "Vocal expression · uncertain",
            "unavailable": "Vocal expression analysis unavailable",
        }
    else:
        messages = {
            "detected": "Possible emotion from wording · " + ", ".join(reading.labels),
            "neutral": "Neutral wording",
            "uncertain": "Uncertain · the text did not provide a clear signal",
            "unavailable": "Emotion analysis unavailable · see details below",
        }
    top_scores = []
    if reading.mode != "voice_prosody" or reading.labels:
        top_scores = sorted(reading.scores.items(), key=lambda entry: -entry[1])[:5]
    return messages[reading.status], {
        "status": reading.status,
        "reason": reading.reason,
        "mode": reading.mode,
        "top_uncalibrated_scores": {label: round(score, 3) for label, score in top_scores},
    }


ASSISTANT_MESSAGES = {
    "assistant_unavailable": "The assistant is off. Restart with ANTHROPIC_API_KEY set to enable it.",
    "missing_api_key": "The assistant is off. Restart with ANTHROPIC_API_KEY set to enable it.",
    "assistant_sdk_missing": "Install requirements-assistant.txt to enable the assistant.",
    "credential_rejected": "The assistant key was rejected. Check ANTHROPIC_API_KEY.",
    "rate_limited": "The assistant is rate limited. Try again shortly.",
    "provider_unreachable": "Could not reach the assistant. Check the connection and try again.",
    "provider_refusal": "The assistant declined this request.",
    "model_abstained": "The assistant could not restate this faithfully at that level.",
    "explanation_timed_out": "The simpler wording took too long for this turn.",
    "empty_question": "Type a question first.",
    "question_too_long": f"Keep the question under {MAX_QUESTION_CHARS} characters.",
    "unsupported_level": "Select a supported explanation level.",
    "unsupported_language": "Select a configured language.",
    "empty_answer": "The assistant returned no answer. Try rewording the question.",
    "translation_too_long": "This turn is too long to restate. Try a shorter turn.",
    "empty_translation": "There is nothing to restate yet.",
    "local_model_unreachable": "The local model did not respond. Check that Ollama is running.",
    "local_model_missing": "No local model is installed. Run: ollama pull qwen2.5:7b",
    "no_reference_found": "No reference passage was found for that question. Try rewording it.",
    "research_unreachable": "Could not reach the reference source. Check the connection.",
}


PORT_SEARCH = 12


def serve(ui, port, **options):
    """Launch on the first free port at or above `port`.

    A stale copy of the app holding the default port used to stop a new one
    from starting at all, which reads as the app being broken.
    """
    last = port + PORT_SEARCH - 1
    for candidate in range(port, last + 1):
        try:
            return ui.launch(server_port=candidate, **options)
        except OSError:
            if candidate == last:
                raise
            print(f"Port {candidate} is in use. Trying {candidate + 1}...", flush=True)
    return None


def resolve_assistant(allow_assistant, demo):
    """Pick the best assistant this machine can actually run.

    Claude when a credential is present, otherwise a free local model through
    Ollama, otherwise nothing and the features stay hidden. CLICK_ASSISTANT
    pins a tier: "claude", "local" or "off".
    """
    if not allow_assistant or demo:
        return None
    pinned = (os.environ.get("CLICK_ASSISTANT") or "").strip().lower()
    if pinned == "off":
        return None
    if pinned == "local":
        return LocalAssistant() if local_assistant_available() else None
    if pinned == "claude":
        return ClaudeAssistant() if assistant_available() else None
    if assistant_available():
        return ClaudeAssistant()
    if local_assistant_available():
        return LocalAssistant()
    return None


def _lookup_message(reason):
    return ASSISTANT_MESSAGES.get(reason, "The assistant could not complete this request.")


def build_app(demo=False, device="cpu", detect_emotion=True, allow_remote=True, *,
              fast_whisper=False, whisper_turbo=False, fast_nllb=None, clone_executable=None,
              native_voice_executable=None, warm=False, allow_assistant=True):
    import gradio as gr

    translator = DemoTranslator() if demo else NLLB(device, turn_beams=1, fast_model=fast_nllb)
    adapter = DemoAdapter() if demo else OpenAIAdapter() if allow_remote else None
    live_adapter = (OpenAIAdapter(transport=partial(post_json, timeout=6))
                    if not demo and allow_remote and os.environ.get("OPENAI_API_KEY") else None)
    detector = EmotionDetector() if detect_emotion and not demo else None
    transcriber = WhisperTranscriber(device, fast=fast_whisper, turbo=whisper_turbo) if not demo else None
    system_voice = bool(transcriber and speech_available())
    voice_match = VoiceMatch(clone_executable) if transcriber and clone_executable else None
    mac_voice = MacVoice(native_voice_executable) if transcriber and native_voice_executable else None
    assistant = resolve_assistant(allow_assistant, demo)
    # Re-rendering runs beside playback so captions and audio are never delayed by it.
    helpers = (ThreadPoolExecutor(max_workers=4, thread_name_prefix="click-assistant")
               if assistant is not None else None)

    if warm and transcriber:
        def warm_recognizer():
            import numpy as np

            silence = np.zeros(16000, dtype=np.float32)
            try:
                transcriber.transcribe_audio(silence, "eng_Latn")
            except SpeechError:
                pass
            try:
                transcriber.transcribe_pair(silence, "eng_Latn", "spa_Latn")
            except SpeechError:
                pass

        def warm_translator():
            try:
                translator.translate_turn("Hello.", "eng_Latn", "spa_Latn")
            except (ImportError, OSError, RuntimeError, ValueError):
                pass

        Thread(target=warm_recognizer, daemon=True).start()
        Thread(target=warm_translator, daemon=True).start()
        if mac_voice:
            Thread(target=mac_voice.warmup, daemon=True).start()
        if voice_match:
            def warm_voice():
                try:
                    voice_match.warmup()
                except SpeechError:
                    pass

            Thread(target=warm_voice, daemon=True).start()

    def explain(text, language, level, *, source="", situation=""):
        """Re-render an accepted translation for a different listener."""
        if assistant is None:
            return {"error": "assistant_unavailable"}
        if level not in EXPLANATION_LEVELS or not EXPLANATION_LEVELS[level][1]:
            return None
        try:
            candidate, model = assistant.simplify(translation=text, language=language, level=level,
                                                  source=source, situation=situation)
        except AssistantError as error:
            return {"error": str(error)}
        reason = validate_simplification(candidate)
        if reason:
            return {"error": reason}
        rendered = candidate["rendered"].strip()
        return {
            "level": level,
            "level_label": EXPLANATION_LEVELS[level][0],
            "rendered": rendered,
            "glossary": [{"term": entry["term"].strip(), "meaning": entry["meaning"].strip()}
                         for entry in candidate["glossary"] if entry["term"].strip()][:12],
            "omitted": candidate["omitted"].strip(),
            # A surfaced caution, not a rejection: simpler wording may legitimately
            # restate a quantity, and a silent fallback would hide that from a reader
            # who needs the number.
            "check_numbers": numbers(rendered) != numbers(text),
            "response_model": model,
        }

    def explain_text(text, language, level, situation=""):
        detail = explain(text or "", language, level, situation=situation or "")
        if detail is None:
            return "", {"status": "unchanged", "detail": "Select a level other than As spoken."}
        if "error" in detail:
            return "", {"status": "unavailable", "reason": detail["error"]}
        return detail["rendered"], {"status": "rendered", **detail}

    def look_up(question, language, place="", situation=""):
        """Answer a practical question, searching the web when that changes the answer."""
        if assistant is None:
            return "", {"status": "unavailable", "reason": "assistant_unavailable"}, []
        try:
            answer, model = assistant.look_up(question=question or "", language=language,
                                              place=place or "", situation=situation or "")
        except AssistantError as error:
            return "", {"status": "unavailable", "reason": str(error)}, []
        return answer["answer"], {"status": "answered", "searched": answer["searched"],
                                 "response_model": model}, answer["sources"]

    def say_answer_in(answer, language):
        """Repeat an answer already given, in another language."""
        if assistant is None:
            return "", {"status": "unavailable", "reason": "assistant_unavailable"}
        try:
            restated, model = assistant.say_in(answer=answer or "", language=language)
        except AssistantError as error:
            return "", {"status": "unavailable", "reason": str(error)}
        return restated, {"status": "restated", "language": language, "response_model": model}

    def live_answer_language(answer, language):
        restated, detail = say_answer_in(answer, language)
        return {"answer": restated, "language": language,
                "error": "" if restated else _lookup_message(detail.get("reason", ""))}

    def live_answer_audio(text, language):
        """Read an answer aloud in the language it is written in."""
        if language not in LANGUAGES:
            return {"error": "Select a configured language."}
        if not (text or "").strip():
            return {"error": "There is nothing to read aloud yet."}
        if len(text) > MAX_PASSAGE_CHARS:
            return {"error": "This answer is too long to read aloud."}
        try:
            if mac_voice is not None:
                audio = mac_voice.synthesize(text, language)
            elif system_voice:
                audio = speak(text, language)
            else:
                return {"error": "This host has no system voice for playback."}
        except SpeechError as error:
            return {"error": str(error)}
        return {"audio": wav_base64(audio)}

    def live_lookup(question, language, place="", situation=""):
        answer, detail, sources = look_up(question, language, place, situation)
        return {"answer": answer, "sources": sources, "searched": detail.get("searched", False),
                "error": "" if answer else _lookup_message(detail.get("reason", ""))}

    def live_explain(text, language, level, situation=""):
        detail = explain(text or "", language, level, situation=situation or "")
        if detail is None:
            return {}
        if "error" in detail:
            return {"error": _lookup_message(detail["error"])}
        return detail

    def analyze_emotion(source, before, after):
        try:
            validate_text(source, before, after)
            summary, detail = describe_emotion(detector.detect(source, before, after))
            return "", "", summary, {"expressed_emotion": detail}
        except ValueError as error:
            return "", "", "", {"status": "input_error", "detail": str(error)}

    def translate(source, src, tgt, before, after, relationship, locale, tone, emotion, opt_in,
                  *, voice_reading=None):
        long_turn = voice_reading is not None and len(source) > 500
        try:
            result = run(Sentence(source, src, tgt, before, after, relationship, locale, tone,
                                  emotion), translator, adapter if opt_in and not long_turn else None,
                         emotion_detector=detector if voice_reading is None else None,
                         emotion_reading=voice_reading, spoken_turn=voice_reading is not None)
        except ValueError as error:
            return "", "", "", {"status": "input_error", "detail": str(error)}
        except (ImportError, OSError, RuntimeError):
            return "", "", "", {"status": "translation_error",
                                "detail": "Check dependencies, model cache, device and connectivity."}
        summary, emotion_detail = describe_emotion(result.emotion)
        return result.baseline, result.output if result.status == "adapted" else "", summary, {
            "mode": "fixed fixture" if demo else "research",
            "status": result.status,
            "fallback_reason": "long_turn_uses_local_translation" if opt_in and long_turn else result.reason,
            "explanation": result.explanation,
            "intended_emotion_user_supplied": result.intended_emotion,
            "expressed_emotion": emotion_detail,
            "emotion_hints_for_adaptation": list(result.emotion_hints),
            "ambiguity": result.ambiguity,
            "uncalibrated_adequacy_estimate": None if demo else result.adequacy_probability,
            "response_model": result.response_model,
            "nllb_ms": round(result.nllb_ms, 1),
            "emotion_ms": round(result.emotion_ms, 1),
            "adaptation_ms": round(result.adaptation_ms, 1),
        }

    def translate_and_speak(source, src, tgt, before, after, relationship, locale, tone, emotion, opt_in,
                            *, voice_reading=None, voice_style=None):
        result = translate(source, src, tgt, before, after, relationship, locale, tone, emotion, opt_in,
                           voice_reading=voice_reading)
        baseline, adapted, _, detail = result
        if not baseline:
            return *result, None, detail.get("detail", "Translation unavailable.")
        if not system_voice:
            return *result, None, "Translation ready."
        try:
            audio = speak(adapted or baseline, tgt, locale, style=voice_style, voice_range="auto")
        except SpeechError as error:
            return *result, None, str(error)
        audio = apply_vocal_emphasis(audio, voice_style)
        version = "adapted translation" if adapted else "NLLB translation"
        return *result, audio, f"Spoken {version} is ready. Check the transcript for misheard words."

    def translate_recording(recording, src, tgt, before, after, relationship, locale, tone, emotion, opt_in):
        if src == tgt:
            return "", "", "", "", {}, None, "Choose two different languages."
        try:
            signal = load_recording(recording)
            transcript = transcriber.transcribe_audio(signal, src)
        except SpeechError as error:
            return "", "", "", "", {}, None, str(error)
        voice_reading, style, _ = analyze_turn(signal, enabled=detect_emotion, source_language=src)
        return transcript, *translate_and_speak(
            transcript, src, tgt, before, after, relationship, locale, tone, emotion, opt_in,
            voice_reading=voice_reading, voice_style=style,
        )

    def recognize_live(encoded, src, tgt, *, two_way=False, max_seconds=MAX_AUDIO_SECONDS):
        if src not in LANGUAGES or tgt not in LANGUAGES:
            raise SpeechError("Choose supported languages for captions.")
        signal = decode_live_phrase(encoded, max_seconds=max_seconds)
        if two_way and src != tgt:
            actual, transcript = transcriber.transcribe_pair(signal, src, tgt)
            return signal, transcript, actual, tgt if actual == src else src
        return signal, transcriber.transcribe_audio(signal, src), src, tgt

    def live_translation(transcript, src, tgt):
        validate_text(transcript, "", "", source_limit=12000)
        return transcript if src == tgt else translator.translate_turn(transcript, src, tgt)

    def draft_caption(encoded, src, tgt, options=""):
        try:
            settings = InterpreterSettings.parse(options)
            _, transcript, src, tgt = recognize_live(encoded, src, tgt, two_way=settings.two_way,
                                                      max_seconds=CAPTION_WINDOW_SECONDS)
        except (SpeechError, ValueError):
            return {}
        try:
            translated = reviewed_spoken_translation(transcript, src, tgt) or live_translation(transcript, src, tgt)
        except (ImportError, OSError, RuntimeError, ValueError):
            translated = ""
        return {"transcript": transcript, "translation": translated, "source_language": src, "target_language": tgt}

    def live_playback(translated, tgt, voice_mode, signal, style, tone, profile="", voice_range=""):
        engine, notice, audio = "Expressive voice", "", None
        if voice_mode == "match":
            try:
                if voice_match is None:
                    raise SpeechError("Voice matching is unavailable on this host.")
                used_profile = False
                if profile:
                    audio, used_profile = voice_match.synthesize_profiled(translated, tgt, signal, style, profile)
                    if not used_profile:
                        notice = "AI voice match used this turn's recording as the reference."
                else:
                    audio = voice_match.synthesize(translated, tgt, signal, style)
                pace = style.rate if tone != "auto" else 168
                if tone == "auto" and used_profile:
                    pace = round((168 + style.rate) / 2)
                if pace != 168:
                    audio = adjust_voice_pace(audio, pace)
                engine = "AI voice match · personal profile" if used_profile else "AI voice match"
            except SpeechError as error:
                notice = f"{error} Playing the expressive voice."
        if audio is None and mac_voice:
            try:
                audio = mac_voice.synthesize(translated, tgt, style=style, voice_range=voice_range)
            except SpeechError:
                pass
        if audio is None:
            try:
                audio = speak(translated, tgt, style=style, voice_range=voice_range)
            except SpeechError as error:
                notice = str(error)
        return engine, notice, apply_vocal_emphasis(audio, style) if audio is not None else None

    def translate_live(encoded, src, tgt, voice_mode="expressive", options="", profile=""):
        started = perf_counter()
        try:
            settings = InterpreterSettings.parse(options)
            signal, transcript, src, tgt = recognize_live(encoded, src, tgt, two_way=settings.two_way)
        except (SpeechError, ValueError) as error:
            yield {"error": str(error)}
            return
        asr_done = perf_counter()
        direction = {"source_language": src, "target_language": tgt}
        try:
            reviewed = reviewed_spoken_translation(transcript, src, tgt)
            if reviewed:
                translated, note = reviewed, "Used a reviewed translation for this expression."
            else:
                supplied, plain, note = local_interpretation(
                    transcript, src, tgt, settings) if src != tgt else (None, transcript, "")
                translated = supplied or live_translation(plain, src, tgt)
        except (ImportError, OSError, RuntimeError, ValueError):
            yield {**direction, "transcript": transcript, "error": "Translated captions are temporarily unavailable."}
            return
        nllb_done = perf_counter()
        if not reviewed and src != tgt and settings.remote and live_adapter and len(transcript) <= 500:
            tone = settings.tone if settings.tone in TONES else "preserve"
            sentence = Sentence(transcript, src, tgt, before=settings.situation,
                                relationship=settings.relationship, tone=tone)
            revised = run(sentence, translator, live_adapter, baseline=translated)
            if revised.status == "adapted":
                translated, note = revised.output, "Context-aware wording applied."
            elif not note:
                note = "Using local wording for this turn."
        reading, style, changes = analyze_turn(signal, enabled=detect_emotion, source_language=src)
        style = intended_delivery(style, settings.tone)
        summary = describe_emotion(reading)[0] if reading.status in {"detected", "neutral"} else ""
        if src == tgt or not (system_voice or voice_match or mac_voice):
            voice_mode = "captions"
        band = resolve_voice_range(settings.voice_range, style)
        common = {**direction, "transcript": transcript, "translation": translated, "emotion": summary,
                  "cues": caption_cues(changes, tgt), "voice_mode": voice_mode, "context_note": note,
                  "tone": settings.tone, "tone_label": VOICE_TONES[settings.tone][0],
                  "voice_band": band, "level": settings.level}
        text_done = perf_counter()
        yield {**common, "stage": "translated"}
        pending = None
        if assistant is not None and src != tgt and settings.level != "as_spoken":
            pending = helpers.submit(explain, translated, tgt, settings.level,
                                     source=transcript, situation=settings.situation)
        engine, notice, audio = "", "", None
        if voice_mode != "captions":
            reference = profile if voice_mode == "match" and src == settings.profile_language else ""
            engine, notice, audio = live_playback(translated, tgt, voice_mode, signal, style,
                                                  settings.tone, reference, settings.voice_range)
        explanation = {}
        if pending is not None:
            try:
                explanation = pending.result(timeout=25) or {}
            except FutureTimeout:
                explanation = {"error": "explanation_timed_out"}
            if "error" in explanation:
                explanation = {"error": _lookup_message(explanation["error"])}
        yield {**common, "stage": "ready", "voice_engine": engine if audio is not None else "",
               "audio": wav_base64(audio) if audio is not None else "", "playback_error": notice,
               "explanation": explanation,
               "timings_ms": {"recognition": round((asr_done - started) * 1000),
                              "nllb": round((nllb_done - asr_done) * 1000),
                              "text_ready": round((text_done - started) * 1000),
                              "voice_ready": round((perf_counter() - started) * 1000)}}

    languages = [(label, code) for code, (label, _, _) in LANGUAGES.items()
                 if not demo or code == "eng_Latn" or code in TARGETS]
    assets = Path(__file__).parent / "pragmatics"
    live_js = (assets / "live.js").read_text(encoding="utf-8") if transcriber else None
    live_css = (assets / "live.css").read_text(encoding="utf-8") if transcriber else None

    with gr.Blocks(title="Click", analytics_enabled=False, css=live_css,
                   delete_cache=(300, 1800), js=live_js) as ui:
        if transcriber:
            def options(selected):
                return "".join(
                    f'<option value="{escape(code)}" data-lang="{escape(LANGUAGES[code][2][0])}" '
                    f'{"selected" if code == selected else ""}>{escape(label)}</option>' for label, code in languages
                )

            playback_options = (
                ('<option value="expressive">Expressive · faster</option>' if system_voice or mac_voice else "")
                + ('<option value="match">Match my voice · AI-generated</option>' if voice_match else "")
                + f'<option value="captions" {"" if system_voice or mac_voice else "selected"}>Captions only</option>'
            )

            live_html = (assets / "live.html").read_text(encoding="utf-8").format(
                max_seconds=MAX_AUDIO_SECONDS, caption_seconds=CAPTION_WINDOW_SECONDS,
                profile_seconds=VOICE_PROFILE_SECONDS, profile_speech_seconds=MIN_PROFILE_SPEECH_SECONDS,
                profile_visibility="" if voice_match else "hidden",
                language_count=len(LANGUAGES), source_options=options("eng_Latn"),
                target_options=options("spa_Latn"), playback_options=playback_options,
                tone_options="".join(f'<option value="{key}">{label}</option>' for key, (label, _) in VOICE_TONES.items()),
                voice_range_options="".join(f'<option value="{escape(key)}">{escape(label)}</option>'
                                            for key, label in VOICE_RANGES.items()),
                level_options="".join(f'<option value="{escape(key)}">{escape(label)}</option>'
                                      for key, (label, _) in EXPLANATION_LEVELS.items()),
                assistant_visibility="" if assistant else "hidden",
                assistant_engine=escape(getattr(assistant, "label", "")),
                playback_visibility="" if (system_voice or mac_voice) else "hidden",
                question_limit=MAX_QUESTION_CHARS,
                relationship_options="".join(f'<option value="{key}">{key.title()}</option>' for key in RELATIONSHIPS),
                remote_option=(
                    '<label class="say-check"><input type="checkbox" data-live="remote"> '
                    'Use OpenAI for broader context · sends the transcript and context and may add delay</label>'
                    if live_adapter else ""
                ),
            )
            gr.HTML(live_html)
            pcm, live_src, live_tgt, voice_mode, live_settings, profile = [gr.Textbox(visible=False) for _ in range(6)]
            reply = gr.JSON(visible=False)
            route = gr.Button(visible=False)
            route.click(translate_live, [pcm, live_src, live_tgt, voice_mode, live_settings, profile], reply, api_name="live_phrase",
                        show_api=False, concurrency_id="live_turn", concurrency_limit=2)
            gr.Button(visible=False).click(
                draft_caption, [pcm, live_src, live_tgt, live_settings], reply, api_name="live_caption",
                show_api=False, concurrency_id="caption_draft",
            )
            if assistant is not None:
                ask_inputs = [gr.Textbox(visible=False) for _ in range(4)]
                ask_reply = gr.JSON(visible=False)
                gr.Button(visible=False).click(
                    live_lookup, ask_inputs, ask_reply, api_name="live_lookup", show_api=False,
                    concurrency_id="assistant", concurrency_limit=2,
                )
                answer_inputs = [gr.Textbox(visible=False) for _ in range(2)]
                answer_reply = gr.JSON(visible=False)
                gr.Button(visible=False).click(
                    live_answer_language, answer_inputs, answer_reply,
                    api_name="live_answer_language", show_api=False,
                    concurrency_id="assistant", concurrency_limit=2,
                )
                speech_inputs = [gr.Textbox(visible=False) for _ in range(2)]
                speech_reply = gr.JSON(visible=False)
                gr.Button(visible=False).click(
                    live_answer_audio, speech_inputs, speech_reply,
                    api_name="live_answer_audio", show_api=False, concurrency_id="translation",
                )
                explain_inputs = [gr.Textbox(visible=False) for _ in range(4)]
                explain_reply = gr.JSON(visible=False)
                gr.Button(visible=False).click(
                    live_explain, explain_inputs, explain_reply, api_name="live_explain", show_api=False,
                    concurrency_id="assistant", concurrency_limit=2,
                )
        if demo:
            gr.Markdown("**Offline fixture mode.** Outputs are fixed examples. Keep context and "
                        "preferences at their defaults to see both panes.")
        with gr.Accordion("Text and file tools", open=False, elem_classes=["say-tools"]) if transcriber else nullcontext():
            with gr.Row():
                src = gr.Dropdown(languages, value="eng_Latn", label="Source language")
                tgt = gr.Dropdown(languages, value="spa_Latn", label="Target language")
                locale = gr.Dropdown([("Unspecified", ""), *LANGUAGES["spa_Latn"][2]],
                                     value="", label="Target locale")
                if transcriber:
                    swap = gr.Button("⇄ Swap languages")
            tgt.change(lambda code: gr.Dropdown(
                choices=[("Unspecified", ""), *LANGUAGES[code][2]], value=""),
                inputs=tgt, outputs=locale, api_name=False)
            if transcriber:
                recording = gr.Audio(sources=["upload"], type="filepath", label="Upload a recording")
                gr.Markdown(f"Upload up to {MAX_AUDIO_SECONDS} seconds of audio to translate automatically.")
            source = gr.Textbox(label="Transcript or typed sentence" if transcriber else "Sentence",
                                value=SOURCE if demo else "", lines=3)
            if transcriber:
                spoken = gr.Audio(label="Spoken translation", interactive=False, autoplay=True, format="wav")
                voice_status = gr.Textbox(label="Voice status", interactive=False)
            with gr.Accordion("Optional context and tone", open=False):
                with gr.Row():
                    before = gr.Textbox(label="Preceding context (optional)")
                    after = gr.Textbox(label="Following context (optional)")
                with gr.Row():
                    relationship = gr.Dropdown(list(RELATIONSHIPS), value="unspecified",
                                               label="Recipient relationship to speaker")
                    tone = gr.Dropdown(list(TONES), value="preserve", label="Preferred tone")
                    emotion = gr.Textbox(label="Speaker's intended emotion (optional)")
            opt_in = gr.Checkbox(label="Show fixture adaptation" if demo else
                                 "Adapt with OpenAI · send sentence, context, draft and any emotion labels",
                                 value=demo, visible=demo or allow_remote)
            with gr.Row():
                button = gr.Button("Translate text and speak" if system_voice else "Translate",
                                   variant="secondary" if transcriber else "primary")
                inspect = gr.Button("Analyze wording only", visible=detector is not None)
            emotion_summary = gr.Textbox(label="Expressed emotion estimate", interactive=False,
                                         info="Recordings use vocal tone. Typed text uses a separate wording analysis.")
            with gr.Row():
                baseline = gr.Textbox(label="Literal baseline · fixed fixture" if demo else
                                      "Literal baseline · raw NLLB", interactive=False)
                adapted = gr.Textbox(label="Context-aware adaptation", interactive=False)
            with gr.Accordion("Decision details and scores", open=False):
                detail = gr.JSON(label="Translation decision and emotion analysis")
            with gr.Accordion("Explain at another level", open=False, visible=assistant is not None):
                gr.Markdown("Restates the translation above for a different listener. The original "
                            "translation is kept and shown unchanged.")
                with gr.Row():
                    level = gr.Dropdown([(label, key) for key, (label, audience) in
                                         EXPLANATION_LEVELS.items() if audience],
                                        value="plain", label="Explanation level")
                    restate = gr.Button("Restate translation")
                restated = gr.Textbox(label="Restated translation", interactive=False, lines=4)
                restated_detail = gr.JSON(label="Terms explained and what the simpler wording loses")
            with gr.Accordion("Ask about where you are", open=False, visible=assistant is not None):
                gr.Markdown("Asks Claude a practical question and searches the web when current or "
                            "local facts would change the answer. Answers are informational only.")
                question = gr.Textbox(label="Question", lines=2, max_lines=4,
                                      placeholder="For example, how do I pay for a bus ticket here?")
                with gr.Row():
                    question_language = gr.Dropdown(languages, value="eng_Latn", label="Answer in")
                    place = gr.Textbox(label="Where you are (optional)",
                                       placeholder="For example, Athens, Greece")
                ask = gr.Button("Ask")
                answer = gr.Textbox(label="Answer", interactive=False, lines=8)
                with gr.Row():
                    answer_language = gr.Dropdown(languages, value="spa_Latn", label="Repeat the answer in")
                    repeat = gr.Button("Repeat in this language")
                repeated = gr.Textbox(label="The same answer, restated", interactive=False, lines=8)
                with gr.Row():
                    answer_sources = gr.JSON(label="Sources the answer used")
                    answer_detail = gr.JSON(label="Lookup status")
            outputs = [baseline, adapted, emotion_summary, detail]
            text_inputs = [source, src, tgt, before, after, relationship, locale, tone, emotion, opt_in]
            if transcriber:
                voice_outputs = [*outputs, spoken, voice_status]
                recording_inputs = [recording, *text_inputs[1:]]
                recording_outputs = [source, *voice_outputs]
                recording.upload(translate_recording, recording_inputs, recording_outputs,
                                 api_name="translate_recording", concurrency_id="translation")
                button.click(translate_and_speak, text_inputs, voice_outputs, api_name="translate",
                             concurrency_id="translation")
                swap.click(
                    lambda source_lang, target_lang: (
                        target_lang, source_lang,
                        gr.Dropdown(choices=[("Unspecified", ""), *LANGUAGES[source_lang][2]], value=""),
                        None, "", "", "", "", {}, None, "",
                    ),
                    [src, tgt], [src, tgt, locale, recording, source, *voice_outputs],
                    api_name=False, concurrency_id="translation",
                )
            else:
                button.click(translate, text_inputs, outputs, api_name="translate")
            if detector is not None:
                if transcriber:
                    def analyze_and_clear_audio(source, before, after):
                        return *analyze_emotion(source, before, after), None, ""
                    inspect.click(analyze_and_clear_audio, [source, before, after], voice_outputs,
                                  api_name="detect_emotion", concurrency_id="translation")
                else:
                    inspect.click(analyze_emotion, [source, before, after], outputs, api_name="detect_emotion")
            if assistant is not None:
                def restate_translation(baseline_text, adapted_text, target, chosen, situation):
                    text = (adapted_text or baseline_text or "").strip()
                    if not text:
                        return "", {"status": "input_error", "detail": "Translate a sentence first."}
                    return explain_text(text, target, chosen, situation)

                restate.click(restate_translation, [baseline, adapted, tgt, level, before],
                              [restated, restated_detail], api_name="restate",
                              concurrency_id="assistant", concurrency_limit=2)
                ask.click(look_up, [question, question_language, place, before],
                          [answer, answer_detail, answer_sources], api_name="look_up",
                          concurrency_id="assistant", concurrency_limit=2)
                repeat.click(say_answer_in, [answer, answer_language], [repeated, answer_detail],
                             api_name="say_answer_in", concurrency_id="assistant", concurrency_limit=2)
    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-emotion", action="store_true", help="Skip local emotion analysis.")
    parser.add_argument("--no-assistant", action="store_true",
                        help="Skip the optional Claude restating and lookup features.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    speech_models = parser.add_mutually_exclusive_group()
    speech_models.add_argument("--fast-whisper", action="store_true", help="Use cached Whisper Base on Apple Silicon.")
    speech_models.add_argument("--whisper-turbo", action="store_true", help="Use cached Whisper Large V3 Turbo on Apple Silicon.")
    parser.add_argument("--fast-nllb", type=Path, help="Directory of the converted CTranslate2 NLLB model.")
    parser.add_argument("--warm", action="store_true", help="Load live models at startup.")
    parser.add_argument("--native-voice", help="Path to the compiled fast macOS voice renderer.")
    parser.add_argument("--voice-engine", help="Python executable for the isolated MLX matching engine.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open-browser", action="store_true", help="Open the app in the default browser.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. See PUBLISH.md for hosted access.")
    args = parser.parse_args()
    auth = None
    hosted = args.host not in ("127.0.0.1", "localhost", "::1")
    public_review = os.environ.get("POLYGLOT_PUBLIC_REVIEW") == "1"
    if hosted and not public_review:
        password = os.environ.get("POLYGLOT_ACCESS_PASSWORD", "")
        if len(password) < 16:
            parser.error("Hosting requires POLYGLOT_ACCESS_PASSWORD of at least 16 characters.")
        auth = ("partner", password)
    # A hosted instance makes no outbound model calls unless the operator opts in,
    # because every visitor would otherwise spend the operator's credentials.
    # A credential in the environment is taken as intent to offer the features,
    # hosted included. POLYGLOT_HOSTED_ASSISTANT=0 serves translation only.
    hosted_assistant = hosted and os.environ.get("POLYGLOT_HOSTED_ASSISTANT") != "0"
    if hosted_assistant:
        if not assistant_available():
            hosted_assistant = False
        elif public_review:
            print("Warning: the restating and lookup features are enabled for anonymous visitors. "
                  "Every visitor spends this deployment's Anthropic credentials. Remove "
                  "POLYGLOT_PUBLIC_REVIEW=1 to keep the password gate, or unset "
                  "POLYGLOT_HOSTED_ASSISTANT to serve translation only.", flush=True)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    ui = build_app(args.demo, args.device, not args.no_emotion, allow_remote=not hosted,
                   fast_whisper=args.fast_whisper, whisper_turbo=args.whisper_turbo,
                   fast_nllb=args.fast_nllb, clone_executable=args.voice_engine,
                   native_voice_executable=args.native_voice, warm=args.warm,
                   allow_assistant=not args.no_assistant and (not hosted or hosted_assistant)
                   ).queue(max_size=16)
    try:
        serve(ui, args.port, server_name=args.host, share=False, auth=auth, pwa=True,
              inbrowser=args.open_browser, max_file_size=MAX_AUDIO_BYTES,
              favicon_path=Path(__file__).parent / "pragmatics" / "icon.svg")
    except OSError:
        raise SystemExit(
            f"Ports {args.port} to {args.port + PORT_SEARCH - 1} are all in use. Another copy of "
            f"Click is probably still running: close its terminal window, or stop it with "
            f"`lsof -ti:{args.port} | xargs kill`, then start Click again."
        ) from None
