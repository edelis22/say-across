# Speech-to-English pilot

In the reported microphone turn, **Ευχαριστώ πάρα πολύ.** means **Thank you very much.** The displayed transcript was **Ευχαριστώ πάρω πολύ.** Greedy NLLB reproduced the displayed English output, “Thank you. I'm very much in love.” It also repeated the thanks when given the clean Greek sentence. Four-beam decoding returned “Thank you very much, sir.” on the clean sentence and supplied a different erroneous clause on the misheard sentence.

The live app now uses the stronger Whisper Large V3 Turbo model on the configured Mac. It shares the audio encoding for language identification and transcription on turns up to 30 seconds. On complete, reviewed expressions of thanks where NLLB added or repeated meaning, captions and playback use a fixed translation. The source caption preserves the actual ASR transcript. Longer sentences with the same opening go through the normal translation path.

## Method

The pilot sampled the FLEURS test split at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. For each language, select the first eligible unique sentence IDs in audio archive order. Eligible recordings decode as speech with the app’s reader, run for 2–20 seconds, and have a matching English test sentence ID. This gives 62 human recordings in the 11 non-English languages, including 12 in Greek, plus 8 English controls. The actual durations are 4.38–17.94 seconds. [Recording manifest](data/inbound-fleurs.csv) · [Pinned FLEURS release](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) · [FLEURS paper](https://research.google/pubs/fleurs-few-shot-learning-evaluation-of-universal-representations-of-speech/)

Test pair selection separately from transcription. For each non-English clip, the model chooses between the expected language and English, with the app’s existing abstention thresholds. English controls use the Greek–English pair. Compute transcription character error with the expected source language supplied, independent of the pair selection. Case-fold and apply Unicode NFKC, retain letters, combining marks and numbers, then divide summed Levenshtein edits by summed reference characters. Differences in numeral representation and Chinese orthography can count as character errors.

The comparison uses `mlx-whisper==0.4.3` on a 16 GB Apple M2 MacBook Air. The original profile is Whisper Base MLX at `1e3e249fb8d01c655324bd6841b1deadffd6d04c` with separate language detection and transcription. The new profile is the 4-bit Turbo checkpoint at `0f058d38170d183f9fdee07908f5b515d91793a8` with shared encoding and greedy decoding. Whisper Small 8-bit at `c2f1b4b958f9037b53ef14b467a77208e23c10d3` and full Whisper Medium 4-bit at `c851cec2642c6e314c56a9fa33acf054ca16ccf6` provide additional local comparisons. The Turbo release is intended for multilingual transcription. Whisper’s full checkpoints support a separate speech-to-English task. [Whisper documentation](https://github.com/openai/whisper#command-line-usage) · [Turbo release notes](https://github.com/openai/whisper/discussions/2363)

## Transcription and direction

| Human recordings | Count | Base character error | Turbo character error | Base + NLLB chrF | Turbo + NLLB chrF |
| --- | ---: | ---: | ---: | ---: | ---: |
| All non-English | 62 | 13.6% | 3.6% | 44.7 | 55.8 |
| Arabic | 5 | 10.5% | 4.8% | 43.6 | 54.6 |
| Dutch | 5 | 11.5% | 1.3% | 40.6 | 49.4 |
| French | 5 | 20.8% | 7.6% | 41.9 | 58.6 |
| German | 5 | 3.3% | 0.3% | 62.9 | 69.4 |
| Greek | 12 | 20.0% | 5.1% | 40.9 | 60.9 |
| Italian | 5 | 5.0% | 0.2% | 49.7 | 57.3 |
| Japanese | 5 | 29.6% | 8.4% | 30.5 | 44.9 |
| Korean | 5 | 18.8% | 4.4% | 37.8 | 48.1 |
| Mandarin Chinese | 5 | 23.1% | 5.1% | 42.5 | 46.2 |
| Portuguese | 5 | 9.2% | 3.3% | 57.6 | 66.0 |
| Spanish | 5 | 3.7% | 0.8% | 48.5 | 50.5 |
| English controls | 8 | 5.2% | 2.9% | — | — |

Whisper Small and Medium scored 6.8% and 5.2% character error on the non-English clips. Base routed 61 of 62 non-English clips correctly and abstained on one Greek clip. Small, Medium and Turbo each routed all 62 correctly.

A separate set of 28 generated Mac-voice turns stresses short phrases. It uses two gratitude expressions per language and four extra Greek examples covering a one-word thanks, a greeting, a negation, and a contrast between 27 and 72. Base routed 25 correctly, guessed the wrong language once, and abstained twice. Small routed 25 correctly, guessed the wrong language twice, and abstained once. Medium routed 27 correctly and abstained once. Turbo routed all 28 correctly. The short Greek regression transcript was **ευχαριστώ παραπολή.** with Base and **Ευχαριστώ πάρα πολύ.** with Turbo. Median warmed recognition on those short fixtures was 0.14 seconds for Base and 1.07 seconds for Turbo, including the paired language check. These times exclude the quiet interval, translation and voice synthesis.

## Translation comparison

The table uses corpus chrF from SacreBLEU 2.5.1 with its default character 6-gram and beta 2 settings. chrF measures similarity to the English reference on a 0–100 scale. Both cascades use the expected source language and the same NLLB distilled 600M checkpoint with int8 weights and greedy decoding. The pilot batches sentences with a limit of 128 output tokens. One Japanese translation from a Base transcript reached that limit and was scored as empty. The phrase catalogue and voice synthesis are outside this comparison. NLLB’s model card describes its research scope and CC BY-NC 4.0 license. [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M)

| Speech-to-English path | Overall chrF | Greek chrF |
| --- | ---: | ---: |
| Whisper Base → greedy NLLB | 44.7 | 40.9 |
| Full Whisper Medium direct translation | 51.2 | 52.9 |
| Whisper Turbo → TranslateGemma 4B | 53.2 | 58.6 |
| Whisper Turbo → greedy NLLB | 55.8 | 60.9 |

TranslateGemma used the same Turbo transcripts, its published translation prompt, greedy generation, and the end-of-turn stop token. The local 4-bit checkpoint was `5788ec08c047f3f2e17808101b8d9566ac930d58` with `mlx-lm==0.31.3`. The deployed profile keeps NLLB based on this pilot, with a source transcript available for inspection and one translation used for both captions and playback. [TranslateGemma release](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) · [Pinned 4-bit checkpoint](https://huggingface.co/mlx-community/translategemma-4b-it-4bit/tree/5788ec08c047f3f2e17808101b8d9566ac930d58)

Supplying the reference transcripts to NLLB raised chrF to 59.6 overall and 65.9 on Greek with greedy decoding. Four-beam decoding scored 60.5 overall and 64.7 on Greek. The clean transcript comparison estimates how much recognition still contributes to this cascade's errors.

The browser check covered automatic Greek–English direction changes, the reported Greek thanks with matched-voice playback, and Spanish thanks with expressive playback. A separate endpoint check preserved Greek negation and the distinction between 27 and 72 in short turns. It also exercised a 40-second concatenated recording with content from the first and last sentences appearing in the transcript.

## Review limits

FLEURS is read speech with a small, deterministic sample per language. Generated phrases test specific failures under controlled conditions. Testing spontaneous turns, accents, room noise, code-switching and emotional fidelity requires conversational recordings. Treat character error and reference similarity as diagnostics. For the next comparative study, collect consented conversational turns, use blinded bilingual raters, and score changed names, numbers, negation, additions, omissions, and tone separately. Include Google Translate and the frozen NLLB baseline in the same evaluation, and report the distribution of time to caption and time to speech. See the [study protocol](PROTOCOL.md).
