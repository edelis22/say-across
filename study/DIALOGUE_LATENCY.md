# Dialogue latency pilot

ElliQ’s user guide describes translation for words and short phrases, and its FAQ describes a conversational back-and-forth. The design goal here is a shorter gap between speakers while preserving the completed utterance. [ElliQ companion guide](https://tips.elliq.com/) · [ElliQ FAQ](https://elliq.com/pages/faqs)

Click prepares eligible local turns of up to eight seconds after the first quarter second of detected quiet. It retains the same audio used by the normal turn path and holds captions and playback until the configured pause confirms the turn. Resumed speech cancels the pending result. An utterance gets at most one early attempt. Turns with remote context adaptation or competing work use the normal path.

## Browser timing

An isolated Chrome browser fed four generated test turns into the microphone capture path on a 16 GB Apple M2 MacBook Air. Both runs used the same Whisper Turbo and int8 NLLB checkpoints, expressive system playback, and the default 0.75-second pause. Timers start at the last voiced sample and end when the final caption enters the conversation or audio starts playing. These include the quiet interval, application processing, and local delivery.

| Generated test turn | Caption before | Caption after | Audio before | Audio after |
| --- | ---: | ---: | ---: | ---: |
| Greek thanks | 2.02 s | 1.46 s | 2.07 s | 1.72 s |
| English reply | 2.26 s | 1.72 s | 2.31 s | 1.88 s |
| Greek thanks repeated later | 2.02 s | 1.49 s | 2.08 s | 1.49 s |
| Greek sentence with a hesitation | 2.43 s | 2.36 s | 2.48 s | 2.52 s |

The three turns with uninterrupted endings had a median caption delay of **2.02 → 1.49 seconds** and median playback delay of **2.08 → 1.72 seconds**. The hesitation test canceled the early request and produced a single final translation containing the rest of the sentence. All four final translations matched their earlier wording. Matched-voice synthesis remains a separate stage. On six local matched-voice test requests, it added roughly 1.0–2.6 seconds after text was ready.

## CPU profile and limits

The portable profile now shares Whisper’s encoder output between language detection and short-turn transcription. Six warmed CPU checks on the same three generated clips produced identical transcripts and directions, with median recognition time of **559 → 375 ms** on this Mac. The experiment measures computation in the local CPU environment. Hosted latency depends on hardware, model warmup, load, and the network.

A reduced Whisper encoder window was faster and caused repeated words or phrases on several test clips. The shipped path uses the full recognition window. The browser check tests generated Greek–English dialogue, one hesitation, and one machine. Measure latency percentiles, premature turn endings, stale captions, and playback overlap on consented spontaneous dialogues before generalizing these results.
