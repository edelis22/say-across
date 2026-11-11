() => {
    const initialize = () => {
        const panel = document.querySelector("#live-voice");
        if (!panel) return false;
        if (panel.dataset.ready) return true;
        panel.dataset.ready = "yes";
        const control = name => panel.querySelector(`[data-live="${name}"]`);
        const start = control("start");
        const stop = control("stop");
        const source = control("source");
        const target = control("target");
        const swap = control("swap");
        const pause = control("pause");
        const voice = control("voice");
        const profileLanguage = control("profile-language");
        const profileStatus = control("profile-status");
        const profileRecord = control("profile-record");
        const profileCancel = control("profile-cancel");
        const profileClear = control("profile-clear");
        const profileProgress = control("profile-progress");
        const profilePrompt = control("profile-prompt");
        const twoWay = control("two-way");
        const level = control("level");
        const voiceRange = control("voice-range");
        const question = control("question");
        const place = control("place");
        const ask = control("ask");
        const askClear = control("ask-clear");
        const askStatus = control("ask-status");
        const askAnswerGroup = control("ask-answer-group");
        const askAnswer = control("ask-answer");
        const askSources = control("ask-sources");
        const answerLanguage = control("answer-language");
        const answerRepeat = control("answer-repeat");
        const answerSpeak = control("answer-speak");
        const answerStatus = control("answer-status");
        const answerRepeated = control("answer-repeated");
        const answerRepeatedGroup = control("answer-repeated-group");
        const sampleRate = 16000;
        const maxSamples = sampleRate * Number(panel.dataset.maxSeconds);
        const captionSamples = sampleRate * Number(panel.dataset.captionSeconds);
        const firstCaption = Math.round(2.4 * sampleRate);
        const captionToggle = control("caption-toggle");
        const captionScreen = control("caption-screen");
        const captionState = control("caption-state");
        const captionEmpty = control("caption-empty");
        const captionLines = control("caption-lines");
        const status = control("status");
        const indicator = control("indicator");
        const progress = control("progress");
        const empty = control("empty");
        const labels = {ready: "Ready", listening: "Mic on", waiting: "Mic on", playing: "Mic paused",
                        working: "Translating", opening: "Connecting", error: "Check mic"};
        const player = control("player");
        const history = control("history");
        const endpoint = new URL("gradio_api/call/live_phrase", document.baseURI);
        const draftEndpoint = new URL("gradio_api/call/live_caption", document.baseURI);
        const lookupEndpoint = new URL("gradio_api/call/live_lookup", document.baseURI);
        const explainEndpoint = new URL("gradio_api/call/live_explain", document.baseURI);
        const answerLanguageEndpoint = new URL("gradio_api/call/live_answer_language", document.baseURI);
        const answerAudioEndpoint = new URL("gradio_api/call/live_answer_audio", document.baseURI);
        const canExplain = panel.querySelector(".say-assistant")?.hidden === false;
        const bandNames = {lower: "lower pitch", higher: "higher pitch"};
        let session, cancelProfile;
        let profileSample = "", profileRecording = false;

        function show(state, message, fraction) {
            panel.dataset.state = state;
            indicator.lastChild.nodeValue = labels[state];
            if (status.textContent !== message) status.textContent = message;
            progress.hidden = fraction === undefined;
            if (fraction !== undefined) progress.value = fraction;
        }

        function encode(chunks) {
            const bytes = new Uint8Array(2 * chunks.reduce((n, chunk) => n + chunk.length, 0));
            const view = new DataView(bytes.buffer);
            let offset = 0;
            for (const chunk of chunks) {
                for (const sample of chunk) {
                    view.setInt16(offset, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true);
                    offset += 2;
                }
            }
            let raw = "";
            for (let i = 0; i < bytes.length; i += 8192) {
                raw += String.fromCharCode(...bytes.subarray(i, i + 8192));
            }
            return btoa(raw);
        }

        function rms(chunk) {
            return Math.sqrt(chunk.reduce((sum, sample) => sum + sample * sample, 0) / chunk.length);
        }

        async function openMicrophone() {
            const context = new AudioContext({sampleRate});
            let media;
            try {
                await context.resume();
                media = await navigator.mediaDevices.getUserMedia({
                    audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1},
                });
                const node = context.createScriptProcessor(1024, 1, 1);
                context.createMediaStreamSource(media).connect(node);
                node.connect(context.destination);
                return {context, media, node};
            } catch (error) {
                media?.getTracks().forEach(track => track.stop());
                void context.close();
                throw error;
            }
        }

        function closeMicrophone({context, media, node}) {
            node.onaudioprocess = null;
            node.disconnect();
            media.getTracks().forEach(track => track.stop());
            void context.close();
        }

        async function requestLive(route, data, onText, timeout = 240000, cancel) {
            const deadline = AbortSignal.timeout(timeout);
            const options = {signal: cancel ? AbortSignal.any([deadline, cancel]) : deadline};
            const response = await fetch(route, {
                ...options, method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({data}),
            });
            if (!response.ok) throw new Error("Could not send the phrase.");
            const {event_id: id} = await response.json();
            if (!id) throw new Error("The translation queue is unavailable.");
            const stream = await fetch(`${route}/${id}`, options);
            if (!stream.ok) throw new Error("Could not receive the translation.");
            const reader = stream.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            try {
                for (;;) {
                    const {value, done} = await reader.read();
                    if (done) throw new Error("The phrase could not be translated.");
                    buffer += decoder.decode(value, {stream: true});
                    const events = buffer.split(/\r?\n\r?\n/);
                    buffer = events.pop();
                    for (const event of events) {
                        const line = event.split("\n").find(row => row.startsWith("data: "));
                        if (!line) continue;
                        if (event.startsWith("event: error")) throw new Error("The phrase could not be translated.");
                        if (event.startsWith("event: generating")) {
                            const [result] = JSON.parse(line.slice(6));
                            if (result.stage === "translated") onText?.(result);
                        } else if (event.startsWith("event: complete")) {
                            return JSON.parse(line.slice(6))[0];
                        }
                    }
                }
            } finally {
                await reader.cancel().catch(() => {});
            }
        }

        function setText(element, text, language) {
            element.textContent = text;
            element.lang = language;
            element.dir = "auto";
        }

        function fillCues(list, cues, language) {
            list.replaceChildren(...cues.map(cue => {
                const item = document.createElement("li");
                item.dataset.cue = cue.kind;
                const time = document.createElement("time");
                time.dateTime = `PT${cue.second}S`;
                time.textContent = `${Math.floor(cue.second / 60)}:${(cue.second % 60).toFixed(1).padStart(4, "0")}`;
                const label = document.createElement("span");
                setText(label, cue.text, language);
                item.append(time, label);
                return item;
            }));
        }

        function fillGlossary(list, glossary, language) {
            list.replaceChildren(...glossary.map(entry => {
                const row = document.createElement("li");
                const term = document.createElement("strong");
                setText(term, entry.term, language);
                const meaning = document.createElement("span");
                setText(meaning, entry.meaning, language);
                row.append(term, meaning);
                return row;
            }));
        }

        function explanationNote(explanation) {
            return [explanation.check_numbers ? "Check the numbers against the original." : "",
                    explanation.omitted || ""].filter(Boolean).join(" ");
        }

        function paintExplanation(result, to) {
            const explanation = result.explanation || {};
            const text = explanation.rendered || "";
            control("caption-explained").hidden = !text && !explanation.error;
            control("caption-explained-label").textContent = explanation.level_label
                ? `Simpler wording · ${explanation.level_label}` : "Simpler wording";
            setText(control("caption-explained-text"), text || explanation.error || "",
                    text ? to.tag : "en");
            fillGlossary(control("caption-glossary"), text ? explanation.glossary || [] : [], to.tag);
            control("caption-explained-note").textContent = text ? explanationNote(explanation) : "";
        }

        function explainButton(item, result, to) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "say-end say-explain";
            button.textContent = "Explain simply";
            button.addEventListener("click", async () => {
                button.disabled = true;
                button.textContent = "Restating\u2026";
                // Without a standing level, an on-demand request uses plain language.
                const chosen = level.value === "as_spoken" ? "plain" : level.value;
                try {
                    const explanation = await requestLive(explainEndpoint,
                        [result.translation, result.target_language, chosen,
                            control("situation").value], null, 90000);
                    button.remove();
                    appendExplanation(item, {...result, explanation}, to);
                } catch (error) {
                    button.disabled = false;
                    button.textContent = "Explain simply";
                    voiceNote(item, error.message);
                }
            });
            item.append(button);
        }

        function appendExplanation(item, result, to) {
            const explanation = result.explanation || {};
            if (!item || item.querySelector(".say-turn-explained")) return;
            if (!explanation.rendered) {
                voiceNote(item, explanation.error);
                return;
            }
            const restated = document.createElement("span");
            restated.className = "say-turn-explained";
            setText(restated, explanation.rendered, to.tag);
            item.append(restated);
            if (explanation.glossary?.length) {
                const glossary = document.createElement("ul");
                glossary.className = "say-glossary";
                glossary.setAttribute("role", "list");
                glossary.setAttribute("aria-label", "Terms explained");
                fillGlossary(glossary, explanation.glossary, to.tag);
                item.append(glossary);
            }
            voiceNote(item, explanationNote(explanation));
        }

        function direction(current, result) {
            return [current.languages[result.source_language || current.source],
                    current.languages[result.target_language || current.target]];
        }

        function paintCaption(current, {result, draft}, announce) {
            const original = result.transcript || "";
            const translated = result.translation || "";
            const [from, to] = direction(current, result);
            const same = from.tag === to.tag;
            captionEmpty.hidden = true;
            captionLines.hidden = false;
            control("caption-original").hidden = !original || (same && Boolean(translated));
            control("caption-translated").hidden = !translated;
            control("caption-source-label").textContent = `Original · ${from.name}`;
            control("caption-target-label").textContent = `${same ? "Transcript" : "Translation"} · ${to.name}`;
            setText(control("caption-source"), original, from.tag);
            setText(control("caption-target"), translated, to.tag);
            const error = control("caption-error");
            error.textContent = result.error || "";
            error.hidden = !result.error;
            paintExplanation(result, to);
            const cues = result.cues || [];
            control("caption-cue-group").hidden = !cues.length;
            fillCues(control("caption-cues"), cues, to.tag);
            captionState.textContent = draft ? "Draft · recent speech" : result.error ? "Caption notice" : "Final";
            captionScreen.scrollTop = 0;
            if (!draft && announce) {
                setText(control("caption-announce"), translated || original || result.error || "",
                        translated ? to.tag : original ? from.tag : "en");
            }
        }

        function showCaption(current, turn, result, draft) {
            if (session !== current || turn < (current.lastCaption?.turn ?? -1)) return;
            current.lastCaption = {turn, result, draft};
            if (captionToggle.checked) paintCaption(current, current.lastCaption, true);
        }

        function recentAudio(chunks) {
            const recent = [];
            let remaining = captionSamples;
            for (let i = chunks.length - 1; i >= 0 && remaining > 0; i--) {
                const chunk = chunks[i];
                recent.push(chunk.subarray(Math.max(0, chunk.length - remaining)));
                remaining -= chunk.length;
            }
            return recent.reverse();
        }

        async function updateDraft(current) {
            if (!captionToggle.checked || current.draftBusy || current.length < current.nextCaption ||
                current.speech < sampleRate * 0.75) return;
            const turn = current.turn;
            current.nextCaption = current.length + 3 * sampleRate;
            current.draftBusy = true;
            try {
                const result = await requestLive(draftEndpoint,
                    [encode(recentAudio(current.chunks)), current.source, current.target,
                        JSON.stringify({two_way: current.twoWay})], null, 15000);
                if (current.listening && current.turn === turn && captionToggle.checked && result.transcript) {
                    showCaption(current, turn, result, true);
                }
            } catch {
                // Final captions are handled independently when the turn ends.
            } finally {
                current.draftBusy = false;
            }
        }

        function playNext(current) {
            if (current.playing || !current.sounds.length || session !== current ||
                (current.twoWay && current.chunks.length)) return;
            current.playing = true;
            player.hidden = false;
            player.src = `data:audio/wav;base64,${current.sounds.shift()}`;
            player.play().catch(() => {
                current.playing = false;
                show("ready", "Translation ready. Press play to hear it.");
            });
        }

        function append(result, current, turn) {
            const item = document.createElement("li");
            item.className = "say-turn";
            item.dataset.turn = turn;
            const [from, to] = direction(current, result);
            if (result.translation && from.tag !== to.tag) voiceNote(item, `${from.name} → ${to.name}`);
            if (result.transcript && (from.tag !== to.tag || !result.translation)) {
                const original = document.createElement("span");
                original.className = "say-turn-source";
                setText(original, result.transcript, from.tag);
                item.append(original);
            }
            const translated = document.createElement("strong");
            translated.className = "say-turn-target";
            setText(translated, result.translation || result.error || "", result.translation ? to.tag : "en");
            item.append(translated);
            if (result.cues?.length) {
                const cues = document.createElement("ul");
                cues.className = "say-cues";
                cues.setAttribute("aria-label", "Vocal cues from the original");
                cues.setAttribute("role", "list");
                fillCues(cues, result.cues, to.tag);
                item.append(cues);
            }
            voiceNote(item, result.translation && result.emotion);
            voiceNote(item, result.context_note);
            if (result.tone_label && result.tone !== "auto" && result.voice_mode !== "captions") {
                voiceNote(item, `Spoken tone · ${result.tone_label}`);
            }
            if (bandNames[result.voice_band] && result.voice_mode === "expressive") {
                voiceNote(item, `Voice range · ${bandNames[result.voice_band]}`);
            }
            empty.hidden = true;
            const later = [...history.children].find(other => Number(other.dataset.turn) > turn);
            history.insertBefore(item, later || null);
            history.scrollTop = history.scrollHeight;
            if (history.children.length > 20) history.firstElementChild.remove();
            return item;
        }

        function voiceNote(item, text) {
            if (!text) return;
            const note = document.createElement("span");
            note.className = "say-turn-note";
            note.textContent = text;
            item.append(note);
        }

        function unlock(current) {
            if (current.listening || current.inflight || current.pending.length) return;
            start.disabled = source.disabled = target.disabled = swap.disabled = pause.disabled = twoWay.disabled = false;
            syncPlayback();
            stop.hidden = true;
            if (!current.lastCaption) {
                captionEmpty.textContent = "Captions appear here when the microphone is on.";
                if (captionToggle.checked) captionState.textContent = "Ready";
            }
            show(current.problem ? "error" : "ready", current.problem || "Microphone off");
        }

        function releaseAudio(current, turn, audio) {
            current.completed.set(turn, audio);
            while (current.playOrder.length && current.completed.has(current.playOrder[0])) {
                const ready = current.playOrder.shift();
                const clip = current.completed.get(ready);
                current.completed.delete(ready);
                if (clip) current.sounds.push(clip);
                if (current.sounds.length > 8) {
                    current.problem = "Audio playback fell behind. Start again after listening.";
                    stopCapture(current);
                    break;
                }
            }
            playNext(current);
        }

        function submitTurn(current, chunks, options, onText, cancel) {
            return requestLive(endpoint,
                [encode(chunks), current.source, current.target, current.voice, options,
                    current.profileLanguage ? profileSample : ""], onText, 240000, cancel);
        }

        async function translateTurn(current, {chunks, turn, options, prepared}) {
            let item, audio;
            const showText = result => {
                item ||= append(result, current, turn);
                showCaption(current, turn, result, false);
            };
            try {
                let result;
                if (prepared) {
                    prepared.showText = showText;
                    if (prepared.preview) showText(prepared.preview);
                    result = await prepared.result;
                }
                if (!result) result = await submitTurn(current, chunks, options, showText);
                if (!item) showText(result);
                voiceNote(item, result.playback_error || result.voice_engine);
                const [, to] = direction(current, result);
                if (result.explanation && Object.keys(result.explanation).length) {
                    appendExplanation(item, result, to);
                    if (current.lastCaption?.turn === turn) {
                        current.lastCaption.result = {...current.lastCaption.result,
                                                      explanation: result.explanation};
                        if (captionToggle.checked) paintCaption(current, current.lastCaption, false);
                    }
                } else if (canExplain && item && result.translation &&
                           result.target_language !== result.source_language) {
                    explainButton(item, result, to);
                }
                audio = result.audio;
            } catch (error) {
                if (item) voiceNote(item, error.message);
                else showText({error: error.message});
            } finally {
                releaseAudio(current, turn, audio);
            }
        }

        function translatePending(current) {
            while (current.inflight < 2 && current.pending.length) {
                current.inflight++;
                show("working", current.listening ? "Listening and translating…" : "Finishing translation…");
                void translateTurn(current, current.pending.shift()).finally(() => {
                    current.inflight--;
                    translatePending(current);
                    if (!current.inflight && current.listening && !(current.twoWay && current.playing)) {
                        show("listening", "Listening for your next turn…");
                    }
                    unlock(current);
                });
            }
        }

        function trimmedChunks(current) {
            const chunks = current.chunks.slice();
            let trim = Math.max(0, current.bufferedQuiet - sampleRate / 4);
            while (chunks.length && trim >= chunks.at(-1).length) trim -= chunks.pop().length;
            if (trim && chunks.length) chunks[chunks.length - 1] = chunks.at(-1).slice(0, -trim);
            return chunks;
        }

        function turnSettings(current) {
            return {two_way: current.twoWay, tone: control("tone").value,
                relationship: control("relationship").value, situation: control("situation").value,
                remote: control("remote")?.checked || false, profile_language: current.profileLanguage,
                level: level.value, voice_range: voiceRange.value};
        }

        function prepareTurn(current) {
            if (!AbortSignal.any || current.triedPreparation || current.inflight || current.pending.length ||
                current.draftBusy || current.speech < 3200 || current.bufferedQuiet < sampleRate / 4 ||
                current.length > 8 * sampleRate) return;
            const settings = turnSettings(current);
            if (settings.remote) return;
            current.triedPreparation = true;
            const prepared = {controller: new AbortController(), options: JSON.stringify(settings)};
            current.prepared = prepared;
            // The final turn retains the same quarter second of quiet. Hold output until the pause confirms it.
            prepared.result = submitTurn(current, trimmedChunks(current), prepared.options, preview => {
                prepared.preview = preview;
                prepared.showText?.(preview);
            }, prepared.controller.signal).catch(() => null);
        }

        function endPhrase(current) {
            const chunks = trimmedChunks(current);
            const turn = current.turn++;
            const speech = current.speech;
            const options = JSON.stringify(turnSettings(current));
            const prepared = current.prepared;
            current.prepared = null;
            current.triedPreparation = false;
            current.chunks = [];
            current.length = current.quiet = current.bufferedQuiet = current.speech = 0;
            current.nextCaption = firstCaption;
            if (speech < 3200) {
                prepared?.controller.abort();
                playNext(current);
                return;
            }
            if (current.pending.length >= 6) {
                prepared?.controller.abort();
                current.problem = "Translation fell behind. Start again and speak in shorter bursts.";
                stopCapture(current);
                return;
            }
            if (prepared && prepared.options !== options) prepared.controller.abort();
            current.playOrder.push(turn);
            current.pending.push({chunks, turn, options, prepared: prepared?.options === options ? prepared : null});
            translatePending(current);
            playNext(current);
        }

        function accept(current, chunk) {
            const energy = rms(chunk);
            const levels = current.energyHistory;
            levels.push(energy);
            if (levels.length > 24) levels.shift();
            // Track quiet room noise without letting loud speech raise the gate indefinitely.
            const sorted = [...levels].sort((a, b) => a - b);
            const floor = sorted[Math.floor((sorted.length - 1) * 0.2)];
            const threshold = levels.length < 8 ? 0.008 : Math.max(0.005, Math.min(0.025, 2.2 * floor));
            const active = energy > threshold;
            if (!current.chunks.length && !active) {
                current.prefix = [...current.prefix.slice(-1), chunk];
                return;
            }
            if (!current.chunks.length) {
                current.chunks.push(...current.prefix);
                current.length = current.prefix.reduce((size, frame) => size + frame.length, 0);
                current.prefix = [];
            }
            const remaining = maxSamples - current.length;
            const frame = chunk.length <= remaining ? chunk : chunk.subarray(0, remaining);
            current.chunks.push(frame);
            current.length += frame.length;
            current.quiet = active ? 0 : current.quiet + chunk.length;
            current.bufferedQuiet = active ? 0 : current.bufferedQuiet + frame.length;
            if (active) {
                current.speech += frame.length;
                current.prepared?.controller.abort();
                current.prepared = null;
            }
            if (current.quiet >= current.pauseSamples) {
                endPhrase(current);
            } else if (active && current.length >= maxSamples) {
                current.problem = "Two-minute turn limit reached. Enable the microphone to continue.";
                stopCapture(current);
            } else {
                if (active) void updateDraft(current);
                else prepareTurn(current);
                if (!current.inflight) {
                    const seconds = Math.ceil((current.pauseSamples - current.quiet) / sampleRate);
                    if (active) show("listening", "Listening…");
                    else show("waiting", `Finishing turn in ${seconds} s`, current.quiet / current.pauseSamples);
                }
            }
        }

        function stopCapture(current) {
            if (!current?.listening) return;
            current.listening = false;
            stop.disabled = true;
            show("working", "Finishing the last phrase…");
            closeMicrophone(current);
            endPhrase(current);
            unlock(current);
        }

        function syncPlayback() {
            const same = source.value === target.value;
            voice.disabled = source.disabled || same;
            control("same-language").hidden = !same;
            control("source-label").textContent = twoWay.checked && !same ? "First language" : "Speaking";
            control("target-label").textContent = twoWay.checked && !same ? "Second language" : "Read or hear in";
            profileLanguage.disabled = profileRecord.disabled = source.disabled || profileRecording;
            profileClear.disabled = profileRecord.disabled || !profileSample;
        }

        function describeProfile() {
            profileStatus.textContent = profileSample
                ? `Sample ready for ${profileLanguage.selectedOptions[0].textContent}.`
                : "Ready to record a voice sample.";
        }

        profileRecord.addEventListener("click", async () => {
            if (start.disabled) return;
            start.disabled = profileRecording = true;
            syncPlayback();
            profileStatus.textContent = "Opening the microphone…";
            let mic, timer;
            try {
                mic = await openMicrophone();
                const seconds = Number(panel.dataset.profileSeconds);
                const limit = sampleRate * seconds;
                const chunks = [];
                let length = 0;
                profileStatus.textContent = "Recording your sample…";
                profileProgress.value = 0;
                profileProgress.hidden = profilePrompt.hidden = profileCancel.hidden = false;
                await new Promise((resolve, reject) => {
                    cancelProfile = () => reject(new Error("Recording canceled."));
                    timer = setTimeout(() => reject(new Error("The microphone stopped sending audio. Try again.")),
                        (seconds + 5) * 1000);
                    mic.media.getAudioTracks()[0].addEventListener("ended", () => {
                        reject(new Error("The microphone stopped recording."));
                    }, {once: true});
                    mic.node.onaudioprocess = ({inputBuffer}) => {
                        const chunk = inputBuffer.getChannelData(0).slice(0, limit - length);
                        chunks.push(chunk);
                        length += chunk.length;
                        profileProgress.value = length / limit;
                        if (length === limit) resolve();
                    };
                });
                const levels = chunks.map(rms);
                const floor = Math.max(0.004, Math.max(...levels) * 0.08);
                const speech = chunks.reduce((count, chunk, index) => count + (levels[index] >= floor ? chunk.length : 0), 0);
                if (speech < sampleRate * Number(panel.dataset.profileSpeechSeconds)) {
                    throw new Error("Speak for at least three seconds and try recording again.");
                }
                profileSample = encode(chunks);
                voice.value = "match";
                describeProfile();
            } catch (error) {
                profileStatus.textContent = error.message;
            } finally {
                clearTimeout(timer);
                if (mic) closeMicrophone(mic);
                cancelProfile = null;
                profileProgress.hidden = profilePrompt.hidden = profileCancel.hidden = true;
                start.disabled = profileRecording = false;
                syncPlayback();
            }
        });
        profileCancel.addEventListener("click", () => cancelProfile?.());
        profileClear.addEventListener("click", () => {
            profileSample = "";
            describeProfile();
            syncPlayback();
        });
        profileLanguage.addEventListener("change", describeProfile);

        function sourceLink(source) {
            const row = document.createElement("li");
            let safe = null;
            try {
                const parsed = new URL(source.url);
                if (parsed.protocol === "https:" || parsed.protocol === "http:") safe = parsed;
            } catch {
                safe = null;
            }
            if (!safe) {
                row.textContent = source.title || "";
                return row;
            }
            const link = document.createElement("a");
            link.href = safe.href;
            link.textContent = source.title || safe.hostname;
            link.title = safe.href;
            link.rel = "noreferrer noopener";
            link.target = "_blank";
            row.append(link);
            return row;
        }

        // The answer currently on screen, with the language it is written in.
        let currentAnswer = null;

        function resetRepeat() {
            answerRepeatedGroup.hidden = true;
            answerRepeated.textContent = "";
            answerStatus.textContent = "";
        }

        async function repeatAnswer() {
            if (!currentAnswer) return;
            const option = answerLanguage.selectedOptions[0];
            answerRepeat.disabled = answerSpeak.disabled = true;
            answerStatus.textContent = `Restating in ${option.textContent}\u2026`;
            try {
                const result = await requestLive(answerLanguageEndpoint,
                    [currentAnswer.text, answerLanguage.value], null, 120000);
                if (result.error) {
                    answerStatus.textContent = result.error;
                } else {
                    control("answer-repeated-label").textContent =
                        `The same answer \u00b7 ${option.textContent}`;
                    setText(answerRepeated, result.answer, option.dataset.lang);
                    answerRepeatedGroup.hidden = false;
                    answerStatus.textContent = "";
                    // Reading aloud now follows the language on screen.
                    currentAnswer = {...currentAnswer, spoken: {text: result.answer,
                                                                language: answerLanguage.value,
                                                                label: option.textContent}};
                }
            } catch (error) {
                answerStatus.textContent = error.message;
            } finally {
                answerRepeat.disabled = answerSpeak.disabled = false;
            }
        }

        async function speakAnswer() {
            if (!currentAnswer) return;
            const spoken = currentAnswer.spoken || currentAnswer;
            answerRepeat.disabled = answerSpeak.disabled = true;
            answerStatus.textContent = `Reading the ${spoken.label} answer aloud\u2026`;
            try {
                const result = await requestLive(answerAudioEndpoint,
                    [spoken.text, spoken.language], null, 120000);
                if (result.error) {
                    answerStatus.textContent = result.error;
                } else {
                    player.hidden = false;
                    player.src = `data:audio/wav;base64,${result.audio}`;
                    await player.play().catch(() => {});
                    answerStatus.textContent = "Press play if the answer did not start.";
                }
            } catch (error) {
                answerStatus.textContent = error.message;
            } finally {
                answerRepeat.disabled = answerSpeak.disabled = false;
            }
        }

        answerRepeat.addEventListener("click", () => void repeatAnswer());
        answerSpeak.addEventListener("click", () => void speakAnswer());

        async function askQuestion() {
            const text = question.value.trim();
            if (!text) {
                askStatus.textContent = "Type a question first.";
                return;
            }
            ask.disabled = true;
            askStatus.textContent = "Searching and answering\u2026";
            try {
                // Answers come back in the language this speaker is using.
                const from = source.selectedOptions[0];
                const result = await requestLive(lookupEndpoint,
                    [text, source.value, place.value, control("situation").value], null, 180000);
                if (result.error) {
                    askStatus.textContent = result.error;
                } else {
                    askStatus.textContent = result.searched
                        ? "Answered using the web sources listed below."
                        : "Answered without a web search.";
                    control("ask-answer-label").textContent = `Answer \u00b7 ${from.textContent}`;
                    setText(askAnswer, result.answer, from.dataset.lang);
                    askSources.replaceChildren(...(result.sources || []).map(sourceLink));
                    askAnswerGroup.hidden = false;
                    askClear.disabled = false;
                    currentAnswer = {text: result.answer, language: source.value,
                                     label: from.textContent};
                    resetRepeat();
                }
            } catch (error) {
                askStatus.textContent = error.message;
            } finally {
                ask.disabled = false;
            }
        }

        ask.addEventListener("click", () => void askQuestion());
        question.addEventListener("keydown", event => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            void askQuestion();
        });
        askClear.addEventListener("click", () => {
            askAnswer.textContent = "";
            askSources.replaceChildren();
            askAnswerGroup.hidden = true;
            askClear.disabled = true;
            currentAnswer = null;
            resetRepeat();
            askStatus.textContent = "Ask about a place, a form, or a procedure.";
        });

        start.addEventListener("click", async () => {
            start.disabled = source.disabled = target.disabled = swap.disabled = pause.disabled = voice.disabled = twoWay.disabled = true;
            syncPlayback();
            show("opening", "Opening microphone…");
            let mic;
            try {
                mic = await openMicrophone();
                player.pause();
                player.removeAttribute("src");
                player.hidden = true;
                history.replaceChildren();
                empty.hidden = false;
                captionLines.hidden = true;
                captionEmpty.hidden = false;
                captionEmpty.textContent = "Listening for speech…";
                captionState.textContent = captionToggle.checked ? "Listening" : "Hidden";
                control("caption-announce").textContent = "";
                const from = source.selectedOptions[0];
                const to = target.selectedOptions[0];
                const both = twoWay.checked && from.value !== to.value;
                const useProfile = voice.value === "match" && profileSample && from.value !== to.value &&
                    (profileLanguage.value === from.value || (both && profileLanguage.value === to.value));
                const current = {
                    ...mic, source: source.value, target: target.value,
                    languages: {[from.value]: {name: from.textContent, tag: from.dataset.lang},
                                [to.value]: {name: to.textContent, tag: to.dataset.lang}},
                    twoWay: both, profileLanguage: useProfile ? profileLanguage.value : "",
                    listening: true, inflight: 0, playing: false, resumeAfter: 0,
                    pauseSamples: sampleRate * Number(pause.value),
                    voice: source.value === target.value ? "captions" : voice.value,
                    turn: 0, nextCaption: firstCaption, draftBusy: false,
                    playOrder: [], completed: new Map(),
                    prefix: [], chunks: [], pending: [], sounds: [], energyHistory: [], length: 0, quiet: 0, bufferedQuiet: 0, speech: 0,
                };
                session = current;
                mic.node.onaudioprocess = ({inputBuffer}) => {
                    if (current.listening && (!current.twoWay ||
                        ((!current.playing || player.paused) && performance.now() >= current.resumeAfter))) {
                        accept(current, inputBuffer.getChannelData(0).slice());
                    }
                };
                mic.media.getAudioTracks()[0].addEventListener("ended", () => stopCapture(current));
                stop.hidden = false;
                stop.disabled = false;
                show("listening", "Listening…");
            } catch (error) {
                if (mic) closeMicrophone(mic);
                start.disabled = source.disabled = target.disabled = swap.disabled = pause.disabled = twoWay.disabled = false;
                syncPlayback();
                show("error", `Microphone unavailable. Allow access and try again. ${error.message}`);
            }
        });
        stop.addEventListener("click", () => stopCapture(session));
        player.addEventListener("playing", () => {
            if (!session) return;
            session.playing = true;
            if (session.listening && session.twoWay) show("playing", "Playing translation. Listening resumes after playback.");
        });
        player.addEventListener("ended", () => {
            if (!session) return;
            session.playing = false;
            session.resumeAfter = performance.now() + 150;
            playNext(session);
            if (session.listening && !session.playing) show("listening", "Listening for your next turn…");
        });
        swap.addEventListener("click", () => {
            const previous = source.value;
            source.value = target.value;
            target.value = previous;
            syncPlayback();
        });
        source.addEventListener("change", syncPlayback);
        target.addEventListener("change", syncPlayback);
        twoWay.addEventListener("change", syncPlayback);
        captionToggle.addEventListener("change", () => {
            captionScreen.hidden = !captionToggle.checked;
            if (!captionToggle.checked) captionState.textContent = "Hidden";
            else if (session?.lastCaption) paintCaption(session, session.lastCaption, false);
            else captionState.textContent = session?.listening ? "Listening" : "Ready";
        });
        syncPlayback();
        return true;
    };

    if (!initialize()) {
        const observer = new MutationObserver(() => {
            if (initialize()) observer.disconnect();
        });
        observer.observe(document.body, {childList: true, subtree: true});
        setTimeout(() => observer.disconnect(), 10000);
    }
}
