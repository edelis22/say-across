const examples = {
  greek: [
    {source: "Greek", target: "English", from: "el-GR", to: "en-US",
      original: "Ευχαριστώ πάρα πολύ.", translation: "Thank you very much."},
    {source: "English", target: "Greek", from: "en-US", to: "el-GR",
      original: "Thank you for helping me.", translation: "Ευχαριστώ που με βοήθησες."},
  ],
  spanish: [
    {source: "English", target: "Spanish", from: "en-US", to: "es-ES",
      original: "Thank you for helping me.", translation: "Gracias por ayudarme."},
    {source: "Spanish", target: "English", from: "es-ES", to: "en-US",
      original: "Muchas gracias.", translation: "Thank you very much."},
  ],
};

const element = id => document.getElementById(id);
const pairButtons = [...document.querySelectorAll("[data-pair]")];
const audioStatus = element("audio-status");
const canSpeak = typeof SpeechSynthesisUtterance === "function" && Boolean(window.speechSynthesis);
const idleMessage = "These examples use generated test speech and fixed translations.";
let pair = "greek";
let index = 0;

function showTurn() {
  if (canSpeak) window.speechSynthesis.cancel();
  const turn = examples[pair][index];
  element("direction").textContent = `${turn.source} → ${turn.target}`;
  element("count").textContent = `Turn ${index + 1} of ${examples[pair].length}`;
  element("source-label").textContent = `Original · ${turn.source}`;
  element("target-label").textContent = `Translation · ${turn.target}`;
  element("source-text").textContent = turn.original;
  element("source-text").lang = turn.from;
  element("target-text").textContent = turn.translation;
  element("target-text").lang = turn.to;
  audioStatus.textContent = canSpeak ? idleMessage : "Browser voice playback is unavailable on this device.";
}

function speak(id) {
  const message = element(id);
  const utterance = new SpeechSynthesisUtterance(message.textContent);
  const voices = window.speechSynthesis.getVoices();
  const language = message.lang.toLowerCase();
  const voice = voices.find(candidate => candidate.lang.toLowerCase() === language)
    || voices.find(candidate => candidate.lang.split("-")[0].toLowerCase() === language.split("-")[0]);
  if (voices.length && !voice) {
    audioStatus.textContent = "A voice for this language is unavailable in your browser.";
    return;
  }
  window.speechSynthesis.cancel();
  utterance.lang = message.lang;
  if (voice) utterance.voice = voice;
  utterance.onend = () => { audioStatus.textContent = idleMessage; };
  utterance.onerror = event => {
    if (!["interrupted", "canceled"].includes(event.error))
      audioStatus.textContent = "Your browser could not play this example.";
  };
  audioStatus.textContent = "Playing the browser voice…";
  window.speechSynthesis.speak(utterance);
}

for (const button of pairButtons) {
  button.addEventListener("click", () => {
    pair = button.dataset.pair;
    index = 0;
    for (const choice of pairButtons) choice.setAttribute("aria-pressed", choice === button);
    showTurn();
  });
}

element("next").addEventListener("click", () => {
  index = (index + 1) % examples[pair].length;
  showTurn();
});
for (const [button, content] of [["hear-source", "source-text"], ["hear-target", "target-text"]]) {
  element(button).disabled = !canSpeak;
  if (canSpeak) element(button).addEventListener("click", () => speak(content));
}

const githubOwner = location.hostname.match(/^([a-z0-9-]+)\.github\.io$/i)?.[1];
const githubProject = location.pathname.split("/").filter(Boolean)[0];
if (githubOwner && /^[a-z0-9._-]+$/i.test(githubProject || "")) {
  const repository = element("repository");
  repository.href = `https://github.com/${githubOwner}/${githubProject}`;
  repository.hidden = false;
}
showTurn();
