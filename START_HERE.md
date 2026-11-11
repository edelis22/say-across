# Run Click on your computer

## Open your own copy

1. Unzip the source package. You’ll need Python 3.11. The standard setup uses PyTorch CPU wheels for Apple Silicon, Windows x64, or Linux x64. [Python downloads](https://www.python.org/downloads/) · [PyTorch installation](https://pytorch.org/get-started/previous-versions/)
2. Open Terminal on Mac or Linux, or PowerShell on Windows, in the extracted `polyglot-pragmatics` folder. Run the command for your computer.

   **Mac or Linux**

   ```sh
   python3.11 start.py
   ```

   **Windows**

   ```powershell
   py -3.11 start.py
   ```

3. Keep that terminal window open. The launcher opens your browser. Choose two languages, click **Start listening**, and allow microphone access if prompted. Use the same command to reopen Click later. [Local app](http://127.0.0.1:7860/)

If a copy of Click is already running, the launcher moves to the next free port and prints the address it used. Close the old terminal window to free the original port.

The first run installs dependencies and downloads the recognition and translation models when first used. Allow several GB of disk space and a few minutes for setup. The microphone and translation work without an API key. Add `--demo` for a smaller setup that opens fixed examples. Use `--port 7861` if the default port is busy.

The standard setup includes captions and the two-way interpreter. macOS provides system playback, and the other platforms can use eSpeak NG when installed. For the Apple Silicon setup with Whisper Turbo and AI voice matching, follow the [Apple Silicon instructions](README.md#faster-playback-and-voice-matching-on-apple-silicon).

## Turn on the understanding features

The spoken translation, including voice range matching, needs no API key. **Explain at another level** and **Ask about where you are** need an AI engine, and there are two ways to get one.

**Free, with no account and nothing to pay.** Install [Ollama](https://ollama.com) and download one model. The model runs on your own computer, so nothing goes to a paid service, and lookups are grounded in Wikipedia.

```sh
brew install ollama
ollama serve
```

```sh
ollama pull qwen2.5:7b
```

Leave `ollama serve` running and start Click as usual. The features appear by themselves and the panel reads **Answered by local model**. Expect a few GB of download, and simpler, slower answers than the paid tier.

**Paid, and more accurate.** Set a Claude key instead, in the terminal you start Click from. Click prefers it over the local model, and it adds live web search and better quality across all twelve languages.

```sh
read -rs 'ANTHROPIC_API_KEY?Anthropic API key: '
export ANTHROPIC_API_KEY
printf '\n'
python3.11 start.py
```

The launcher installs the client for you when it sees the key. Open **Understanding help** in the app to choose an explanation level, or to ask a question and then repeat the answer in another language or hear it read aloud. Both send text to Anthropic and are informational only. Run `python3.11 app.py --no-assistant` to keep them off while the key is still set. [Anthropic API keys](https://platform.claude.com/settings/keys)

## Ask Click out loud

Start a turn with the app's name and it answers instead of translating. Try **"Hey Click, look up what the current policy is in Uganda"**, or **"Hey Click, explain that in simpler terms"** right after a turn you did not follow. Looking things up works with nothing installed, from Wikipedia. Explaining in simpler words needs the free Ollama model above.

## Work with someone else

Send them the source ZIP or the GitHub repository. Each computer gets its own local app and conversation history. For a public walkthrough or shared HTTPS microphone app, follow the [publishing guide](PUBLISH.md). The [private hosting guide](HOSTING.md) covers password-protected review.
