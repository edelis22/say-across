# Run Say Across on your computer

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

3. Keep that terminal window open. The launcher opens your browser. Choose two languages, click **Start listening**, and allow microphone access if prompted. Use the same command to reopen Say Across later. [Local app](http://127.0.0.1:7860/)

The first run installs dependencies and downloads the recognition and translation models when first used. Allow several GB of disk space and a few minutes for setup. The microphone and translation work without an API key. Add `--demo` for a smaller setup that opens fixed examples. Use `--port 7861` if the default port is busy.

The standard setup includes captions and the two-way interpreter. macOS provides system playback, and the other platforms can use eSpeak NG when installed. For the Apple Silicon setup with Whisper Turbo and AI voice matching, follow the [Apple Silicon instructions](README.md#faster-playback-and-voice-matching-on-apple-silicon).

## Work with someone else

Send them the source ZIP or the GitHub repository. Each computer gets its own local app and conversation history. For a public walkthrough or shared HTTPS microphone app, follow the [publishing guide](PUBLISH.md). The [private hosting guide](HOSTING.md) covers password-protected review.
