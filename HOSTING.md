# Private browser access

A private Docker Space on Hugging Face gives you a stable HTTPS page and runs independently of your Mac. Use an organization that both you and your partner can access. Hugging Face currently requires an eligible paid organization plan to create a compute-backed organization Space. Start with the included CPU Basic hardware and test live latency on your own voices before selecting faster hardware. [Spaces visibility and hardware](https://huggingface.co/docs/hub/spaces-overview) · [Private repository access](https://huggingface.co/docs/huggingface_hub/guides/repository)

1. Sign in to Hugging Face and use or create an eligible organization with your partner as a member.
2. Create a Space under that organization. Select **Docker** and **Private**. Upload this repository to the Space. Its README declares the Docker runtime and port.
3. Under **Settings → Variables and secrets**, add a secret named `POLYGLOT_ACCESS_PASSWORD` with a strong password of at least 16 characters. Share that password privately. The app username is `partner`.
4. When the Space is running, open its **App** tab from each account, sign in, and allow microphone access. Check that **Start listening** shows draft captions during longer speech and completes turns after the selected pauses. The first turn may need time to fetch the model weights. Keep the Space page as the bookmark for both of you. [Create and configure a Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker)

The app listens on port 7860 inside the container. The hosted process requires the app password and disables optional remote adaptation. The browser sends short rolling caption windows and completed audio turns to the hosting instance, which runs recognition and translation. Live audio is held in memory for processing. Single uploaded recordings and generated clips may be cached temporarily by Gradio. The Linux image offers captions and expressive eSpeak NG playback. The optional voice-matching engine runs on Apple Silicon. Evaluate caption quality and intelligibility on the deployed service by language. [eSpeak NG language coverage](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)

For this research prototype, keep access within the private test group. The NLLB checkpoint is published for noncommercial research under CC BY-NC 4.0. [Meta model card](https://huggingface.co/facebook/nllb-200-distilled-600M)

The hosted app can be installed from a compatible browser when its page is opened in a top-level HTTPS context. Verify microphone permissions in that browser as part of setup.
