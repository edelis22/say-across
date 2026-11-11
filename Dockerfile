FROM python:3.11-slim-trixie
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/home/app/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt requirements-emotion.txt requirements-assistant.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-assistant.txt
COPY app.py ./
COPY pragmatics ./pragmatics
USER app
EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
