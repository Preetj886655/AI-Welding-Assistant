# ── AI Welding Assistant — Dockerfile ────────────────────────────────────────
# For deployment on Hugging Face Spaces using the Docker SDK.
# Runs on CPU Basic (free tier).  No GPU required.
# No local model downloads — all AI calls use the HF Inference API.

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer-cached) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py        .
COPY chatbot.py    .
COPY embeddings.py .
COPY retriever.py  .

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p data vectorstore

# ── Hugging Face Spaces requires port 7860 ────────────────────────────────────
EXPOSE 7860

# ── HF_TOKEN is injected at runtime via Space secrets — never bake it in ──────
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
