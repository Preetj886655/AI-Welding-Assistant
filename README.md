---
title: AI Welding Assistant
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

# 🔥 AI Welding Assistant

An AI-powered chatbot that answers welding questions from your own PDF documents.
Built with **Retrieval-Augmented Generation (RAG)** and the **Hugging Face Inference API**.

> **Zero local model downloads.** All AI runs through API calls.
> Deployable on the free CPU Basic tier of Hugging Face Spaces.

---

## Project Overview

### What the AI Welding Assistant Does

The AI Welding Assistant is a domain-specific chatbot that reads your welding
PDF documents (standards, handbooks, guides), extracts their content, and
answers questions grounded strictly in those documents.

Every answer includes a source citation showing which PDF and page number the
information came from — so you can verify every response.

### How RAG Works

Retrieval-Augmented Generation (RAG) combines a **search engine** with a
**language model**:

```
Your Question
      │
      ▼
 ┌────────────────────────────────────────┐
 │  1. Embed your question via HF API     │
 │     (BAAI/bge-small-en-v1.5)          │
 └────────────────┬───────────────────────┘
                  │ query vector
                  ▼
 ┌────────────────────────────────────────┐
 │  2. Search FAISS for top-5 passages    │
 │     most similar to your question      │
 └────────────────┬───────────────────────┘
                  │ relevant chunks
                  ▼
 ┌────────────────────────────────────────┐
 │  3. Send chunks + question to LLM API  │
 │     (Qwen/Qwen2.5-72B-Instruct)       │
 └────────────────┬───────────────────────┘
                  │
                  ▼
        Answer + Source Citations
```

The LLM is instructed to answer **only** from the retrieved passages —
it cannot hallucinate information that isn't in your documents.

### Why Hugging Face Inference API?

| Approach | Download size | RAM needed | Cold start |
|---|---|---|---|
| Local model (Phi-3-mini) | ~4 GB | ~8 GB | 3–5 min |
| **HF Inference API** (this app) | **0 MB** | **~200 MB** | **< 5 s** |

Using the API means:
- The app starts instantly on the free CPU tier
- No GPU required
- Model upgrades are free (just change an environment variable)
- Works within Hugging Face Spaces' 16 GB RAM limit

---

## File Structure

```
HF_Welding_Assistant/
├── app.py            ← Gradio UI + event handlers
├── chatbot.py        ← LLM via HF InferenceClient
├── embeddings.py     ← Embeddings via HF Feature-Extraction API
├── retriever.py      ← PDF loading, chunking, FAISS index
├── requirements.txt  ← Python dependencies
├── Dockerfile        ← For HF Spaces Docker SDK
├── .env.example      ← Template for local development
├── data/             ← Place your welding PDFs here
└── vectorstore/      ← Auto-created; stores FAISS index
```

---

## Installation & Deployment

---

### Method 1 — Deploy via Hugging Face Web UI (Easiest)

#### Step 1: Create a Hugging Face Account

1. Open **https://huggingface.co**
2. Click **Sign Up** (top right)
3. Fill in your username, email, and password
4. Verify your email by clicking the link in the confirmation email

---

#### Step 2: Generate a Hugging Face Access Token

1. Log in and go to **https://huggingface.co/settings/tokens**
2. Click **"+ New token"**
3. Enter a name (e.g. `welding-assistant`)
4. Select role: **"Read"** (sufficient for Inference API)
5. Click **"Generate a token"**
6. **Copy the token immediately** — it starts with `hf_...`
   You will not be able to see it again.

---

#### Step 3: Create a Hugging Face Space

1. Go to **https://huggingface.co/spaces**
2. Click **"+ Create new Space"**
3. Fill in the form:

   | Field | Value |
   |---|---|
   | **Owner** | Your username |
   | **Space name** | `ai-welding-assistant` |
   | **License** | MIT |
   | **SDK** | **Docker** |
   | **Docker template** | Blank |
   | **Hardware** | **CPU Basic · Free** |
   | **Visibility** | Public or Private |

4. Click **"Create Space"**

> ⚠️ Select **Docker** as the SDK — the app uses a Dockerfile for
> dependency control and correct port binding.

---

#### Step 4: Add the HF_TOKEN Secret

This is the most important step. **Never upload your token as a file.**

1. In your Space, click the **"Settings"** tab
2. Scroll down to **"Repository secrets"**
3. Click **"New secret"**
4. Fill in:
   - **Name:** `HF_TOKEN`
   - **Value:** paste your token (`hf_...`)
5. Click **"Add secret"**

The secret is now securely injected as an environment variable at runtime.

**Optional secrets** you can add the same way:

| Name | Default | Description |
|---|---|---|
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `HF_LLM_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | LLM model |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Characters per chunk |

---

#### Step 5: Upload Project Files

1. In your Space, click the **"Files"** tab
2. Click **"+ Add file"** → **"Upload files"**
3. Upload these files **one by one** or drag them all at once:
   - `app.py`
   - `chatbot.py`
   - `embeddings.py`
   - `retriever.py`
   - `requirements.txt`
   - `Dockerfile`
4. In the commit message box type: `Initial deployment`
5. Click **"Commit changes to main"**

---

#### Step 6: Wait for Build

1. Click the **"App"** tab in your Space
2. You will see a **yellow "Building"** badge — this takes 2–5 minutes
3. Hugging Face will:
   - Pull the Docker base image
   - Run `pip install -r requirements.txt`
   - Start `python app.py`
4. When the badge turns **green "Running"**, your app is live
5. The URL is: `https://huggingface.co/spaces/YOUR_USERNAME/ai-welding-assistant`

---

#### Step 7: Upload Your Welding PDFs

1. Open your running Space
2. Click the **"📄 Documents"** tab
3. Click **"Select PDF files"** and choose your welding PDFs
4. Click **"⬆️ Upload PDFs"**
5. Click **"🔨 Build Index"** — this calls the HF Embedding API to
   generate vectors and builds the FAISS index
6. Switch to the **"💬 Chat"** tab and ask your first question!

---

### Method 2 — Deploy via Git (Recommended for Updates)

#### Prerequisites

```bash
# Install Git
# Windows: https://git-scm.com/download/win
# macOS:   brew install git
# Linux:   sudo apt install git

# Install Git LFS (for large files like PDFs)
git lfs install

# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Paste your HF token when prompted
```

#### Clone Your Space

```bash
# Replace YOUR_USERNAME with your HF username
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-welding-assistant
cd ai-welding-assistant
```

#### Add Project Files

```bash
# Copy all project files into the cloned folder
cp /path/to/HF_Welding_Assistant/app.py        .
cp /path/to/HF_Welding_Assistant/chatbot.py    .
cp /path/to/HF_Welding_Assistant/embeddings.py .
cp /path/to/HF_Welding_Assistant/retriever.py  .
cp /path/to/HF_Welding_Assistant/requirements.txt .
cp /path/to/HF_Welding_Assistant/Dockerfile    .
```

#### Track Large Files with Git LFS

```bash
git lfs track "data/*.pdf"
git lfs track "vectorstore/*.faiss"
git lfs track "vectorstore/*.pkl"
git add .gitattributes
```

#### Commit and Push

```bash
git add .
git commit -m "Deploy AI Welding Assistant"
git push
```

When prompted for credentials:
- **Username:** your Hugging Face username
- **Password:** your Hugging Face **Access Token** (not your account password)

The Space will rebuild automatically after every push.

---

## Running the Application Locally

For local development and testing:

```bash
# 1. Clone or download the project
cd HF_Welding_Assistant

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 5. Add PDFs to data/
# (copy your welding PDF files into the data/ folder)

# 6. Launch
python app.py
```

Open **http://localhost:7860** in your browser.

---

## How Hugging Face Spaces Builds Your App

When you push code to your Space, HF automatically:

1. **Detects the Dockerfile** and builds a Docker image
2. **Runs** `pip install -r requirements.txt` inside the container
3. **Executes** `python app.py` as the entry point
4. **Exposes port 7860** and assigns it your Space URL
5. **Injects secrets** as environment variables (including `HF_TOKEN`)

The Gradio server starts on `0.0.0.0:7860` — Hugging Face's proxy
routes public traffic to that port automatically.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    HUGGING FACE SPACES                       │
│                    (CPU Basic — Free)                        │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                  app.py (Gradio UI)                  │  │
│   │  ┌──────────────┐          ┌───────────────────────┐ │  │
│   │  │  Chat Tab    │          │   Documents Tab       │ │  │
│   │  │  - Chatbot   │          │   - PDF Upload        │ │  │
│   │  │  - Sources   │          │   - Build Index       │ │  │
│   │  └──────┬───────┘          └───────────────────────┘ │  │
│   └─────────┼────────────────────────────────────────────┘  │
│             │ user query                                      │
│   ┌─────────▼────────────────────────────────────────────┐  │
│   │              retriever.py (Retriever)                 │  │
│   │   PDFs → clean → chunk → embed → FAISS               │  │
│   └─────────┬────────────────────────────────────────────┘  │
│             │                                                 │
│    ┌────────▼──────────┐     ┌────────────────────────────┐  │
│    │  embeddings.py    │     │       chatbot.py           │  │
│    │  EmbeddingClient  │     │     WeldingChatbot         │  │
│    └────────┬──────────┘     └────────────┬───────────────┘  │
│             │                             │                   │
└─────────────┼─────────────────────────────┼───────────────────┘
              │ HTTPS API call              │ HTTPS API call
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│  HF Feature-Extraction  │   │    HF Chat-Completion API       │
│  BAAI/bge-small-en-v1.5 │   │  Qwen/Qwen2.5-72B-Instruct     │
└─────────────────────────┘   └─────────────────────────────────┘
```

---

## Troubleshooting

### ❌ "HF_TOKEN is not set"

**Cause:** The secret was not added to the Space, or the app was
built before the secret was added.

**Fix:**
1. Go to Space → **Settings → Repository Secrets**
2. Add `HF_TOKEN` with your token value
3. Go to Space → **Settings** → scroll to bottom → **"Factory reboot"**

---

### ❌ Build failure: "ModuleNotFoundError"

**Cause:** A package in `requirements.txt` failed to install.

**Fix:**
1. Click the **"Logs"** tab in your Space
2. Find the failing package name
3. Check that the package name is correct (e.g. `pymupdf`, not `PyMuPDF`)
4. Push a fix to trigger a rebuild

---

### ❌ "No PDF files found in data/"

**Cause:** No PDFs have been uploaded yet.

**Fix:**
1. Open the **📄 Documents** tab
2. Upload your welding PDFs
3. Click **🔨 Build Index**

---

### ❌ Embedding API error 503 "Model is loading"

**Cause:** The embedding model was cold (not recently used) and is
loading on HF's infrastructure. The app retries automatically.

**Fix:** Wait 20–30 seconds and try again. The model stays warm
after the first successful call.

---

### ❌ LLM API error 429 "Rate limit exceeded"

**Cause:** You've exceeded the free-tier rate limit for the
Inference API.

**Fixes:**
- Wait a few minutes and retry
- Switch to a smaller model: set `HF_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3`
- Upgrade to a HF Pro account for higher rate limits

---

### ❌ FAISS index loads but returns no results

**Cause:** The index was built with a different embedding model than
the one currently configured.

**Fix:**
1. Delete the vectorstore files (or click Build Index again)
2. Ensure `HF_EMBEDDING_MODEL` is consistent between index builds and queries

---

### ❌ PDF loads but text is empty

**Cause:** The PDF is scanned (image-only) rather than text-based.

**Fix:** Use a PDF with selectable text, or run an OCR tool (e.g.
Adobe Acrobat, Tesseract) on your PDFs before uploading.

---

### ❌ App crashes with "Out of Memory"

**Cause:** Too many large PDFs were indexed, filling the 16 GB RAM
of the free tier.

**Fixes:**
- Reduce `CHUNK_SIZE` to `300` (smaller chunks = smaller metadata)
- Index fewer PDFs at a time
- Upgrade to a Space with more RAM

---

## Cost Optimisation

### Minimise API Calls

The FAISS index is **persisted to disk** after the first build.
On subsequent restarts, the app loads the saved index — no embedding
API calls are made unless you click **Build Index** again.

```
First startup  : N_chunks × (1 embedding API call)   ← one-time cost
Every query    : 1 embedding call + 1 LLM call        ← per question
```

### Cache Embeddings Aggressively

Never rebuild the index unless you add new PDFs. The `vectorstore/`
folder stores `index.faiss` and `metadata.pkl` — keep these files
between deployments using Git LFS:

```bash
git lfs track "vectorstore/*.faiss"
git lfs track "vectorstore/*.pkl"
git add vectorstore/
git commit -m "Save pre-built FAISS index"
git push
```

This means your Space loads instantly on every restart with zero
API calls during startup.

### Reduce Token Usage

Control how many tokens the LLM receives per request:

| Variable | Default | Effect |
|---|---|---|
| `TOP_K` | `5` | Fewer chunks = fewer context tokens |
| `CHUNK_SIZE` | `500` | Smaller chunks = shorter context |
| `MAX_CONTEXT_CHARS` | `3000` | Hard cap in `chatbot.py` |

For simple factual Q&A, `TOP_K=3` and `CHUNK_SIZE=300` work well.

### Choose the Right Models

| Use Case | Embedding Model | LLM |
|---|---|---|
| Best free performance | `BAAI/bge-small-en-v1.5` | `Qwen/Qwen2.5-72B-Instruct` |
| Fastest (lowest latency) | `sentence-transformers/all-MiniLM-L6-v2` | `mistralai/Mistral-7B-Instruct-v0.3` |
| Best quality | `BAAI/bge-base-en-v1.5` | `meta-llama/Llama-3.1-8B-Instruct` |

### Reduce Latency

The main sources of latency are:
1. **Cold embedding model** (~15–25 s first call) — warms up automatically
2. **LLM generation** (~3–8 s per answer) — irreducible on free API

To minimise perceived latency, keep conversations short and focused.

---

## Security Notes

- `HF_TOKEN` is **never** stored in code or committed to Git
- Use **Repository Secrets** in HF Spaces — they are encrypted at rest
- For production, use a **read-only token** (sufficient for Inference API)
- Rotate your token periodically at https://huggingface.co/settings/tokens

---

## Supported Models

### Embedding Models

| Model | Dimensions | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | Default. Fast and accurate. |
| `BAAI/bge-base-en-v1.5` | 768 | Better quality, 2× slower. |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Very fast, good baseline. |

### LLM Models

| Model | Context | Notes |
|---|---|---|
| `Qwen/Qwen2.5-72B-Instruct` | 32k | Default. Best reasoning. |
| `meta-llama/Llama-3.1-8B-Instruct` | 128k | Fast, lower rate limits. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 32k | Reliable fallback. |

---

## License

MIT License — free to use, modify, and deploy.
