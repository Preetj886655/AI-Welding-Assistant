"""
app.py
======
AI Welding Assistant — Gradio interface.

Runs entirely on CPU Basic (free tier) on Hugging Face Spaces.
No local model downloads.  All AI calls go to the HF Inference API.

Environment variables required
-------------------------------
  HF_TOKEN : Your Hugging Face access token.
             Set in Space → Settings → Repository Secrets.

Optional overrides
------------------
  HF_EMBEDDING_MODEL : Embedding model (default: BAAI/bge-small-en-v1.5)
  HF_LLM_MODEL       : LLM model (default: Qwen/Qwen2.5-72B-Instruct)
  DATA_DIR           : Path to PDF folder (default: data/)
  TOP_K              : Chunks retrieved per query (default: 5)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from chatbot import WeldingChatbot, ChatbotError
from embeddings import EmbeddingAPIError
from retriever import Retriever

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR",   "data"))
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "vectorstore/index.faiss"))
META_PATH  = Path(os.getenv("FAISS_META_PATH",  "vectorstore/metadata.pkl"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Global singletons (lazy init inside Gradio handlers) ─────────────────────
_retriever: Optional[Retriever] = None
_chatbot:   Optional[WeldingChatbot] = None
_init_error: Optional[str] = None


def _initialise() -> Tuple[Optional[Retriever], Optional[WeldingChatbot], Optional[str]]:
    """
    Initialise Retriever and WeldingChatbot singletons.

    Returns (retriever, chatbot, error_message).
    Called once lazily on first user interaction.
    """
    global _retriever, _chatbot, _init_error

    if _retriever is not None and _chatbot is not None:
        return _retriever, _chatbot, None

    token = os.getenv("HF_TOKEN", "")
    if not token:
        msg = (
            "⚠️ **HF_TOKEN is not set.**\n\n"
            "Go to your Hugging Face Space → **Settings → Repository Secrets** "
            "and add `HF_TOKEN` with your access token.\n\n"
            "Generate a token at https://huggingface.co/settings/tokens"
        )
        _init_error = msg
        return None, None, msg

    # Build Retriever (loads/builds FAISS index)
    try:
        logger.info("Initialising Retriever …")
        _retriever = Retriever(
            data_dir=DATA_DIR,
            index_path=INDEX_PATH,
            meta_path=META_PATH,
            hf_token=token,
        )
        logger.info("Retriever ready — %d chunks indexed.", _retriever.document_count)
    except FileNotFoundError as exc:
        msg = (
            f"⚠️ **No PDF documents found.**\n\n{exc}\n\n"
            "Upload welding PDF files using the **Upload PDFs** panel below, "
            "then click **Build Index**."
        )
        _init_error = msg
        return None, None, msg
    except EmbeddingAPIError as exc:
        msg = f"⚠️ **Embedding API error:** {exc}"
        _init_error = msg
        return None, None, msg
    except Exception as exc:
        msg = f"⚠️ **Initialisation error:** {exc}"
        _init_error = msg
        logger.exception("Retriever init failed")
        return None, None, msg

    # Build Chatbot
    try:
        logger.info("Initialising WeldingChatbot …")
        _chatbot = WeldingChatbot(token=token)
        logger.info("Chatbot ready.")
    except ChatbotError as exc:
        msg = f"⚠️ **Chatbot error:** {exc}"
        _init_error = msg
        return None, None, msg

    _init_error = None
    return _retriever, _chatbot, None


# ═════════════════════════════════════════════════════════════════════════════
# Gradio event handlers
# ═════════════════════════════════════════════════════════════════════════════

def chat(
    user_message: str,
    history: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], str, str]:
    """
    Main chat handler.

    Parameters
    ----------
    user_message : Text typed by the user.
    history      : Gradio chatbot history [(user_msg, bot_msg), ...].

    Returns
    -------
    (updated_history, sources_markdown, status_markdown)
    """
    if not user_message.strip():
        return history, "", "⚡ Ready"

    retriever, chatbot, err = _initialise()

    if err:
        updated = history + [(user_message, err)]
        return updated, "", "❌ Error"

    # ── Retrieve ──────────────────────────────────────────────────────────
    status = "🔍 Retrieving relevant passages …"
    try:
        chunks = retriever.retrieve(user_message)
    except EmbeddingAPIError as exc:
        error_msg = f"Embedding API error: {exc}"
        logger.error(error_msg)
        updated = history + [(user_message, f"⚠️ {error_msg}")]
        return updated, "", "❌ Embedding error"

    # ── Generate ──────────────────────────────────────────────────────────
    status = "🤖 Generating answer …"
    hf_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for pair in history
        for i, msg in enumerate(pair)
        if msg
    ]

    try:
        answer = chatbot.answer(user_message, chunks, history=hf_history)
    except ChatbotError as exc:
        error_msg = f"LLM API error: {exc}"
        logger.error(error_msg)
        updated = history + [(user_message, f"⚠️ {error_msg}")]
        return updated, "", "❌ LLM error"

    # ── Format sources ────────────────────────────────────────────────────
    sources_md = chatbot.format_sources(chunks) if chunks else ""

    updated_history = history + [(user_message, answer)]
    return updated_history, sources_md, "✅ Ready"


def upload_pdfs(files) -> Tuple[str, str]:
    """
    Save uploaded PDF files to the data/ directory.

    Returns (status_message, upload_log).
    """
    if not files:
        return "No files selected.", ""

    saved: List[str] = []
    errors: List[str] = []

    for file in files:
        src = Path(file.name)
        if src.suffix.lower() != ".pdf":
            errors.append(f"Skipped (not a PDF): {src.name}")
            continue
        dest = DATA_DIR / src.name
        try:
            shutil.copy(str(src), str(dest))
            saved.append(src.name)
            logger.info("PDF uploaded: %s → %s", src.name, dest)
        except Exception as exc:
            errors.append(f"Error saving {src.name}: {exc}")
            logger.error("Upload error: %s", exc)

    lines: List[str] = []
    if saved:
        lines.append(f"✅ Saved {len(saved)} PDF(s): {', '.join(saved)}")
    if errors:
        lines.extend([f"⚠️ {e}" for e in errors])

    return "\n".join(lines), "\n".join(saved)


def build_index() -> str:
    """Rebuild the FAISS index from the current data/ contents."""
    global _retriever, _chatbot, _init_error

    token = os.getenv("HF_TOKEN", "")
    if not token:
        return "⚠️ HF_TOKEN is not set. Cannot build index."

    # Delete existing index so Retriever rebuilds it
    for p in [INDEX_PATH, META_PATH]:
        if p.exists():
            p.unlink()

    _retriever = None
    _chatbot = None
    _init_error = None

    try:
        retriever = Retriever(
            data_dir=DATA_DIR,
            index_path=INDEX_PATH,
            meta_path=META_PATH,
            hf_token=token,
        )
        _retriever = retriever
        _chatbot = WeldingChatbot(token=token)
        return (
            f"✅ Index built successfully!\n"
            f"   {retriever.document_count} chunks indexed from PDF(s) in {DATA_DIR}/"
        )
    except FileNotFoundError as exc:
        return f"⚠️ {exc}"
    except Exception as exc:
        logger.exception("Index build failed")
        return f"❌ Build failed: {exc}"


def get_status() -> str:
    """Return current system status as a markdown string."""
    token = os.getenv("HF_TOKEN", "")
    token_ok  = "✅" if token else "❌"
    index_ok  = "✅" if INDEX_PATH.exists() else "⚠️ Not built"
    pdfs = list(DATA_DIR.glob("*.pdf"))
    pdf_count = len(pdfs)

    chunks = _retriever.document_count if _retriever else "—"
    embed_model = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    llm_model   = os.getenv("HF_LLM_MODEL",       "Qwen/Qwen2.5-72B-Instruct")

    return (
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| HF_TOKEN | {token_ok} |\n"
        f"| PDFs loaded | {pdf_count} |\n"
        f"| FAISS index | {index_ok} |\n"
        f"| Chunks indexed | {chunks} |\n"
        f"| Embedding model | `{embed_model}` |\n"
        f"| LLM model | `{llm_model}` |"
    )


def clear_chat() -> Tuple[List, str, str]:
    """Clear the chat history."""
    return [], "", "⚡ Ready"


# ═════════════════════════════════════════════════════════════════════════════
# Gradio Layout
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Global ── */
body, .gradio-container {
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* ── Header ── */
.header-box {
    background: linear-gradient(135deg, #1a1f2e, #0d1117);
    border-bottom: 2px solid #e36b2d;
    padding: 1.4rem 2rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.header-box h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e36b2d;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-box p {
    color: #8b949e;
    margin: 0.3rem 0 0;
    font-size: 0.9rem;
}

/* ── Panels ── */
.panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

/* ── Chat bubbles ── */
.message.user .bubble-wrap { background: #1f4068 !important; border-radius: 12px 12px 4px 12px !important; }
.message.bot  .bubble-wrap { background: #1c2128 !important; border-radius: 12px 12px 12px 4px !important; }

/* ── Buttons ── */
button.primary {
    background: #e36b2d !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
button.primary:hover { opacity: 0.88 !important; }
button.secondary {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* ── Inputs ── */
input[type="text"], textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #e36b2d !important;
}

/* ── Source panel ── */
.sources-panel {
    background: #1c2128;
    border: 1px solid #e36b2d44;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
}

/* ── Status bar ── */
.status-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
}

/* ── Tab styling ── */
.tab-nav button { color: #8b949e !important; }
.tab-nav button.selected { color: #e36b2d !important; border-bottom: 2px solid #e36b2d !important; }
"""

HEADER_HTML = """
<div class="header-box">
    <h1>🔥 AI Welding Assistant</h1>
    <p>Powered by Hugging Face Inference API &nbsp;·&nbsp;
       FAISS Vector Search &nbsp;·&nbsp;
       Zero local model downloads</p>
</div>
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="AI Welding Assistant") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot_ui = gr.Chatbot(
                            label="AI Welding Assistant",
                            height=480,
                            show_label=False,
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            user_input = gr.Textbox(
                                placeholder="Ask a welding question …  (e.g. What is TIG welding?)",
                                show_label=False,
                                scale=5,
                                lines=1,
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                        status_md = gr.Markdown("⚡ Ready", elem_classes=["status-bar"])

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Sources")
                        sources_box = gr.Markdown(
                            value="*Sources will appear here after your first question.*",
                            elem_classes=["sources-panel"],
                        )
                        gr.Markdown("---")
                        gr.Markdown("### ℹ️ System Status")
                        status_table = gr.Markdown(get_status())
                        refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # ── Tab 2: Documents ──────────────────────────────────────────
            with gr.TabItem("📄 Documents"):
                gr.Markdown(
                    "### Upload Welding PDFs\n"
                    "Upload one or more welding PDF documents, then click **Build Index** "
                    "to make them searchable."
                )
                with gr.Row():
                    with gr.Column():
                        upload_widget = gr.File(
                            label="Select PDF files",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        upload_btn   = gr.Button("⬆️ Upload PDFs",  variant="primary")
                        build_btn    = gr.Button("🔨 Build Index",  variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False,
                            lines=3,
                        )
                        build_status = gr.Textbox(
                            label="Index Build Status",
                            interactive=False,
                            lines=4,
                        )

                    with gr.Column():
                        gr.Markdown(
                            "### Instructions\n\n"
                            "1. Click **Select PDF files** and choose your welding documents.\n"
                            "2. Click **⬆️ Upload PDFs** to save them to the `data/` folder.\n"
                            "3. Click **🔨 Build Index** to generate embeddings and build the FAISS index.\n"
                            "4. Switch to the **Chat** tab and ask your questions!\n\n"
                            "---\n"
                            "**Supported formats:** PDF only\n\n"
                            "**Tip:** Larger PDFs take longer to index because each chunk "
                            "requires an API call for embedding generation."
                        )

            # ── Tab 3: About ──────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## About AI Welding Assistant

This application uses **Retrieval-Augmented Generation (RAG)** to answer
welding questions from your uploaded documents.

### How it works

```
User Question
     ↓
HF Embedding API  ←  query embedding (no local download)
     ↓
FAISS Vector Search  ←  finds top-5 relevant passages
     ↓
Context Assembly
     ↓
HF LLM API  ←  generates grounded answer (no local download)
     ↓
Answer + Source Citations
```

### Models used

| Component | Model |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| LLM | `Qwen/Qwen2.5-72B-Instruct` |

### No local model downloads

All AI inference happens through the Hugging Face Inference API.
The app runs on **CPU Basic (free tier)** on Hugging Face Spaces.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | ✅ Yes | Your HF access token |
| `HF_EMBEDDING_MODEL` | Optional | Override embedding model |
| `HF_LLM_MODEL` | Optional | Override LLM model |
| `TOP_K` | Optional | Chunks retrieved per query (default: 5) |
                """)

        # ── Event bindings ────────────────────────────────────────────────

        # Send message
        send_btn.click(
            fn=chat,
            inputs=[user_input, chatbot_ui],
            outputs=[chatbot_ui, sources_box, status_md],
        ).then(lambda: "", outputs=user_input)

        user_input.submit(
            fn=chat,
            inputs=[user_input, chatbot_ui],
            outputs=[chatbot_ui, sources_box, status_md],
        ).then(lambda: "", outputs=user_input)

        # Upload PDFs
        upload_btn.click(
            fn=upload_pdfs,
            inputs=[upload_widget],
            outputs=[upload_status, gr.State()],
        )

        # Build index
        build_btn.click(
            fn=build_index,
            outputs=[build_status],
        ).then(
            fn=get_status,
            outputs=[status_table],
        )

        # Refresh status
        refresh_status_btn.click(fn=get_status, outputs=[status_table])

        # Clear chat
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot_ui, sources_box, status_md],
        )

        # Auto-init on load (non-blocking; errors shown on first query)
        demo.load(fn=get_status, outputs=[status_table])

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
