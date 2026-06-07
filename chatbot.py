"""
chatbot.py
==========
Handles all LLM interactions through the Hugging Face Inference API.

No model weights are downloaded locally.  All generation goes through
huggingface_hub.InferenceClient.

Supported LLM models (HF_LLM_MODEL env var):
  - Qwen/Qwen2.5-72B-Instruct      (default — strong reasoning)
  - meta-llama/Llama-3.1-8B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3

Environment variables
---------------------
  HF_TOKEN     : Required. Your Hugging Face access token.
  HF_LLM_MODEL : Optional. Override the LLM model.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL  = "Qwen/Qwen2.5-72B-Instruct"
MAX_CONTEXT_CHARS  = 3000     # chars from retrieved chunks sent to LLM
MAX_NEW_TOKENS     = 512
TEMPERATURE        = 0.1
MAX_RETRIES        = 3
RETRY_BACKOFF      = 3.0      # seconds

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Welding Assistant. Answer welding-related questions using ONLY the context passages provided.

Rules:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. If the answer is not in the context, respond: "I couldn't find this information in the welding documents."
3. Always cite your source at the end: "Source: <filename>, Page <n>"
4. Be precise and technical when the question requires it.
5. Keep answers concise and focused."""


class ChatbotError(Exception):
    """Raised when the LLM API returns an unrecoverable error."""


class WeldingChatbot:
    """
    Generates answers to welding questions using retrieved context
    and the Hugging Face Inference API.

    Usage
    -----
    bot = WeldingChatbot()
    answer = bot.answer(query="What is MIG welding?", chunks=[...])
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.token: str = token or os.getenv("HF_TOKEN", "")
        self.model: str = model or os.getenv("HF_LLM_MODEL", DEFAULT_LLM_MODEL)

        if not self.token:
            raise ChatbotError(
                "HF_TOKEN is not set. "
                "Add it in Hugging Face Space → Settings → Repository Secrets."
            )

        self.client = InferenceClient(
            model=self.model,
            token=self.token,
        )
        logger.info("WeldingChatbot initialised  model=%s", self.model)

    # ── Public API ────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        chunks: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate an answer for *query* using *chunks* as grounding context.

        Parameters
        ----------
        query   : User's question.
        chunks  : Retrieved context dicts (from Retriever.retrieve()).
        history : Optional conversation history
                  [{"role": "user"|"assistant", "content": str}, ...]

        Returns
        -------
        Generated answer string.
        """
        if not chunks:
            return (
                "I couldn't find relevant information in the welding documents "
                "to answer your question. Please ensure welding PDFs are uploaded "
                "to the data/ folder and the index has been built."
            )

        context = self._build_context(chunks)
        messages = self._build_messages(query, context, history or [])

        return self._generate(messages)

    # ── Context builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts: List[str] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            src  = chunk.get("source_pdf",  "Unknown")
            page = chunk.get("page_number", "?")
            text = chunk.get("chunk_text",  "")

            # Truncate if we're approaching the context limit
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining] + "…"

            entry = f"[{i}] Source: {src}, Page {page}\n{text}"
            parts.append(entry)
            total_chars += len(text)

        return "\n\n---\n\n".join(parts)

    # ── Message builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        query: str,
        context: str,
        history: List[Dict],
    ) -> List[Dict]:
        """Construct the full chat-completion message list."""
        messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add recent conversation history (last 4 turns max to save tokens)
        for turn in history[-8:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Final user message with context
        user_content = (
            f"Context from welding documents:\n\n{context}\n\n"
            f"Question: {query}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── LLM call with retry ───────────────────────────────────────────────

    def _generate(self, messages: List[Dict]) -> str:
        """Call the HF chat-completion endpoint with exponential-backoff retry."""
        last_error: Exception = RuntimeError("Unknown error")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                text: str = response.choices[0].message.content
                return text.strip()

            except HfHubHTTPError as exc:
                status = getattr(exc, "response", None)
                status_code = status.status_code if status else 0

                if status_code == 401:
                    raise ChatbotError(
                        "Invalid or expired HF_TOKEN. "
                        "Regenerate at https://huggingface.co/settings/tokens"
                    ) from exc

                if status_code == 429:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "LLM rate limited (attempt %d/%d). Waiting %.1f s …",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue

                if status_code in (503, 504):
                    wait = RETRY_BACKOFF * attempt
                    logger.warning(
                        "LLM API unavailable (attempt %d/%d). Waiting %.1f s …",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue

                last_error = ChatbotError(
                    f"LLM API error {status_code}: {exc}"
                )
                logger.error(str(last_error))

            except Exception as exc:
                last_error = exc
                logger.warning("LLM call error on attempt %d: %s", attempt, exc)

            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF * attempt
                logger.info("Retrying LLM call in %.1f s …", sleep_time)
                time.sleep(sleep_time)

        logger.error("All LLM retry attempts exhausted.")
        return (
            "I'm sorry, the AI service is temporarily unavailable. "
            "Please try again in a moment."
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def format_sources(self, chunks: List[Dict]) -> str:
        """Return a formatted source citation block."""
        if not chunks:
            return ""
        seen: set = set()
        lines: List[str] = ["**Sources:**"]
        for chunk in chunks:
            src  = chunk.get("source_pdf",  "Unknown")
            page = chunk.get("page_number", "?")
            key  = (src, page)
            if key not in seen:
                seen.add(key)
                lines.append(f"  📄 {src} — Page {page}")
        return "\n".join(lines)
