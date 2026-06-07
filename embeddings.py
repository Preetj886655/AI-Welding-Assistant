"""
embeddings.py
=============
Generates text embeddings exclusively through the Hugging Face
Inference API.  No model weights are downloaded or stored locally.

Supported models (configured via HF_EMBEDDING_MODEL env var):
  - BAAI/bge-small-en-v1.5      (default, ~33 MB API-side)
  - BAAI/bge-base-en-v1.5
  - sentence-transformers/all-MiniLM-L6-v2

Environment variables
---------------------
  HF_TOKEN           : Required. Your Hugging Face access token.
  HF_EMBEDDING_MODEL : Optional. Override the embedding model.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_API_BASE = "https://api-inference.huggingface.co/pipeline/feature-extraction"

# Retry settings for transient API errors
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0   # seconds; doubles each attempt


class EmbeddingAPIError(Exception):
    """Raised when the Hugging Face Embedding API returns an unrecoverable error."""


class EmbeddingClient:
    """
    Thin wrapper around the HF Feature-Extraction Inference API.

    Usage
    -----
    client = EmbeddingClient()
    vectors = client.embed(["What is MIG welding?", "TIG welding uses argon."])
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.token: str = token or os.getenv("HF_TOKEN", "")
        self.model: str = (
            model
            or os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )

        if not self.token:
            raise EmbeddingAPIError(
                "HF_TOKEN is not set. "
                "Add it in Hugging Face Space → Settings → Repository Secrets."
            )

        self._url = f"{HF_API_BASE}/{self.model}"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        logger.info("EmbeddingClient initialised  model=%s", self.model)

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Parameters
        ----------
        texts : List of strings to embed (max ~100 at a time).

        Returns
        -------
        List of float vectors (one per input string).

        Raises
        ------
        EmbeddingAPIError : On authentication failure or unrecoverable API error.
        """
        if not texts:
            return []

        # Sanitise: replace empty strings to avoid API errors
        cleaned = [t.strip() or "." for t in texts]

        return self._call_with_retry(cleaned)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        BGE models recommend a short instruction prefix for retrieval queries.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vectors = self.embed([prefixed])
        return vectors[0]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _call_with_retry(self, texts: List[str]) -> List[List[float]]:
        """POST to the HF Inference API with exponential-backoff retries."""
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True},
        }

        last_error: Exception = RuntimeError("Unknown error")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_response(data, len(texts))

                if response.status_code == 401:
                    raise EmbeddingAPIError(
                        "Invalid or expired HF_TOKEN. "
                        "Regenerate your token at https://huggingface.co/settings/tokens"
                    )

                if response.status_code == 503:
                    # Model loading — wait and retry
                    wait = float(
                        response.json().get("estimated_time", RETRY_BACKOFF * attempt)
                    )
                    logger.warning(
                        "Embedding model loading (attempt %d/%d). Waiting %.1fs …",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(min(wait, 30))
                    continue

                if response.status_code == 429:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d). Waiting %.1fs …",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue

                # Other HTTP errors
                last_error = EmbeddingAPIError(
                    f"HF Embedding API error {response.status_code}: {response.text[:200]}"
                )
                logger.error(str(last_error))

            except requests.exceptions.Timeout:
                last_error = EmbeddingAPIError("Embedding API request timed out.")
                logger.warning("Timeout on attempt %d/%d", attempt, MAX_RETRIES)

            except requests.exceptions.ConnectionError as exc:
                last_error = EmbeddingAPIError(f"Network error: {exc}")
                logger.warning("Connection error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)

            # Exponential back-off before next attempt
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.info("Retrying in %.1f s …", sleep_time)
                time.sleep(sleep_time)

        raise last_error

    @staticmethod
    def _parse_response(
        data: object,
        expected_count: int,
    ) -> List[List[float]]:
        """
        Normalise the API response into a flat list-of-vectors.

        The HF feature-extraction API can return:
          - List[List[float]]            — one vector per input  ✓
          - List[List[List[float]]]      — token-level embeddings (need mean pooling)
        """
        if not isinstance(data, list):
            raise EmbeddingAPIError(f"Unexpected API response type: {type(data)}")

        vectors: List[List[float]] = []

        for item in data:
            if isinstance(item[0], float):
                # Already a flat vector
                vectors.append(item)
            elif isinstance(item[0], list):
                # Token-level → mean pool across token dimension
                n_tokens = len(item)
                dim = len(item[0])
                mean_vec = [
                    sum(item[t][d] for t in range(n_tokens)) / n_tokens
                    for d in range(dim)
                ]
                vectors.append(mean_vec)
            else:
                raise EmbeddingAPIError(
                    f"Unrecognised embedding format: {type(item[0])}"
                )

        if len(vectors) != expected_count:
            raise EmbeddingAPIError(
                f"Expected {expected_count} vectors, got {len(vectors)}."
            )

        return vectors


# ── Batch helper ──────────────────────────────────────────────────────────────

def embed_in_batches(
    client: EmbeddingClient,
    texts: List[str],
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Embed a large list of texts in batches to stay within API limits.

    Parameters
    ----------
    client     : Initialised EmbeddingClient.
    texts      : All texts to embed.
    batch_size : Number of texts per API call (default 32).

    Returns
    -------
    Flat list of all embedding vectors, in input order.
    """
    all_vectors: List[List[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        logger.info(
            "Embedding batch %d–%d of %d …",
            start + 1,
            min(start + batch_size, total),
            total,
        )
        vectors = client.embed(batch)
        all_vectors.extend(vectors)

    logger.info("Embedding complete: %d vectors generated.", len(all_vectors))
    return all_vectors
