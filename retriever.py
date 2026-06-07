"""
retriever.py
============
Handles the full document pipeline:
  1. Load PDFs from the data/ directory
  2. Extract and clean text
  3. Split into overlapping chunks
  4. Generate embeddings via HF Inference API (no local models)
  5. Build and persist a FAISS vector index
  6. Retrieve top-k relevant chunks for a given query

No model weights are downloaded.  All embedding calls go to
the Hugging Face Inference API through embeddings.py.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz                          # PyMuPDF
import numpy as np

# FAISS is imported lazily to give a friendly error if not installed
try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is not installed. Add it to requirements.txt."
    ) from exc

from embeddings import EmbeddingClient, EmbeddingAPIError, embed_in_batches

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Defaults (all overridable via environment variables) ──────────────────────
DEFAULT_DATA_DIR       = Path(os.getenv("DATA_DIR",       "data"))
DEFAULT_INDEX_PATH     = Path(os.getenv("FAISS_INDEX_PATH", "vectorstore/index.faiss"))
DEFAULT_META_PATH      = Path(os.getenv("FAISS_META_PATH",  "vectorstore/metadata.pkl"))
DEFAULT_CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",    "500"))
DEFAULT_CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K          = int(os.getenv("TOP_K",          "5"))
EMBED_BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "32"))


# ═════════════════════════════════════════════════════════════════════════════
# Text utilities
# ═════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Remove noise common in PDF-extracted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse spaces
    text = re.sub(r"-\s*\d+\s*-", "", text)          # page markers
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)     # non-printable chars
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping fixed-size chunks.

    Uses a simple sliding window.  Splits prefer paragraph boundaries
    before falling back to word boundaries.
    """
    # Try to split on paragraph boundaries first
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(current)
            # If paragraph itself exceeds chunk_size, split by words
            if len(para) > chunk_size:
                words = para.split()
                buf = ""
                for word in words:
                    if len(buf) + len(word) + 1 <= chunk_size:
                        buf = (buf + " " + word).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = word
                if buf:
                    current = buf
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap: append chunk_overlap chars from previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            suffix = chunks[i - 1][-chunk_overlap:]
            overlapped.append((suffix + " " + chunks[i]).strip())
        return overlapped

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# PDF loading
# ═════════════════════════════════════════════════════════════════════════════

def load_pdfs(data_dir: Path = DEFAULT_DATA_DIR) -> List[Dict]:
    """
    Load all PDF files from *data_dir*.

    Returns
    -------
    List of dicts, each with keys:
      text        : cleaned page text
      source_pdf  : filename
      page_number : 1-based page index
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return []

    pages: List[Dict] = []
    for pdf_path in pdf_files:
        logger.info("Loading PDF: %s", pdf_path.name)
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to open %s: %s", pdf_path.name, exc)
            continue

        for page_idx in range(len(doc)):
            try:
                raw = doc[page_idx].get_text()
                cleaned = clean_text(raw)
                if cleaned:
                    pages.append({
                        "text": cleaned,
                        "source_pdf": pdf_path.name,
                        "page_number": page_idx + 1,
                    })
            except Exception as exc:
                logger.warning(
                    "Error reading page %d of %s: %s",
                    page_idx + 1, pdf_path.name, exc
                )

        doc.close()
        logger.info("  → %d pages loaded from %s", len(doc), pdf_path.name)

    logger.info("Total pages loaded: %d from %d PDF(s)", len(pages), len(pdf_files))
    return pages


# ═════════════════════════════════════════════════════════════════════════════
# FAISS index management
# ═════════════════════════════════════════════════════════════════════════════

class FAISSStore:
    """
    Flat L2 FAISS index backed by a parallel metadata list.

    Each entry in the index corresponds to one text chunk, with its
    source PDF name, page number, and the raw text stored in metadata.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict] = []    # parallel to index vectors

    # ── Build ──────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks: List[str],
        meta: List[Dict],
        client: EmbeddingClient,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> "FAISSStore":
        """
        Embed all chunks and build the FAISS index.

        Parameters
        ----------
        chunks     : List of text strings to index.
        meta       : Parallel metadata dicts (source_pdf, page_number, …).
        client     : Initialised EmbeddingClient.
        batch_size : Texts per API call.
        """
        logger.info("Building FAISS index for %d chunks …", len(chunks))
        vectors = embed_in_batches(client, chunks, batch_size=batch_size)

        dim = len(vectors[0])
        store = cls(dim)

        matrix = np.array(vectors, dtype=np.float32)
        store.index.add(matrix)

        for i, (chunk_text, m) in enumerate(zip(chunks, meta)):
            store.metadata.append({
                **m,
                "chunk_text": chunk_text,
                "chunk_index": i,
            })

        logger.info("FAISS index built: %d vectors, dim=%d", store.index.ntotal, dim)
        return store

    # ── Persistence ────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)

        logger.info("FAISS index saved → %s  metadata → %s", index_path, meta_path)

    @classmethod
    def load(
        cls,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> Optional["FAISSStore"]:
        """
        Load a previously saved FAISS index.

        Returns None if either file does not exist.
        """
        if not index_path.exists() or not meta_path.exists():
            logger.info("No existing FAISS index found at %s", index_path)
            return None

        try:
            with open(meta_path, "rb") as f:
                saved = pickle.load(f)

            store = cls(saved["dim"])
            store.metadata = saved["metadata"]
            store.index = faiss.read_index(str(index_path))
            logger.info(
                "FAISS index loaded: %d vectors, dim=%d",
                store.index.ntotal, store.dim,
            )
            return store
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return None

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[Dict, float]]:
        """
        Find the top-k most similar chunks for *query_vector*.

        Returns
        -------
        List of (metadata_dict, distance) pairs, sorted by ascending distance
        (lower = more similar).
        """
        if self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results: List[Tuple[Dict, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(dist)))

        return results


# ═════════════════════════════════════════════════════════════════════════════
# High-level Retriever
# ═════════════════════════════════════════════════════════════════════════════

class Retriever:
    """
    Orchestrates the full ingestion and retrieval pipeline.

    On init it either loads an existing FAISS index from disk or
    builds a new one from PDFs in the data/ directory.

    All embedding calls go through the HF Inference API — no local
    model downloads.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        hf_token: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.index_path = index_path
        self.meta_path = meta_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialise embedding client (validates HF_TOKEN immediately)
        self.embed_client = EmbeddingClient(token=hf_token)

        # Load or build the FAISS store
        self.store: Optional[FAISSStore] = FAISSStore.load(index_path, meta_path)
        if self.store is None:
            self.store = self._build_index()

    # ── Index building ─────────────────────────────────────────────────────

    def _build_index(self) -> FAISSStore:
        """Load PDFs → chunk → embed via API → build FAISS index."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pages = load_pdfs(self.data_dir)

        if not pages:
            raise FileNotFoundError(
                f"No PDFs found in '{self.data_dir}'. "
                "Upload welding PDF documents to the data/ folder."
            )

        all_chunks: List[str] = []
        all_meta: List[Dict] = []

        for page in pages:
            chunks = split_text(
                page["text"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            for chunk in chunks:
                all_chunks.append(chunk)
                all_meta.append({
                    "source_pdf": page["source_pdf"],
                    "page_number": page["page_number"],
                })

        logger.info(
            "Chunked %d pages → %d chunks (size=%d, overlap=%d)",
            len(pages), len(all_chunks),
            self.chunk_size, self.chunk_overlap,
        )

        store = FAISSStore.build(all_chunks, all_meta, self.embed_client)
        store.save(self.index_path, self.meta_path)
        return store

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query : User's natural-language question.

        Returns
        -------
        List of metadata dicts, each containing:
          chunk_text  : The passage text
          source_pdf  : Source filename
          page_number : Page in the source PDF
          distance    : L2 distance (lower = more relevant)
        """
        if self.store is None or self.store.index.ntotal == 0:
            logger.warning("FAISS index is empty — cannot retrieve.")
            return []

        try:
            query_vec = self.embed_client.embed_query(query)
        except EmbeddingAPIError as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

        results = self.store.search(query_vec, top_k=self.top_k)
        enriched = []
        for meta, dist in results:
            enriched.append({**meta, "distance": dist})

        logger.debug("Retrieved %d chunks for query: '%s'", len(enriched), query[:80])
        return enriched

    # ── Index refresh ──────────────────────────────────────────────────────

    def refresh_index(self) -> None:
        """Force rebuild the index from PDFs (e.g. after uploading new docs)."""
        logger.info("Refreshing FAISS index …")
        self.store = self._build_index()

    @property
    def document_count(self) -> int:
        """Number of chunks currently in the index."""
        return self.store.index.ntotal if self.store else 0
