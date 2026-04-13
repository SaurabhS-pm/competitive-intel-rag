"""
Embedder module: chunks review text and produces local embeddings.

Runs BAAI/bge-small-en-v1.5 (384-dim) via ONNX Runtime directly.
This avoids fastembed (py-rust-stemmers has no cp314 wheel) and
sentence-transformers (torch has no cp314 wheel).

On first use the model files are downloaded from HuggingFace Hub (~130 MB)
and cached in the default HF cache (~/.cache/huggingface).
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REPO = "BAAI/bge-small-en-v1.5"
ONNX_FILE = "onnx/model.onnx"   # path inside the repo

# Lazy singletons
_tokenizer = None
_session = None


def _load_model():
    global _tokenizer, _session
    if _tokenizer is not None and _session is not None:
        return _tokenizer, _session

    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    import onnxruntime as ort

    logger.info(f"Loading tokenizer for '{MODEL_REPO}'...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    logger.info(f"Downloading ONNX model '{MODEL_REPO}/{ONNX_FILE}'...")
    onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename=ONNX_FILE)

    logger.info(f"Creating ONNX Runtime session from {onnx_path}...")
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    _session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info("Embedding model ready.")
    return _tokenizer, _session


def _mean_pool_normalize(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling over token embeddings, then L2-normalise (matches BGE training)."""
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    mean_pooled = summed / counts
    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True).clip(min=1e-9)
    return mean_pooled / norms


def _batch_embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    tokenizer, session = _load_model()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        # token_type_ids is optional — only pass if the model expects it
        input_names = {inp.name for inp in session.get_inputs()}
        if "token_type_ids" in input_names and "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = session.run(None, inputs)
        # outputs[0] shape: (batch, seq_len, hidden) — last hidden state
        token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"]
        embeddings = _mean_pool_normalize(token_embeddings, attention_mask)
        all_embeddings.append(embeddings)

    combined = np.concatenate(all_embeddings, axis=0)
    return combined.tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_review(text: str, max_chars: int = 500) -> list[str]:
    """
    Split a single review into chunks of at most max_chars characters,
    splitting on sentence boundaries where possible.
    Most app reviews are short (<300 chars), so this usually returns one chunk.
    """
    import re

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                for j in range(0, len(sentence), max_chars):
                    chunks.append(sentence[j : j + max_chars])
            else:
                current = sentence
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_chars]]


def _make_chunk_id(review_id: str, chunk_index: int, text: str) -> str:
    if review_id:
        return f"{review_id}__chunk{chunk_index}"
    digest = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"hash_{digest}__chunk{chunk_index}"


def embed_reviews(reviews: list[dict], max_chars: int = 500) -> list[dict]:
    """
    Chunk each review and compute embeddings.

    Returns a list of chunk dicts:
        {
            "id":        str,
            "text":      str,
            "embedding": list[float],   # 384-dim, L2-normalised
            "metadata":  { review_id, source, rating, date, title, app_name }
        }
    """
    if not reviews:
        return []

    all_chunks: list[dict] = []
    for review in reviews:
        raw_text = review.get("text", "").strip()
        if not raw_text:
            continue
        for idx, chunk_text in enumerate(chunk_review(raw_text, max_chars)):
            all_chunks.append(
                {
                    "id": _make_chunk_id(review.get("id", ""), idx, chunk_text),
                    "text": chunk_text,
                    "embedding": None,
                    "metadata": {
                        "review_id": review.get("id", ""),
                        "source": review.get("source", "unknown"),
                        "rating": review.get("rating"),
                        "date": review.get("date", ""),
                        "title": review.get("title", ""),
                        "app_name": review.get("app_name", ""),
                    },
                }
            )

    if not all_chunks:
        return []

    texts = [c["text"] for c in all_chunks]
    logger.info(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = _batch_embed(texts)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    logger.info(f"Embeddings done. {len(all_chunks)} chunks ready.")
    return all_chunks


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    return _batch_embed([query])[0]
