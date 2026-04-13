"""
Retriever module: stores embedded review chunks in ChromaDB and retrieves
the most relevant ones for a given query.

ChromaDB persists data to disk at ./chroma_db by default.
Each app gets its own collection (namespace) so you can query multiple
competitors without conflicts.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"


def _get_client() -> chromadb.PersistentClient:
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _collection_name(app_name: str) -> str:
    """Normalise app name to a valid ChromaDB collection name."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", app_name.lower().strip())
    # ChromaDB requires 3-63 chars, starting/ending with alphanumeric
    name = name.strip("_-")
    if len(name) < 3:
        name = name + "_app"
    return name[:63]


def get_or_create_collection(app_name: str):
    """Return the ChromaDB collection for the given app (create if missing)."""
    client = _get_client()
    col_name = _collection_name(app_name)
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(app_name: str, chunks: list[dict]) -> int:
    """
    Upsert a list of embedded chunk dicts into the ChromaDB collection.
    Chunks that already exist (same ID) are overwritten.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    collection = get_or_create_collection(app_name)

    ids = [c["id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {k: (v if v is not None else "") for k, v in c["metadata"].items()}
        # ChromaDB metadata values must be str/int/float/bool
        if meta.get("rating") == "":
            meta["rating"] = 0.0
        elif meta.get("rating") is not None:
            meta["rating"] = float(meta["rating"])
        metadatas.append(meta)

    # Upsert in batches of 500 to stay within ChromaDB limits
    batch_size = 500
    stored = 0
    for i in range(0, len(ids), batch_size):
        sl = slice(i, i + batch_size)
        collection.upsert(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )
        stored += len(ids[sl])

    logger.info(f"Stored {stored} chunks for '{app_name}' in ChromaDB.")
    return stored


def collection_count(app_name: str) -> int:
    """Return the number of chunks currently stored for an app."""
    try:
        col = get_or_create_collection(app_name)
        return col.count()
    except Exception:
        return 0


def retrieve(
    app_name: str,
    query_embedding: list[float],
    top_k: int = 15,
    feature_area: Optional[str] = None,
) -> list[dict]:
    """
    Query ChromaDB for the top_k most relevant chunks.
    Optionally filter by feature_area (stored in metadata or present in document text).

    Returns a list of dicts:
        {
            "id":       str,
            "text":     str,
            "distance": float,   # cosine distance (lower = more similar)
            "metadata": dict,
        }
    """
    collection = get_or_create_collection(app_name)
    count = collection.count()
    if count == 0:
        logger.warning(f"Collection for '{app_name}' is empty.")
        return []

    actual_k = min(top_k, count)

    where_filter = None
    if feature_area and feature_area.strip():
        # ChromaDB's $contains on documents works via where_document
        pass  # handled below via where_document

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": actual_k,
        "include": ["documents", "distances", "metadatas"],
    }
    if feature_area and feature_area.strip():
        query_kwargs["where_document"] = {"$contains": feature_area.lower()}

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        # where_document may not match anything — fall back without filter
        logger.warning(f"Filtered query failed ({e}), retrying without filter.")
        query_kwargs.pop("where_document", None)
        results = collection.query(**query_kwargs)

    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    for rid, doc, dist, meta in zip(ids, docs, distances, metas):
        hits.append(
            {
                "id": rid,
                "text": doc,
                "distance": dist,
                "metadata": meta,
            }
        )

    return hits


def delete_collection(app_name: str) -> bool:
    """Delete all stored reviews for an app. Returns True on success."""
    try:
        client = _get_client()
        client.delete_collection(_collection_name(app_name))
        logger.info(f"Deleted collection for '{app_name}'.")
        return True
    except Exception as e:
        logger.warning(f"Could not delete collection for '{app_name}': {e}")
        return False
