"""
retriever.py - Loads knowledge base, embeds it, and retrieves relevant docs via FAISS.
"""

import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load embedding model once at startup
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_kb(path: str = None) -> list[dict]:
    """Parse the knowledge base into a list of doc dicts."""
    # Try multiple possible locations for docs.txt
    if path is None:
        candidates = [
            "kb/docs.txt",
            "docs.txt",
            os.path.join(os.path.dirname(__file__), "kb", "docs.txt"),
            os.path.join(os.path.dirname(__file__), "docs.txt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                "docs.txt not found. Place it at kb/docs.txt or docs.txt in your project root."
            )

    with open(path, "r") as f:
        raw = f.read()

    # Split by [DOC_N] markers
    sections = raw.split("[DOC_")
    docs = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        if len(lines) < 2:
            continue
        doc_id = lines[0].strip().rstrip("]")
        content = lines[1].strip()

        title_line = content.split("\n")[0] if content else ""
        category_line = content.split("\n")[1] if "\n" in content else ""

        title = title_line.replace("Title:", "").strip()
        category = category_line.replace("Category:", "").strip()

        docs.append({
            "id": f"DOC_{doc_id}",
            "title": title,
            "category": category,
            "content": content
        })

    logger.info(f"Loaded {len(docs)} documents from knowledge base at '{path}'.")
    return docs


def build_index(docs: list[dict]):
    """Embed all docs and build a FAISS index."""
    texts = [d["content"] for d in docs]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    logger.info(f"FAISS index built with {index.ntotal} vectors (dim={dimension}).")
    return index, embeddings


# --- Module-level initialization ---
KB_DOCS = load_kb()
FAISS_INDEX, DOC_EMBEDDINGS = build_index(KB_DOCS)


# Persona-aware category preferences
PERSONA_CATEGORY_BOOST = {
    "technical_expert": ["technical"],
    "frustrated_user": ["troubleshooting"],
    "business_executive": ["business", "billing"],
}


def retrieve(query: str, persona: str = None, top_k: int = 3) -> list[dict]:
    """
    Retrieve top_k most relevant KB docs for the query.
    Optionally re-ranks based on persona preferences.
    """
    query_embedding = EMBED_MODEL.encode([query])
    distances, indices = FAISS_INDEX.search(
        np.array(query_embedding, dtype=np.float32), top_k + 2
    )

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(KB_DOCS):
            doc = KB_DOCS[idx].copy()
            doc["score"] = float(dist)
            candidates.append(doc)

    if persona and persona in PERSONA_CATEGORY_BOOST:
        preferred = PERSONA_CATEGORY_BOOST[persona]
        for doc in candidates:
            if doc["category"] in preferred:
                doc["score"] *= 0.7

        candidates.sort(key=lambda d: d["score"])

    result = candidates[:top_k]
    logger.info(f"Retrieved {len(result)} docs for query (persona={persona}): {[d['id'] for d in result]}")
    return result
