"""
vectorstore_builder.py  —  Member A
Responsibilities:
  - Part 1: Build a FAISS vector store using OpenAI Embeddings
  - Part 2: Build a FAISS vector store using a local open-source model
             (BAAI/bge-small-en-v1.5 — outperforms all-MiniLM-L6-v2 on MTEB)
  - Expose save / load helpers so Members B and C can reuse the indices

Public interface (used by Member B's conversation_chain.py):
    vectorstore = load_vectorstore(path)          # load saved index
    vectorstore = build_and_save_openai(chunks)   # Part 1
    vectorstore = build_and_save_opensource(chunks)  # Part 2
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# ──────────────────────────────────────────────────────────────────────────────
# Embedding model selection
# ──────────────────────────────────────────────────────────────────────────────
#
# Part 1  → OpenAI text-embedding-ada-002
# Part 2  → BAAI/bge-small-en-v1.5  (open-source, runs fully locally)


OPENAI_INDEX_DIR   = "faiss_index_openai"
OPENSOURCE_INDEX_DIR = "faiss_index_opensource"

OSS_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ─── Part 1 — OpenAI Embeddings ───────────────────────────────────────────────

def build_and_save_openai(chunks: list[str], save_dir: str = OPENAI_INDEX_DIR) -> object:
    """
    Embed *chunks* with OpenAI text-embedding-ada-002, build a FAISS index,
    and persist it to *save_dir*.

    Requires OPENAI_API_KEY in the environment (via .env or shell).
    """
    from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai

    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to a .env file or set it in your shell."
        )

    print(f"[OpenAI] Embedding {len(chunks)} chunks with text-embedding-ada-002 ...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"[OpenAI] Index saved to '{save_dir}/'")
    return vectorstore


def load_vectorstore_openai(load_dir: str = OPENAI_INDEX_DIR) -> object:
    """Load a previously saved OpenAI FAISS index from disk."""
    from langchain_openai import OpenAIEmbeddings

    load_dotenv(override=True)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        load_dir, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[OpenAI] Index loaded from '{load_dir}/'")
    return vectorstore


# ─── Part 2 — Open-Source Embeddings (BAAI/bge-small-en-v1.5) ────────────────

def build_and_save_opensource(
    chunks: list[str], save_dir: str = OPENSOURCE_INDEX_DIR
) -> object:
    """
    Embed *chunks* with BAAI/bge-small-en-v1.5 (runs 100% locally, no API key),
    build a FAISS index, and persist it to *save_dir*.

    BGE models expect queries to be prefixed with
    "Represent this sentence for searching relevant passages: "
    for retrieval tasks; the HuggingFaceBgeEmbeddings wrapper handles this
    automatically via encode_kwargs and query_instruction.
    """
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    print(f"[OSS] Loading '{OSS_MODEL_NAME}' (downloads once, ~130 MB) ...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=OSS_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        query_instruction="Represent this sentence for searching relevant passages: ",
    )

    print(f"[OSS] Embedding {len(chunks)} chunks ...")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"[OSS] Index saved to '{save_dir}/'")
    return vectorstore


def load_vectorstore_opensource(load_dir: str = OPENSOURCE_INDEX_DIR) -> object:
    """Load a previously saved open-source FAISS index from disk."""
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=OSS_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages: ",
    )
    vectorstore = FAISS.load_local(
        load_dir, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[OSS] Index loaded from '{load_dir}/'")
    return vectorstore


# ─── Generic load helper (used by Member B) ──────────────────────────────────

def load_vectorstore(path: str, use_openai: bool = True) -> object:
    """
    Convenience function for Member B / C.
      load_vectorstore("faiss_index_openai",    use_openai=True)
      load_vectorstore("faiss_index_opensource", use_openai=False)
    """
    if use_openai:
        return load_vectorstore_openai(path)
    return load_vectorstore_opensource(path)


# ─── Comparison utility ───────────────────────────────────────────────────────

def compare_embeddings(query: str, chunks: list[str], top_k: int = 3) -> None:
    """
    Run the same query against both embedding models and print the top-k
    retrieved chunks side-by-side.  Useful for demonstrating Part 2.
    """
    print("\n" + "=" * 60)
    print(f"Query: {query!r}")
    print("=" * 60)

    loaders = [
        ("OpenAI (ada-002)",              load_vectorstore_openai,    build_and_save_openai,    OPENAI_INDEX_DIR),
        (f"Open-Source ({OSS_MODEL_NAME})", load_vectorstore_opensource, build_and_save_opensource, OPENSOURCE_INDEX_DIR),
    ]
    for label, load_fn, build_fn, index_dir in loaders:
        try:
            # Use existing index if available; build only if needed
            if Path(index_dir).exists():
                vs = load_fn(index_dir)
            else:
                vs = build_fn(chunks)
            results = vs.similarity_search(query, k=top_k)
            print(f"\n── {label} ──")
            for i, doc in enumerate(results, 1):
                snippet = doc.page_content[:200].replace("\n", " ")
                print(f"  [{i}] {snippet}...")
        except Exception as exc:
            print(f"\n── {label} ── ERROR: {exc}")

    print("=" * 60 + "\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sqlite3
    from pdf_extractor import get_all_chunks_from_db, DB_PATH

    parser = argparse.ArgumentParser(
        description="Build FAISS vector stores from the extracted PDF chunks."
    )
    parser.add_argument(
        "--mode",
        choices=["openai", "opensource", "both", "compare"],
        default="both",
        help=(
            "openai      → Part 1 only\n"
            "opensource  → Part 2 only\n"
            "both        → build both indices (default)\n"
            "compare     → build both and run a sample query comparison"
        ),
    )
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    parser.add_argument(
        "--query",
        default="How do I install the software?",
        help="Sample query used in compare mode",
    )
    args = parser.parse_args()

    # Load chunks from the database populated by pdf_extractor.py
    conn = sqlite3.connect(args.db)
    chunks = get_all_chunks_from_db(conn)
    conn.close()

    if not chunks:
        raise RuntimeError(
            f"No chunks found in '{args.db}'. "
            "Run pdf_extractor.py first to populate the database."
        )
    print(f"Loaded {len(chunks)} chunks from '{args.db}'")

    if args.mode == "openai":
        build_and_save_openai(chunks)
    elif args.mode == "opensource":
        build_and_save_opensource(chunks)
    elif args.mode == "both":
        build_and_save_openai(chunks)
        build_and_save_opensource(chunks)
    elif args.mode == "compare":
        compare_embeddings(args.query, chunks)
