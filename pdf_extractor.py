"""
pdf_extractor.py  —  Member A
Responsibilities:
  - Iterate over a folder and extract text from every PDF
  - Clean the raw text
  - Store extracted text in a SQLite database
  - Split text into chunks (500 chars, 100 overlap) for downstream use
"""

import re
import sqlite3
import argparse
from pathlib import Path

from PyPDF2 import PdfReader


# ─── Database ────────────────────────────────────────────────────────────────

DB_PATH = "pdf_data.db"


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the tables exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT    NOT NULL,
            page_num  INTEGER NOT NULL,
            raw_text  TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER DEFAULT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


# ─── Extraction ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic cleaning:
      - Collapse 3+ consecutive newlines into two (preserve paragraph breaks)
      - Remove non-printable / control characters (keep tabs and newlines)
      - Strip leading/trailing whitespace per line
    """
    # Remove non-printable characters except \n and \t
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", " ", text)
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a single PDF, page by page.
    Returns a list of dicts: [{filename, page_num, raw_text}, ...]
    """
    reader = PdfReader(pdf_path)
    filename = Path(pdf_path).name
    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:                        # skip blank pages
            pages.append({
                "filename": filename,
                "page_num": i + 1,
                "raw_text": cleaned,
            })
    return pages


def extract_text_from_folder(folder_path: str) -> list[dict]:
    """
    Recursively find every PDF in *folder_path* and extract text.
    Returns combined list of page dicts from all PDFs.
    """
    folder = Path(folder_path)
    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'")

    all_pages = []
    for pdf_path in pdf_files:
        print(f"  Extracting: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        all_pages.extend(pages)
        print(f"    → {len(pages)} page(s) extracted")

    return all_pages


# ─── Storage ─────────────────────────────────────────────────────────────────

def store_pages(conn: sqlite3.Connection, pages: list[dict]) -> list[int]:
    """Insert page records into the `documents` table; return inserted row IDs."""
    doc_ids = []
    for page in pages:
        cursor = conn.execute(
            "INSERT INTO documents (filename, page_num, raw_text) VALUES (?, ?, ?)",
            (page["filename"], page["page_num"], page["raw_text"]),
        )
        doc_ids.append(cursor.lastrowid)
    conn.commit()
    return doc_ids


def store_chunks(conn: sqlite3.Connection, doc_id: int, chunks: list[str]) -> None:
    """Insert chunk records linked to a document row."""
    conn.executemany(
        "INSERT INTO chunks (doc_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        [(doc_id, i, chunk) for i, chunk in enumerate(chunks)],
    )
    conn.commit()


# ─── Chunking ─────────────────────────────────────────────────────────────────

def get_text_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Design decisions:
      - Split on "\n" boundaries first so chunks stay semantically coherent.
      - chunk_size=500: matches the lab specification.
      - chunk_overlap=100: 20% overlap prevents retrieval misses at boundaries.

    Implemented without langchain_text_splitters to avoid the NumPy 2.x /
    PyTorch compatibility issue triggered by that package's __init__.py.
    Behaviour is equivalent to CharacterTextSplitter(separator="\n").
    """
    # Split on newlines, then greedily re-join lines into chunks ≤ chunk_size,
    # keeping overlap between consecutive chunks.
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the newline we'll re-add
        if current_len + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            # Keep trailing lines whose total length ≤ chunk_overlap
            overlap: list[str] = []
            overlap_len = 0
            for l in reversed(current):
                if overlap_len + len(l) + 1 <= chunk_overlap:
                    overlap.insert(0, l)
                    overlap_len += len(l) + 1
                else:
                    break
            current = overlap
            current_len = overlap_len
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return [c.strip() for c in chunks if c.strip()]


def get_all_chunks_from_db(conn: sqlite3.Connection) -> list[str]:
    """Load every chunk text from the database (used by vectorstore_builder)."""
    rows = conn.execute("SELECT chunk_text FROM chunks ORDER BY id").fetchall()
    return [row[0] for row in rows]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(folder_path: str, db_path: str = DB_PATH) -> list[str]:
    """
    Full pipeline:
      1. Init DB
      2. Extract text from all PDFs in folder
      3. Store pages in DB
      4. Chunk each page's text and store chunks in DB
      5. Return all chunks (for immediate use by vectorstore_builder)
    """
    print(f"\n[1/4] Initialising database at '{db_path}' ...")
    conn = init_db(db_path)

    # Guard against duplicate runs: warn if data already exists
    existing = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if existing > 0:
        print(f"  WARNING: database already contains {existing} document row(s).")
        answer = input("  Re-run and append duplicates? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborted. Load existing chunks from DB instead.")
            all_chunks = get_all_chunks_from_db(conn)
            conn.close()
            return all_chunks

    try:
        print(f"[2/4] Extracting text from PDFs in '{folder_path}' ...")
        pages = extract_text_from_folder(folder_path)
        print(f"      Total pages extracted: {len(pages)}")

        print("[3/4] Storing pages in database ...")
        store_pages(conn, pages)

        # Concatenate ALL page text before chunking (matches reference app.py).
        # This ensures chunks can span page boundaries and overlap works correctly.
        print("[4/4] Chunking combined text and storing chunks ...")
        full_text = "\n".join(page["raw_text"] for page in pages)
        all_chunks = get_text_chunks(full_text)

        conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
            [(0, i, chunk) for i, chunk in enumerate(all_chunks)],
        )
        conn.commit()

        print(f"\n  Done — {len(all_chunks)} chunks stored in '{db_path}'")
    finally:
        conn.close()

    return all_chunks


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF text and store in SQLite.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Path to the folder containing PDFs (default: current directory)",
    )
    parser.add_argument("--db", default=DB_PATH, help=f"SQLite DB path (default: {DB_PATH})")
    args = parser.parse_args()

    chunks = run_pipeline(args.folder, args.db)
    if chunks:
        print(f"\nSample chunk (first):\n{'-'*40}\n{chunks[0][:300]}\n{'-'*40}")
    else:
        print("\nNo chunks produced.")
