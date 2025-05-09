"""Minimal Retrieval-Augmented Generation (RAG) demo for Eclipse manual."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

DB_DIR = Path("output/rag_db")
DEFAULT_CSV = Path("output/ECLIPSE_Keywords_dataset.csv")  # we reuse the answers as context chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"


# ---------------------------------------------------------------------------
# 1. Build or load vector index
# ---------------------------------------------------------------------------

def load_chunks(csv_path: Path) -> list[str]:
    """Return list of text chunks from the supplied CSV/XLSX file."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    chunks: list[str] = []

    if csv_path.suffix.lower() == ".xlsx":
        import pandas as pd
        df = pd.read_excel(csv_path)
        # prefer 'Description' column else fall back
        col = "Description" if "Description" in df.columns else df.columns[-1]
        chunks = df[col].astype(str).tolist()
    else:
        import csv
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        for enc in encodings_to_try:
            try:
                with csv_path.open(encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        chunk = (row.get("Description") or row.get("answer") or "").strip()
                        if chunk:
                            chunks.append(chunk)
                # If we reach here without Unicode errors and collected chunks, stop trying encodings
                if chunks:
                    break
            except UnicodeDecodeError:
                chunks = []  # reset if failed partially
                continue
        if not chunks:
            raise UnicodeDecodeError("Failed to decode CSV with common encodings. Please provide UTF-8/CP1252 file.")
    return chunks


def build_or_load_index(csv_path: Path) -> chromadb.Collection:
    """Embed all chunks from csv_path and persist them into a Chroma collection."""
    client = chromadb.PersistentClient(str(DB_DIR))
    collection = client.get_or_create_collection("eclipse_manual")

    chunks = load_chunks(csv_path)
    already = collection.count()
    if already < len(chunks):
        print(f"Indexing {len(chunks)-already} new chunks…")
        embedder = SentenceTransformer(EMBED_MODEL)
        new_chunks = chunks[already:]
        vectors = embedder.encode(new_chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        ids = [f"chunk_{i}" for i in range(already, len(chunks))]
        collection.add(ids=ids, documents=new_chunks, embeddings=vectors)
    else:
        print("Vector index already up-to-date ✔")

    return collection


# ---------------------------------------------------------------------------
# 2. RAG answer function
# ---------------------------------------------------------------------------

def rag_answer(question: str, collection: chromadb.Collection, k: int = 3) -> str:
    embedder = SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    context = "\n\n".join(res["documents"][0])

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    gen = pipeline("text2text-generation", model=model, tokenizer=tok, max_new_tokens=128)

    prompt = (
        "You are an expert Eclipse simulator assistant.\n"
        "Answer the question using only the context provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return gen(prompt, do_sample=False)[0]["generated_text"]


# ---------------------------------------------------------------------------
# 3. CLI demo entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Eclipse RAG assistant")
    parser.add_argument("-q", "--question", dest="question", help="Ask a single question and exit", type=str)
    parser.add_argument("--csv", dest="csv", type=str, default=str(DEFAULT_CSV), help="Path to dataset CSV/XLSX file")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    collection = build_or_load_index(csv_path)

    # non-interactive, single-question mode
    if args.question:
        print(rag_answer(args.question, collection))
        return

    # interactive mode
    print("\nType an Eclipse question. Empty line to quit.\n")
    while True:
        try:
            q = input("Ask Eclipse > ").strip()
            if not q:
                break
            answer = rag_answer(q, collection)
            print("\n" + textwrap.fill(answer, 100) + "\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
