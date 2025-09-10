import os, argparse, math, uuid
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from utils.pdf_utils import extract_pdf_text_by_page
from utils.text_splitter import chunk_text
from rag import ensure_index, embed_texts, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

def upsert_chunks(pdf_path: str, namespace: str = None, batch_size: int = 100):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index()
    index = pc.Index(PINECONE_INDEX)

    pages = extract_pdf_text_by_page(pdf_path)
    vectors = []
    ids = []
    metadatas = []
    texts = []

    # Build chunks with metadata
    for page_num, page_text in pages:
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            vid = str(uuid.uuid4())
            meta = {"page": page_num, "chunk": i, "source": "PDF", "text": chunk}
            ids.append(vid)
            metadatas.append(meta)
            texts.append(chunk)

    print(f"Prepared {len(texts)} chunks. Embedding...")
    # Embed in batches
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        embs = embed_texts(batch)
        upserts = []
        for j, emb in enumerate(embs):
            idx = start + j
            upserts.append({"id": ids[idx], "values": emb, "metadata": metadatas[idx]})
        if namespace:
            index.upsert(vectors=upserts, namespace=namespace)  # type: ignore
        else:
            index.upsert(vectors=upserts)  # type: ignore
        print(f"Upserted {end} / {len(texts)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest")
    parser.add_argument("--namespace", default=None, help="Optional namespace (e.g. doc name)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--create-index", action="store_true", help="Create the Pinecone index if missing")
    args = parser.parse_args()

    if args.create_index:
        ensure_index()

    upsert_chunks(args.pdf, namespace=args.namespace, batch_size=args.batch_size)
    print("Ingestion complete.")
