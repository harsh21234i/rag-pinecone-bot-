import os
import sys
from dotenv import load_dotenv
from rag import ensure_index, get_index, embed_texts
from pdf_utils import extract_pdf_text_by_page
from text_splitter import chunk_text

load_dotenv()

def ingest(pdf_path: str, namespace: str = "default"):
    ensure_index()
    index = get_index()

    print(f"[INFO] Extracting text from {pdf_path} ...")
    pages = extract_pdf_text_by_page(pdf_path)  # Returns list of tuples (page_num, text)

    vectors = []
    for page_num, text in pages:
        chunks = chunk_text(text)
        embeds = embed_texts(chunks)
        for i, emb in enumerate(embeds):
            vectors.append({
                "id": f"{os.path.basename(pdf_path)}-p{page_num}-c{i}",
                "values": emb,
                "metadata": {"page": page_num, "text": chunks[i]}
            })

    if not vectors:
        print("[WARNING] No vectors to upload!")
        return

    print(f"[INFO] Uploading {len(vectors)} vectors to Pinecone ...")
    index.upsert(vectors=vectors, namespace=namespace)
    print("[INFO] Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path> [namespace]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    ns = sys.argv[2] if len(sys.argv) > 2 else "default"
    ingest(pdf_path, ns)
