import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Load keys ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pdf-rag-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Hugging Face embedding model ---
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(HF_EMBED_MODEL)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Ensure Pinecone index exists ---
def ensure_index(dim: int):
    if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:  # type: ignore
        print(f"[INFO] Creating Pinecone index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"[INFO] Pinecone index '{PINECONE_INDEX}' already exists.")

# --- Test Hugging Face embeddings ---
def test_hf():
    try:
        emb = embedder.encode(["Hello world"])[0]
        print("[INFO] Hugging Face embeddings OK. Sample embedding length:", len(emb))
    except Exception as e:
        print("[ERROR] Hugging Face embedding test failed:", str(e))

if __name__ == "__main__":
    # get dimension from embedding model
    dim = embedder.get_sentence_embedding_dimension()
    ensure_index(dim)
    test_hf()
