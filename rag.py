import os
from typing import List, Dict, Any, Tuple
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pdf-rag-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
TOP_K = int(os.getenv("TOP_K", 5))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.25))

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generation model
llm_model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1
)

# Pinecone index
def ensure_index(dim: int = 384):
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(
            name=PINECONE_INDEX,
            dimension=dim,
            metric="cosine"
        )

def get_index():
    return pinecone.Index(PINECONE_INDEX)

# Embeddings
def embed_texts(texts: List[str]) -> List[List[float]]:
    return embed_model.encode(texts, convert_to_numpy=True).tolist()

# Retrieve top-k matches from Pinecone
def retrieve(query: str, namespace: str = None) -> List[Dict[str, Any]]:
    idx = get_index()
    q_emb = embed_texts([query])[0]
    kwargs = {"vector": q_emb, "top_k": TOP_K, "include_metadata": True}
    if namespace:
        kwargs["namespace"] = namespace
    res = idx.query(**kwargs)
    matches = res.get("matches", [])
    return [m for m in matches if m.get("score", 0.0) >= SCORE_THRESHOLD]

# Generate answer with LLM using retrieved contexts
def generate_answer_hf(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[Page {c['metadata']['page']}] {c['metadata']['text']}" for c in contexts
    )
    prompt = (
        "You are a strict RAG assistant. Only answer using the provided CONTEXT. "
        "If the answer is not present, say: "
        "\"I can only answer from the provided document, and I couldn't find this information.\"\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer:"
    )
    output = generator(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    return output.strip()

# Main answer function
def answer_question_hf(question: str, namespace: str = None) -> Tuple[str, List[Dict[str, Any]]]:
    contexts = retrieve(question, namespace=namespace)
    if not contexts:
        return "I can only answer from the provided document, and I couldn't find this information.", []
    answer = generate_answer_hf(question, contexts)
    return answer, contexts
