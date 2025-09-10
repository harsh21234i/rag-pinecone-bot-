import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pdf-rag-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "6"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index(dim: int = 3072):
    if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:  # type: ignore
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

def get_index():
    return pc.Index(PINECONE_INDEX)

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    citations = []
    context_block = []
    seen = set()
    for c in contexts:
        page = c.get("metadata", {}).get("page")
        txt = c.get("metadata", {}).get("text", "")[:1200]
        context_block.append(f"[Page {page}] {txt}")
        if page not in seen:
            citations.append(f"PDF, page {page}")
            seen.add(page)

    sys_prompt = (
        "You are a strict RAG assistant. Only answer using the provided CONTEXT. "
        "If the answer is not present in the CONTEXT, say: "
        "\"I can only answer from the provided document, and I couldn't find this information.\" "
        "Always include citations as (Source: PDF, page X[, page Y...])."
    )
    user_prompt = (
        "QUESTION:\n" + question.strip() + "\n\n"
        "CONTEXT (excerpts with pages):\n" + "\n\n".join(context_block) + "\n\n"
        "Answer succinctly with citations."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    messages = build_prompt(question, contexts)
    resp = client.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages, temperature=0.1)
    return resp.choices[0].message.content.strip()

def retrieve(query: str, namespace: str = None) -> List[Dict[str, Any]]:
    index = get_index()
    q_emb = embed_texts([query])[0]
    kwargs = {"vector": q_emb, "top_k": TOP_K, "include_metadata": True}
    if namespace:
        kwargs["namespace"] = namespace
    res = index.query(**kwargs)  # type: ignore
    matches = res.get("matches", [])
    # Normalize score guardrail: Pinecone returns 'score' as cosine similarity.
    filtered = [m for m in matches if m.get("score", 0.0) >= SCORE_THRESHOLD]
    return filtered

def answer_question(question: str, namespace: str = None) -> Tuple[str, List[Dict[str, Any]]]:
    contexts = retrieve(question, namespace=namespace)
    if not contexts:
        return ("I can only answer from the provided document, and I couldn't find this information.", [])
    answer = generate_answer(question, contexts)
    return answer, contexts
