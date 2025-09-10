# RAG Pinecone Bot

A Retrieval-Augmented Generation (RAG) chatbot using **Falcon-7B-Instruct** and **Pinecone vector database**.  
This bot can ingest PDF documents, embed the content, store it in Pinecone, and answer questions using LLM-generated responses based on the document context.

---

## Features

- Ingest PDF documents and split them into chunks.
- Generate embeddings using `all-MiniLM-L6-v2`.
- Store embeddings in **Pinecone** (serverless vector database).
- Retrieve top-k relevant chunks for a query.
- Generate answers with **Falcon-7B-Instruct** (4-bit quantized for lower memory usage).
- Fully modular: separate `rag.py` for embeddings, retrieval, and LLM generation.

---

## Requirements

- Python 3.10+  
- Virtual environment recommended

### Install dependencies

```bash
pip install -r requirements.txt
