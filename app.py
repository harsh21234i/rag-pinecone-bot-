import os, streamlit as st
from dotenv import load_dotenv
from rag import answer_question

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="centered")

st.title("📄 RAG PDF Chatbot (Pinecone)")
st.caption("Answers strictly from the provided PDF. Shows page citations.")

with st.sidebar:
    st.header("Settings")
    namespace = st.text_input("Namespace (optional)", value="")
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=int(os.getenv("TOP_K", "6")))
    score_threshold = st.slider("Score Threshold", 0.0, 1.0, float(os.getenv("SCORE_THRESHOLD", "0.25")), 0.01)
    st.markdown("---")
    st.markdown("**Tip:** Run `python ingest.py --pdf ./data/your.pdf --create-index` before chatting.")

st.markdown("### Ask a question")
question = st.text_input("Your question", placeholder="e.g., What is the main objective described on page 2?")

if st.button("Ask") or (question and st.session_state.get("auto_run")):
    with st.spinner("Retrieving and generating..."):
        # Override env for live sliders
        os.environ["TOP_K"] = str(top_k)
        os.environ["SCORE_THRESHOLD"] = str(score_threshold)
        ans, ctxs = answer_question(question, namespace=namespace if namespace else None)

    st.markdown("## Answer")
    st.write(ans)

    if ctxs:
        st.markdown("### Sources")
        for i, m in enumerate(ctxs, start=1):
            meta = m.get("metadata", {})
            page = meta.get("page")
            snippet = meta.get("text", "")[:400].strip().replace("\n", " ")
            score = m.get("score", 0.0)
            with st.expander(f"Match {i}: Page {page} (score={score:.3f})"):
                st.write(snippet)
    else:
        st.info("No relevant sources found above the threshold. The bot is configured to answer only from the document.")
