import streamlit as st
from rag import answer_question_hf

st.title(" PDF RAG Chatbot (Hugging Face)")

namespace = st.text_input("Namespace (for Pinecone):", "default")
question = st.text_input("Ask a question about the PDF:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        answer, sources = answer_question_hf(question, namespace=namespace)
        st.markdown(f"**Answer:** {answer}")
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- Page {s['metadata']['page']}")
