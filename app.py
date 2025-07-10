import streamlit as st
from src.rag_pipeline import RAGPipeline
import time
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="eBay Policy Bot",
    page_icon="📜",
    layout="centered"
)

# Load Model (cached)
@st.cache_resource
def load_model():
    return RAGPipeline()

rag = load_model()

# Title and Info
st.title("eBay User Agreement Assistant")
st.caption(f"Knowledge source: {Path('data/AI Training Document.pdf').name}")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about eBay's policies!"}
    ]

# Display Previous Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input("Your question"):
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response = st.empty()
        full_answer = ""

        try:
            result = rag.query(prompt)
            answer = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"❌ An error occurred: {str(e)}"
            sources = []

        # Token-by-token stream
        for word in answer.split():
            full_answer += word + " "
            time.sleep(0.05)
            response.markdown(full_answer + "▌")
        response.markdown(full_answer)

        # Show sources
        if sources:
            with st.expander("📄 Source References"):
                for i, doc in enumerate(sources, 1):
                    st.caption(f"Source {i}:")
                    st.text(doc.page_content[:500] + "...")

    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer
    })
