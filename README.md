#  eBay Policy Chatbot – RAG-based AI Assistant

This project is an AI-powered chatbot built using **Retrieval-Augmented Generation (RAG)** to answer user queries based on the provided policy document (*AI Training Document.pdf*). It uses document chunking, semantic embeddings, a vector database (FAISS), and a fine-tuned open-source LLM (`flan-t5-base`) served through a Streamlit interface with real-time streaming support.

---

##  Project Objective

To build a chatbot that can:
- Read and process a long policy document.
- Answer user questions using factual information from the document.
- Stream responses in real-time in a user-friendly Streamlit interface.
- Provide the source references used in each response.

---

## Folder Structure

AMLGo_AI_Task/
│
├── app.py ← Streamlit UI (main app)
├── requirements.txt ← All dependencies
├── README.md ← You're reading it now!
│
├── /data ← Raw policy document
├── /chunks ← Chunked and cleaned document pieces
├── /vectordb ← FAISS vector index
├── /notebooks ← Optional: preprocessing/evaluation
└── /src
└── rag_pipeline.py ← RAG pipeline: retriever + generator


---

##  Tech Stack

| Component           | Tool / Model                         |
|--------------------|--------------------------------------|
| Embedding Model     | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM (Generator)     | `google/flan-t5-base`               |
| Vector Store        | FAISS                               |
| UI Framework        | Streamlit                           |
| Chunking Strategy   | Sentence-aware splitting (100–300 words) |

---

## How It Works

1. **Preprocessing**: Load and clean the input document.
2. **Chunking**: Split into sentence-based 100–300 word chunks.
3. **Embedding**: Generate vector representations using MiniLM.
4. **Vector Store**: Store in FAISS for fast semantic retrieval.
5. **Retrieval**: At query time, relevant chunks are retrieved.
6. **Generation**: Inject chunks into a prompt and generate a factual answer using `flan-t5-base`.
7. **Streaming**: Stream the response word-by-word on the frontend.
8. **Source Reveal**: Display source passages used in generation.

---

##  How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ebay-policy-rag-chatbot.git
cd ebay-policy-rag-chatbot
