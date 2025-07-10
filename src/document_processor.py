import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_document():
    # Verify document exists
    pdf_path = os.path.join("data", "AI Training Document.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing PDF at: {os.path.abspath(pdf_path)}")
    
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Clean text
    text = "\n".join([p.page_content.strip() for p in pages])
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    
    # Save chunks
    os.makedirs("chunks", exist_ok=True)
    with open(os.path.join("chunks", "doc_chunks.txt"), "w", encoding="utf-8") as f:
        f.write("\n---CHUNK---\n".join(chunks))
    
    # Create vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(os.path.join("vectordb", "faiss_index"))
    
    print(f"✅ Processed {len(chunks)} chunks. VectorDB saved to {os.path.abspath('vectordb')}")

if __name__ == "__main__":
    process_document()