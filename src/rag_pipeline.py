from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class RAGPipeline:
    def __init__(self):
        # 1. Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Load Vector Store
        self.vectorstore = FAISS.load_local(
            "vectordb/faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 3. Load LLM
        self.llm = self._load_llm()

        # 4. Setup Prompt
        self.prompt = PromptTemplate(
            template="""You are an assistant answering questions based only on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer concisely:""",
            input_variables=["context", "question"]
        )

        # 5. Create QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def _load_llm(self):
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3
        )

        return HuggingFacePipeline(pipeline=pipe)

    def query(self, question):
        try:
            result = self.qa_chain({"query": question})
            print("🔍 QA Chain Output:", result)  # Optional debug output
            return {
                "answer": result.get("result", "[No answer generated]"),
                "sources": result.get("source_documents", [])
            }
        except Exception as e:
            print("❌ Error in RAGPipeline.query():", str(e))
            return {
                "answer": "[Error occurred while generating the answer.]",
                "sources": []
            }
