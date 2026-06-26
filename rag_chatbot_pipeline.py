import os
import pypdf
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ClassBasedRAGChatbot:
    """
    A unified, class-based RAG chatbot that implements the complete RAG pipeline:
    1. Ingestion: PDF parsing (pypdf) and chunking (RecursiveCharacterTextSplitter)
    2. Embeddings: sentence-transformers/all-MiniLM-L6-v2 via HuggingFaceEmbeddings
    3. Storage: Persistent Chroma Vector DB (with duplicate prevention)
    4. Two-Stage Retrieval: Initial similarity search followed by CrossEncoder reranking
    5. Generation: Gemini API response generation grounded on retrieved context
    """

    def __init__(
        self,
        db_dir="./rag_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="ms-marco-MiniLM-L-12-v2",
        llm_model="gemini-2.5-flash"
    ):
        self.db_dir = db_dir

        # 1. Initialize Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. Please set it before initializing."
            )
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(llm_model)

        # 2. Setup Embeddings
        print(f"Loading embedding model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 3. Setup Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            keep_separator=True
        )

        # 4. Setup Reranker (Cross-Encoder)
        print(f"Loading reranker model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device="cpu")

        # 5. Initialize/Load Vector Store (Chroma)
        print(f"Connecting to Chroma DB at: {self.db_dir}...")
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.db_dir
        )
        print("[OK] RAG Chatbot initialized successfully.")

    def ingest_pdf(self, pdf_path):
        """
        Parses a PDF file, splits it into chunks, calculates embeddings, and updates
        the vector database. Includes duplicate prevention.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        filename = os.path.basename(pdf_path)
        print(f"\n[Ingest] Extracting text from PDF: {pdf_path}...")

        reader = pypdf.PdfReader(pdf_path)
        documents = []
        ids = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1
                    }
                )
                documents.append(doc)
                ids.append(f"{filename}_p{page_idx + 1}_c{chunk_idx}")

        if not documents:
            print("[Ingest] Warning: No text could be extracted from this document.")
            return

        print(f"[Ingest] Created {len(documents)} chunks from {filename}.")

        # Prevent duplicates
        existing = self.vector_store.get(where={"source": filename})
        if existing and existing.get("ids"):
            old_ids = existing["ids"]
            self.vector_store.delete(ids=old_ids)
            print(f"[Ingest] Removed {len(old_ids)} existing chunks for {filename}.")

        self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"[Ingest] Successfully ingested & indexed '{filename}'.")

    def retrieve_and_rerank(self, query, initial_k=15, final_k=3, filter_dict=None):
        """
        Two-stage retrieval: embedding similarity search + CrossEncoder reranking.
        """
        print(f"\n[Search] Performing initial semantic search (k={initial_k})...")
        search_kwargs = {"k": initial_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        candidates = self.vector_store.similarity_search(query, **search_kwargs)
        if not candidates:
            print("[Search] No candidate documents retrieved.")
            return []

        print(f"[Search] Reranking {len(candidates)} candidates using Cross-Encoder...")
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        candidate_scores = list(zip(candidates, scores))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = candidate_scores[:final_k]

        return [{"document": doc, "score": float(score)} for doc, score in top_results]

    def generate_answer(self, question, context_results):
        """
        Constructs grounded context and calls the Gemini LLM.
        """
        if not context_results:
            return "The vector database does not contain any relevant context to answer this question."

        context_blocks = []
        for item in context_results:
            doc = item["document"]
            score = item["score"]
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_blocks.append(
                f"[Source: {source} (Page {page}), Relevance Score: {score:.4f}]\nContent: {doc.page_content}"
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are a reliable AI assistant for Mercedes-Benz. Your answers must be based strictly on the provided Context.

Context:
{context_str}

Question:
{question}

Instructions:
1. Ground your answer ONLY on the provided Context.
2. If the answer cannot be found in the Context, respond exactly with: "I cannot find the answer in the provided documents."
3. Do NOT make up, extrapolate, or assume any information outside the Context.
4. Keep your answer clear, factual, and concise. Add page citations where applicable.
"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"

    def ask(self, question, initial_k=15, final_k=3, filter_dict=None):
        """
        Unified endpoint: retrieval → reranking → generation.
        """
        context_results = self.retrieve_and_rerank(
            query=question,
            initial_k=initial_k,
            final_k=final_k,
            filter_dict=filter_dict
        )
        answer = self.generate_answer(question, context_results)
        return {"answer": answer, "sources": context_results}


# --- Demonstration Runner ---
if __name__ == "__main__":
    print("=== Starting Mercedes-Benz RAG Chatbot Runner ===")

    if "GEMINI_API_KEY" not in os.environ:
        try:
            import tomllib
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            secrets_path = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
                    if "GEMINI_API_KEY" in secrets:
                        os.environ["GEMINI_API_KEY"] = secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if "GEMINI_API_KEY" not in os.environ:
        print("WARNING: GEMINI_API_KEY not found. Set it before running.")
        exit(1)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    chatbot = ClassBasedRAGChatbot(
        db_dir=os.path.join(SCRIPT_DIR, "test_rag_db"),
        llm_model="gemini-2.5-flash"
    )

    test_pdf_path = os.path.join(SCRIPT_DIR, "Mercedes-Benz-Group-Report-2024-en.pdf")

    if os.path.exists(test_pdf_path):
        chatbot.ingest_pdf(test_pdf_path)
        test_query = "What were Mercedes-Benz's key financial results in 2024?"
        print(f"\n[Test Query] Asking: '{test_query}'")
        result = chatbot.ask(test_query, initial_k=5, final_k=2)
        print("\n=== Bot Answer ===")
        print(result["answer"])
        print("\n=== References used ===")
        for i, item in enumerate(result["sources"], 1):
            doc = item["document"]
            print(f"{i}. [Page {doc.metadata.get('page')}] (Score: {item['score']:.4f}): {doc.page_content[:150]}...")
    else:
        print(f"\nNote: Test PDF not found at {test_pdf_path}. Skipping ingestion demo.")
