import os
import pypdf
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ClassBasedRAGChatbot:
    """
    A unified, class-based RAG chatbot using the updated google-genai SDK.
    """

    def __init__(
        self,
        db_dir="./rag_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="ms-marco-MiniLM-L-12-v2",
        llm_model="gemini-2.5-flash"
    ):
        self.db_dir = db_dir
        self.llm_model = llm_model

        # 1. Initialize Gemini client (new SDK)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

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

        # 4. Setup Reranker
        print(f"Loading reranker model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device="cpu")

        # 5. Initialize Vector Store
        print(f"Connecting to Chroma DB at: {self.db_dir}...")
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.db_dir
        )
        print("[OK] RAG Chatbot initialized successfully.")

    def ingest_pdf(self, pdf_path):
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
                    metadata={"source": filename, "page": page_idx + 1}
                )
                documents.append(doc)
                ids.append(f"{filename}_p{page_idx + 1}_c{chunk_idx}")

        if not documents:
            print("[Ingest] Warning: No text could be extracted.")
            return

        print(f"[Ingest] Created {len(documents)} chunks from {filename}.")

        existing = self.vector_store.get(where={"source": filename})
        if existing and existing.get("ids"):
            self.vector_store.delete(ids=existing["ids"])
            print(f"[Ingest] Removed {len(existing['ids'])} existing chunks.")

        self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"[Ingest] Successfully ingested '{filename}'.")

    def retrieve_and_rerank(self, query, initial_k=15, final_k=3, filter_dict=None):
        print(f"\n[Search] Performing initial semantic search (k={initial_k})...")
        search_kwargs = {"k": initial_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        candidates = self.vector_store.similarity_search(query, **search_kwargs)
        if not candidates:
            return []

        print(f"[Search] Reranking {len(candidates)} candidates...")
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        candidate_scores = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_results = candidate_scores[:final_k]

        return [{"document": doc, "score": float(score)} for doc, score in top_results]

    def generate_answer(self, question, context_results):
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
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1500,
                    temperature=0.1
                )
            )
            return response.text
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"

    def ask(self, question, initial_k=15, final_k=3, filter_dict=None):
        context_results = self.retrieve_and_rerank(
            query=question,
            initial_k=initial_k,
            final_k=final_k,
            filter_dict=filter_dict
        )
        answer = self.generate_answer(question, context_results)
        return {"answer": answer, "sources": context_results}
