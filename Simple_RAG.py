"""
PDF RAG System with Gemini Vertex AI
====================================

Production-ready RAG pipeline for PDF document Q&A using ChromaDB vector store
and Google Gemini 2.x models (2026).

Author: Ngoda Boece
Date: April 2026
Version: 2.1 - Clean Output
"""

import os
from typing import List
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Anti-verbose
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class PDFRAG:
    """PDF RAG Pipeline with Gemini integration."""
    
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "## YOUR GCP PROJECT ID ##")
    LOCATION = "us-central1" # GCP region for Vertex AI
    CHROMA_PATH = "./chroma_db" # Local path for ChromaDB persistence
    COLLECTION_NAME = "pdf_rag_pro" # Collection name for ChromaDB
    EMBEDDING_MODEL = "all-MiniLM-L6-v2" # SentenceTransformer model for embeddings
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro-exp"] # Gemini models to try
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 500, 50, 3 # Chunking parameters
    
    def __init__(self):
        vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=self.CHROMA_PATH,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
        self.llm_model = None
        print("PDF RAG initialized")

    def extract_pdf_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        print(f"{len(text)} chars extraits")
        return text

    def chunk_document(self, text: str) -> List[str]:
        if len(text) < self.CHUNK_SIZE:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        print(f"{len(chunks)} chunks créés")
        return chunks

    def index_pdf(self, pdf_path: str) -> None:
        text = self.extract_pdf_text(pdf_path)
        chunks = self.chunk_document(text)
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        print(f"{len(chunks)} chunks indexés → Prêt pour Q&A !")

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        top_k = top_k or self.TOP_K
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        documents = results.get("documents", [])
        return documents[0] if documents else []

    def _get_llm(self) -> GenerativeModel:
        if self.llm_model: return self.llm_model
        for model_name in self.GEMINI_MODELS:
            try:
                test_model = GenerativeModel(model_name)
                test_model.generate_content("test")
                self.llm_model = test_model
                print(f"Gemini {model_name} activé")
                return self.llm_model
            except:
                continue
        raise RuntimeError("Gemini indisponible - Vérifiez GCP")

    def generate(self, query: str) -> str:
        try:
            chunks = self.retrieve(query)
            if not chunks:
                return "Aucun PDF indexé ou match trouvé."
            
            context = "\n\n".join(f"[Extrait {i+1}] {chunk[:350]}..." for i, chunk in enumerate(chunks))
            prompt = f"""Assistant RAG PDF. Répondez UNIQUEMENT basé sur le contexte.

CONTEXTE PDF:
{context}

QUESTION: {query}

Instructions:
- Français professionnel
- Citez les extraits
- Précis et concis"""

            llm = self._get_llm()
            config = GenerationConfig(temperature=0.1, max_output_tokens=1024, top_p=0.8)
            response = llm.generate_content(prompt, generation_config=config)
            return response.text
            
        except Exception as e:
            return f"Erreur: {str(e)}"

def main():
    rag = PDFRAG()
    print("\n" + "="*60)
    print("PDF RAG + Gemini Vertex AI")
    print("="*60)
    
    pdf_path = input("\nChemin PDF (Enter=skip): ").strip()
    if pdf_path:
        try: rag.index_pdf(pdf_path)
        except Exception as e: print(f"❌ {e}")
    
    print("\nChat activé ('quit' pour sortir)")
    print("-"*60)
    
    while True:
        query = input("\nVous > ").strip()
        if query.lower() in ['quit', 'q', 'exit']: 
            print("Fin.")
            break
        if query:
            answer = rag.generate(query)
            print(f"\n{answer}")
            print("-"*60)

if __name__ == "__main__":
    main()
