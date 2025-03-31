from typing import List, Optional, Tuple
from langchain_core.documents import Document
from src.config import RAGConfig
from src.ingestion.document_loader import DocumentLoader
from src.processing.text_processor import TextProcessor
from src.retrieval.vector_store import RAGVectorStore
from src.generation.llm_client import LLMClient
from concurrent.futures import ThreadPoolExecutor
import threading

class RAG:
    """Main RAG (Retrieval Augmented Generation) class"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(self.config)
        self.vector_store = RAGVectorStore(self.text_processor.embedding_model)
        self.llm_client = LLMClient(self.config)
        self.documents_lock = threading.Lock()
        
    def _process_document_batch(self, documents: List[Document]) -> List[Document]:
        """Process a batch of documents (split and embed)"""
        try:
            # Split documents into chunks
            chunks = self.text_processor.split_documents(documents)
            return chunks
        except Exception as e:
            print(f"Error processing document batch: {str(e)}")
            return []
        
    def load_and_process_web_documents(self, urls: List[str], max_depth: int = 1) -> List[str]:
        """Load documents from web URLs, process them, and store in vector store
        
        Args:
            urls: List of seed URLs to start scraping from
            max_depth: Maximum depth for recursive scraping (default: 1)
        """
        # Load documents with recursive scraping
        documents = self.document_loader.load_from_web(urls, max_depth=max_depth)
        
        # Process documents sequentially
        all_chunks = []
        for doc in documents:
            try:
                # Split document into chunks
                chunks = self.text_processor.split_documents([doc])
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue
        
        # Store in vector store
        document_ids = self.vector_store.add_documents(all_chunks)
        
        return document_ids
    
    def query(self, query_text: str, k: int = 4) -> List[Document]:
        """Query the RAG system to retrieve relevant documents"""
        return self.vector_store.similarity_search(query_text, k=k)
    
    def query_with_scores(self, query_text: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Query the RAG system to retrieve relevant documents with relevance scores"""
        return self.vector_store.similarity_search_with_score(query_text, k=k)
    
    def generate_answer(self, question: str, k: int = 4) -> str:
        """Generate an answer using retrieved context and LLM
        
        Args:
            question: The user's question
            k: Number of relevant documents to retrieve
            
        Returns:
            Generated answer as a string
        """
        # Retrieve relevant documents
        context_docs = self.query(question, k=k)
        
        # Generate response using LLM
        response = self.llm_client.generate_response(question, context_docs)
        
        return response 