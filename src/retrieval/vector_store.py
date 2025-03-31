from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from functools import lru_cache
import time
import os
from pathlib import Path
import uuid

class RAGVectorStore:
    """Class to handle vector storage and retrieval using Qdrant"""
    
    def __init__(self, embedding_model: Any):
        """Initialize the RAG vector store with Qdrant backend"""
        # Store the embedding model
        self.embedding_model = embedding_model
        
        # Create a unique storage path for this instance
        self.storage_path = f"./data/vector_store_{uuid.uuid4().hex[:8]}"
        os.makedirs(self.storage_path, exist_ok=True)
        print(f"Initializing Qdrant with storage at: {self.storage_path}")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            path=self.storage_path
        )
        
        # Get embedding dimension by embedding a test string
        embedding_dimension = len(embedding_model.embed_query("test"))
        print(f"Detected embedding dimension: {embedding_dimension}")
        
        try:
            # Try to delete existing collection if it exists
            collections = self.client.get_collections().collections
            if any(c.name == "rag_documents" for c in collections):
                print("Deleting existing collection: rag_documents")
                self.client.delete_collection("rag_documents")
                
        except Exception as e:
            print(f"Note: {str(e)}")
            
        # Create new collection
        try:
            self.client.create_collection(
                collection_name="rag_documents",
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
            )
            print(f"Created new collection: rag_documents with dimension {embedding_dimension}")
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise
            
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="rag_documents",
            embedding=embedding_model
        )
        
        self.query_stats = {"total_queries": 0, "avg_response_time": 0}
            
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
            # Remove the storage directory
            if hasattr(self, 'storage_path') and os.path.exists(self.storage_path):
                import shutil
                shutil.rmtree(self.storage_path)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching
        
        Args:
            text: Text to get embedding for
            
        Returns:
            List of embedding values
        """
        return self.embedding_model.embed_query(text)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store one by one
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        document_ids = []
        start_time = time.time()
        
        for doc in documents:
            try:
                # Add document to vector store
                doc_id = self.vector_store.add_documents([doc])
                document_ids.extend(doc_id)
                
                print(f"Added document from source: {doc.metadata.get('source', 'unknown')}")
                
            except Exception as e:
                print(f"Error adding document: {str(e)}")
                continue
        
        processing_time = time.time() - start_time
        print(f"Successfully added {len(document_ids)} documents to vector store in {processing_time:.2f} seconds")
        return document_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        return_scores: bool = False
    ) -> Union[List[Document], List[tuple[Document, float]]]:
        """Search for similar documents using query embedding
        
        Args:
            query: The search query string
            k: Number of similar documents to return
            metadata_filter: Optional dictionary of metadata filters
            similarity_threshold: Optional minimum similarity score threshold
            return_scores: Whether to return similarity scores along with documents
            
        Returns:
            If return_scores is True: List of tuples containing (Document, score)
            If return_scores is False: List of similar Document objects
        """
        try:
            start_time = time.time()
            
            # Search using the query
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter
            )
            
            # Apply similarity threshold if specified
            if similarity_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= similarity_threshold]
            
            # Update query statistics
            processing_time = time.time() - start_time
            self.query_stats["total_queries"] += 1
            self.query_stats["avg_response_time"] = (
                (self.query_stats["avg_response_time"] * (self.query_stats["total_queries"] - 1) + processing_time)
                / self.query_stats["total_queries"]
            )
            
            # Return results based on return_scores parameter
            if return_scores:
                print(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
                return results
            else:
                documents = [doc for doc, _ in results]
                print(f"Found {len(documents)} similar documents for query: {query[:50]}...")
                return documents
            
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return [] if not return_scores else []
    
    def get_query_stats(self) -> Dict[str, Union[int, float]]:
        """Get query statistics
        
        Returns:
            Dictionary containing query statistics
        """
        return self.query_stats
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the current storage configuration
        
        Returns:
            Dictionary containing storage information
        """
        return {
            "storage_path": self.storage_path,
            "collection_name": "rag_documents",
            "collection_info": self.client.get_collection("rag_documents")
        } 