from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from src.config import RAGConfig

class TextProcessor:
    """Class to handle text processing operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        self.embedding_model = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
    
    def split_documents(self, documents: Document) -> List[Document]:
        """Split documents into chunks while preserving metadata
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Document objects, each containing a chunk of text
        """
        print(f"\nStarting document splitting:")
        print(f"Input documents: {len(documents)}")
        print(f"Chunk size: {self.config.chunk_size}")
        print(f"Chunk overlap: {self.config.chunk_overlap}")
        
        # Use split_documents directly from RecursiveCharacterTextSplitter
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks")
        print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0:.2f} characters")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        return self.embedding_model.embed_documents(texts) 