from dotenv import load_dotenv
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for RAG application"""
    chunk_size: int = 4000
    chunk_overlap: int = 400
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4"
    
    def __post_init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables") 