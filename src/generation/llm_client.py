from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import RAGConfig
import tiktoken

class LLMClient:
    """Client for handling LLM-based text generation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=config.openai_api_key,
            temperature=0.7
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If the question is not related to the context, politely respond that you are tuned to only answer questions about the context.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: """),
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Set maximum context length (GPT-4 Turbo has 128K context window)
        self.max_context_tokens = 10000  # Leave room for prompt and response
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    def _truncate_context(self, context_docs: List[Document], max_tokens: int) -> str:
        """Truncate context to fit within token limit while preserving document boundaries
        
        Args:
            context_docs: List of Document objects
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated context string
        """
        current_tokens = 0
        truncated_docs = []
        
        for doc in context_docs:
            doc_tokens = self._count_tokens(doc.page_content)
            
            # If adding this document would exceed the limit, stop
            if current_tokens + doc_tokens > max_tokens:
                break
                
            truncated_docs.append(doc)
            current_tokens += doc_tokens
        
        # Combine truncated documents
        context = "\n\n".join(doc.page_content for doc in truncated_docs)
        
        print(f"Truncated context to {current_tokens} tokens from {len(context_docs)} documents")
        return context
    
    def generate_response(self, 
                         question: str, 
                         context_docs: List[Document],
                         max_context_tokens: Optional[int] = None) -> str:
        """Generate a response using the provided context documents
        
        Args:
            question: The user's question
            context_docs: List of relevant documents to use as context
            max_context_tokens: Optional maximum number of tokens for context
            
        Returns:
            Generated response as a string
        """
        # Use provided max tokens or default
        max_tokens = max_context_tokens or self.max_context_tokens
        
        # Truncate context to fit within token limit
        context = self._truncate_context(context_docs, max_tokens)
        
        # Generate response
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        return response
