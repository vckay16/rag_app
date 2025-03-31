from src.rag import RAG
from src.config import RAGConfig

def main():
    # Initialize RAG system with default config
    rag = RAG()
    
    # Load and process documents from a web URL with recursive scraping
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    document_ids = rag.load_and_process_web_documents(urls, max_depth=2)  # Scrape up to depth 2
    print(f"Processed documents with IDs: {document_ids[:3]}...")
    
    # Example questions to test the system
    questions = [
        "What are the main components of an AI agent?",
        "How does the agent interact with its environment?",
        "What are the different types of memory in an agent?"
    ]
    
    print("\nGenerating answers...")
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag.generate_answer(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main() 