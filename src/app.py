import streamlit as st
from src.rag import RAG
from src.config import RAGConfig
import time

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag' not in st.session_state:
        st.session_state.rag = RAG()
    if 'processed_urls' not in st.session_state:
        st.session_state.processed_urls = set()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def process_url(url: str, max_depth: int = 2):
    """Process a URL and store documents in the RAG system"""
    if url in st.session_state.processed_urls:
        st.info("This URL has already been processed!")
        return
    
    with st.spinner(f"Processing URL: {url}"):
        try:
            document_ids = st.session_state.rag.load_and_process_web_documents(
                [url], 
                max_depth=max_depth
            )
            st.session_state.processed_urls.add(url)
            st.success(f"Successfully processed {len(document_ids)} documents!")
            return True
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return False

def main():
    st.set_page_config(
        page_title="RAG Chat Interface",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chat Interface")
    st.markdown("""
    This interface allows you to:
    1. Input URLs to process and extract information
    2. Ask questions about the processed content
    3. Get AI-generated answers based on the retrieved information
    """)
    
    initialize_session_state()
    
    # Sidebar for URL input and processing
    with st.sidebar:
        st.header("Document Processing")
        url = st.text_input("Enter URL to process:", placeholder="https://example.com")
        max_depth = st.slider("Scraping Depth:", min_value=1, max_value=5, value=2)
        
        if st.button("Process URL"):
            if url:
                process_url(url, max_depth)
            else:
                st.warning("Please enter a URL")
        
        st.markdown("---")
        st.markdown("### Processed URLs:")
        for url in st.session_state.processed_urls:
            st.markdown(f"- {url}")
    
    # Main chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the processed content"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag.generate_answer(prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main() 