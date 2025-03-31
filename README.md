# RAG Application

A Retrieval-Augmented Generation (RAG) application that combines web scraping, document processing, and vector storage to create an intelligent question-answering system.

## Features

- **Web Scraping**: Efficiently scrapes documentation and content from websites
- **Document Processing**: Splits and processes documents into chunks for better retrieval
- **Vector Storage**: Uses Qdrant for efficient vector storage and similarity search
- **LLM Integration**: Powered by OpenAI's language models for accurate responses
- **Streamlit UI**: User-friendly interface for interacting with the RAG system

## Prerequisites

- Python 3.9+
- Qdrant account and API key
- OpenAI API key
- FireCrawl API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vckay16/rag_app.git
cd rag_app
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Project Structure

```
rag_app/
├── src/
│   ├── ingestion/          # Document loading and web scraping
│   ├── processing/         # Text processing and chunking
│   ├── retrieval/          # Vector store and similarity search
│   ├── generation/         # LLM integration and response generation
│   ├── app.py             # Streamlit application
│   ├── config.py          # Configuration settings
│   └── rag.py             # Main RAG implementation
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup file
└── .env                  # Environment variables
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Enter URLs to scrape in the input field
3. Wait for the documents to be processed
4. Ask questions about the scraped content

## Configuration

The application can be configured through the following parameters in `config.py`:

- `chunk_size`: Size of text chunks for processing
- `chunk_overlap`: Overlap between chunks
- `embedding_model`: OpenAI embedding model to use
- `llm_model`: OpenAI language model to use
- `max_depth`: Maximum depth for web scraping
- `temperature`: LLM temperature for response generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Qdrant](https://qdrant.tech/) for vector storage
- [OpenAI](https://openai.com/) for language models
- [Streamlit](https://streamlit.io/) for the UI framework 