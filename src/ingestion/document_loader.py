from typing import List, Optional, Set
import bs4
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents import Document
from collections import deque
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Class to handle document loading from various sources"""
    
    def __init__(self):
        self.bs4_strainer = bs4.SoupStrainer(
            class_=(
                "post-title", "post-header", "post-content",
                "article-content", "article-body", "article",
                "main-content", "content", "documentation-content",
                "help-content", "doc-content"
            )
        )
        
        # Common documentation site patterns
        self.doc_patterns = [
            r'/docs/',
            r'/help/',
            r'/documentation/',
            r'/articles/',
            r'/support/',
            r'/guide/',
            r'/manual/'
        ]
        
        # Get FireCrawl API key
        self.firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
    
    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain"""
        try:
            parsed = urlparse(url)
            # Check if it's a documentation URL
            is_doc_url = any(re.search(pattern, url) for pattern in self.doc_patterns)
            return (parsed.netloc == base_domain and 
                   parsed.scheme in ['http', 'https'] and
                   is_doc_url)
        except:
            return False
            
    def _extract_links(self, url: str, html_content: str, base_domain: str) -> Set[str]:
        """Extract valid links from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        # Look for links in navigation and content areas
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            # Skip common non-documentation links
            if any(x in full_url.lower() for x in ['login', 'signup', 'account', 'profile']):
                continue
                
            if self._is_valid_url(full_url, base_domain):
                links.add(full_url)
                
        return links
    
    def _process_url(self, url: str, depth: int, base_domain: str, max_depth: int) -> List[Document]:
        """Process a single URL and return its documents"""
        try:
            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            # Fetch and parse the page with headers
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            # Load content using WebBaseLoader with custom settings
            '''
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            '''
            loader = FireCrawlLoader(
                api_key=self.firecrawl_api_key,
                url = url,
                mode = "scrape",
            )
            page_documents = loader.load()
            
            #print the document content
            print(f"Document content: {page_documents[0].page_content[:100]}")

            # Add metadata to documents
            for doc in page_documents:
                doc.metadata.update({
                    "source": url,
                    "depth": depth,
                    "base_domain": base_domain
                })
            
            print(f"Scraped {url} (depth: {depth})")
            
            # If not at max depth, extract links for further processing
            if depth < max_depth:
                new_links = self._extract_links(url, response.text, base_domain)
                return page_documents, new_links
                
            return page_documents, set()
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return [], set()
    
    def load_from_web(self, urls: List[str], max_depth: int = 5) -> List[Document]:
        """Load documents from web URLs sequentially
        
        Args:
            urls: List of seed URLs to start scraping from
            max_depth: Maximum depth for recursive scraping (default: 5)
            
        Returns:
            List of Document objects containing the scraped content
        """
        if not urls:
            return []
            
        # Initialize variables
        visited = set(urls)
        documents = []
        base_domain = urlparse(urls[0]).netloc
        url_queue = deque([(url, 0) for url in urls])  # (url, depth)
        
        # Process URLs sequentially
        while url_queue:
            url, depth = url_queue.popleft()
            
            # Process the URL
            page_docs, new_links = self._process_url(url, depth, base_domain, max_depth)
            documents.extend(page_docs)
            
            # Add new unvisited links to queue
            if depth < max_depth:
                for link in new_links:
                    if link not in visited:
                        visited.add(link)
                        url_queue.append((link, depth + 1))
        
        print(f"Scraped {len(documents)} documents from {len(visited)} URLs")
        return documents
    
    def load_from_files(self, file_paths: List[str]) -> List[Document]:
        """Load documents from local files"""
        # TODO: Implement file loading logic
        raise NotImplementedError("File loading not implemented yet") 