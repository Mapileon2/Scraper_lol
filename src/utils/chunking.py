"""HTML content chunking utilities for the Web Scraper application."""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from ..config import SCRAPER_CONFIG

def chunk_html(
    html_content: str, 
    max_chars: int = None, 
    overlap: int = None
) -> List[Dict[str, Any]]:
    """Split HTML content into chunks with semantic boundaries.
    
    Args:
        html_content: The HTML content to split
        max_chars: Maximum characters per chunk (defaults to SCRAPER_CONFIG.CHUNK_SIZE)
        overlap: Number of characters to overlap between chunks (defaults to SCRAPER_CONFIG.CHUNK_OVERLAP)
        
    Returns:
        List of dictionaries containing chunk data with 'content' and 'metadata' keys
    """
    if max_chars is None:
        max_chars = SCRAPER_CONFIG.CHUNK_SIZE
    if overlap is None:
        overlap = SCRAPER_CONFIG.CHUNK_OVERLAP
        
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Process all text nodes in the document
    for element in soup.find_all(True):
        if not element.text.strip():
            continue
            
        text = element.get_text(separator=' ', strip=True)
        if not text:
            continue
            
        element_length = len(text)
        
        # If element is too large, split it
        if element_length > max_chars:
            if current_chunk:
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'metadata': {'type': 'html_chunk'}
                })
                current_chunk = []
                current_length = 0
                
            # Split the large element into smaller chunks
            for i in range(0, element_length, max_chars - overlap):
                chunk_text = text[i:i + max_chars]
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'type': 'split_element',
                        'element': element.name,
                        'position': i
                    }
                })
            continue
            
        # Add element to current chunk if it fits
        if current_length + element_length <= max_chars:
            current_chunk.append(text)
            current_length += element_length
        else:
            # Save current chunk and start a new one
            if current_chunk:
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'metadata': {'type': 'html_chunk'}
                })
            current_chunk = [text]
            current_length = element_length
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append({
            'content': ' '.join(current_chunk),
            'metadata': {'type': 'html_chunk'}
        })
    
    return chunks

def split_html_with_overlap(
    html: str, 
    chunk_size: int = None, 
    overlap: int = None
) -> List[str]:
    """Split HTML into chunks with overlap, respecting semantic boundaries.
    
    This is a simpler version that doesn't preserve HTML structure.
    
    Args:
        html: The HTML content to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of HTML chunks
    """
    if chunk_size is None:
        chunk_size = SCRAPER_CONFIG.CHUNK_SIZE
    if overlap is None:
        overlap = SCRAPER_CONFIG.CHUNK_OVERLAP
    
    # Remove extra whitespace and newlines
    text = ' '.join(html.split())
    chunks = []
    
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences first for better chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If sentence is too long, split it
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            start = 0
            while start < sentence_length:
                end = start + chunk_size
                chunks.append(sentence[start:end])
                start = end - min(overlap, chunk_size // 2)
                
            continue
            
        # Add sentence to current chunk if it fits
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length + 1
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
