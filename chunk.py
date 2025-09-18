import re
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 20):
        """
        Initialize chunker with configurable parameters
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in chunk IDs"""
        # Remove file extension and replace invalid characters with underscores
        name = os.path.splitext(filename)[0]
        # Keep only letters, digits, underscore, dash, equal sign
        sanitized = re.sub(r'[^a-zA-Z0-9_\-=]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "document"
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        # Simple sentence splitting - can be enhanced with NLTK/spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, filename: str) -> List[Dict[str, str]]:
        """Create overlapping chunks from text"""
        if not text or not text.strip():
            return []
        
        cleaned_text = self.clean_text(text)
        sanitized_filename = self.sanitize_filename(filename)
        
        # If text is shorter than chunk_size, return as single chunk
        if len(cleaned_text) <= self.chunk_size:
            return [{
                "chunk_id": f"{sanitized_filename}_chunk_0",
                "filename": filename,
                "text": cleaned_text,
                "chunk_index": 0,
                "char_count": len(cleaned_text)
            }]
        
        chunks = []
        sentences = self.split_by_sentences(cleaned_text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "chunk_id": f"{sanitized_filename}_chunk_{chunk_index}",
                    "filename": filename,
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "char_count": len(current_chunk.strip())
                })
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:] + " " + sentence
                else:
                    current_chunk = sentence
                
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{sanitized_filename}_chunk_{chunk_index}",
                "filename": filename,
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "char_count": len(current_chunk.strip())
            })
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def chunk_documents(self, extracted_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process multiple extracted documents and create chunks"""
        all_chunks = []
        
        for doc in extracted_docs:
            if "error" in doc:
                logger.warning(f"Skipping {doc['filename']} due to error: {doc.get('error')}")
                continue
            
            filename = doc["filename"]
            text = doc["text"]
            
            if not text:
                logger.warning(f"No text found in {filename}")
                continue
            
            chunks = self.create_chunks(text, filename)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks