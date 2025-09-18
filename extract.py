import os
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx']
    
    def extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            text = ""
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            
            doc.close()
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {str(e)}")
            return ""
    
    def extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text(self, file_path: str) -> Dict[str, str]:
        """Extract text from supported file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.extract_from_pdf(file_path)
        elif file_extension == '.docx':
            text = self.extract_from_docx(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return {"filename": os.path.basename(file_path), "text": "", "error": "Unsupported format"}
        
        return {
            "filename": os.path.basename(file_path),
            "text": text,
            "word_count": len(text.split()) if text else 0
        }
    
    def extract_multiple_files(self, file_paths: List[str]) -> List[Dict[str, str]]:
        """Extract text from multiple files"""
        results = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
                
            result = self.extract_text(file_path)
            results.append(result)
            logger.info(f"Extracted {result.get('word_count', 0)} words from {result['filename']}")
        
        return results