import logging
from typing import List
from io import BytesIO
from docx import Document

class DocumentProcessor:
    """Utility class for processing documents."""
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        """Extract text from a DOCX file.
        
        Args:
            file_bytes: Raw bytes of the DOCX file
            
        Returns:
            Extracted text as a single string
        """
        try:
            doc = Document(BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise RuntimeError(f"Error extracting text from DOCX: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 400) -> List[str]:
        """Split text into chunks of specified maximum length.
        
        Args:
            text: Input text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        try:
            lines = text.splitlines()
            chunks = []
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk) + len(line) + 1 <= max_chars:
                    current_chunk += line + "\n"
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            logging.info(f"Split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Error chunking text: {str(e)}")
    
    @staticmethod
    def process_file_content(content: bytes, file_name: str) -> str:
        """Process file content based on file type.
        
        Args:
            content: Raw file content
            file_name: Name of the file
            
        Returns:
            Processed text content
        """
        try:
            if file_name.lower().endswith('.docx'):
                return DocumentProcessor.extract_text_from_docx(content)
            else:
                return content.decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error processing file content: {str(e)}") 