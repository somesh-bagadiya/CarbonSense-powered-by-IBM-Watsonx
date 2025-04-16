import re
from typing import List, Dict
import pandas as pd
import numpy as np
from .logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.unit_patterns = {
            'kg': r'\d+\.?\d*\s*kg',
            'co2': r'\d+\.?\d*\s*kg\s*CO2',
            'emission': r'\d+\.?\d*\s*kg\s*CO2e',
            'percentage': r'\d+\.?\d*\s*%'
        }
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text to improve embedding quality."""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s.,;:()%$]', ' ', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Standardize units
            text = self._standardize_units(text)
            
            # Remove redundant information
            text = self._remove_redundant_info(text)
            
            # Add context markers for important information
            text = self._add_context_markers(text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def _standardize_units(self, text: str) -> str:
        """Standardize units and measurements in the text."""
        # Standardize CO2 units
        text = re.sub(r'kg\s*co2e?', 'kg CO2e', text)
        text = re.sub(r'kilograms\s*co2e?', 'kg CO2e', text)
        
        # Standardize weight units
        text = re.sub(r'kilograms?', 'kg', text)
        text = re.sub(r'kgs?', 'kg', text)
        
        # Standardize percentage
        text = re.sub(r'percent', '%', text)
        text = re.sub(r'per\s*cent', '%', text)
        
        return text

    def _remove_redundant_info(self, text: str) -> str:
        """Remove redundant information from the text."""
        # Remove common redundant phrases
        redundant_phrases = [
            r'please\s+note',
            r'as\s+shown\s+above',
            r'as\s+mentioned\s+earlier',
            r'it\s+should\s+be\s+noted',
            r'it\s+is\s+important\s+to\s+note'
        ]
        
        for phrase in redundant_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
            
        return text

    def _add_context_markers(self, text: str) -> str:
        """Add context markers for important information."""
        # Mark important numbers and units
        for unit, pattern in self.unit_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group()
                text = text.replace(value, f"[{unit}: {value}]")
                
        return text

    def process_excel_file(self, file_path: str) -> str:
        """Process Excel file and extract relevant information."""
        try:
            df = pd.read_excel(file_path)
            
            # Convert numeric columns to strings with proper formatting
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
            
            # Add column context
            processed_text = ""
            for col in df.columns:
                if col.lower() in ['co2', 'emission', 'footprint', 'carbon']:
                    processed_text += f"Column {col} contains carbon emission data:\n"
                    processed_text += df[col].to_string() + "\n\n"
                elif any(unit in col.lower() for unit in ['kg', 'co2', 'emission']):
                    processed_text += f"Column {col} contains measurement data:\n"
                    processed_text += df[col].to_string() + "\n\n"
                else:
                    processed_text += f"Column {col}:\n"
                    processed_text += df[col].to_string() + "\n\n"
            
            return self.preprocess_text(processed_text)
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise

    def split_into_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks with improved context preservation."""
        try:
            # First split by paragraphs to maintain context
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed chunk size, start a new chunk
                if len(current_chunk) + len(paragraph) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # If we have no chunks (very short text), return the whole text
                if not chunks:
                    chunks = [text]
                
                # Ensure chunks contain complete sentences where possible
                processed_chunks = []
                for chunk in chunks:
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk:
                                processed_chunks.append(current_chunk.strip())
                            current_chunk = sentence
                    
                    if current_chunk:
                        processed_chunks.append(current_chunk.strip())
                
                return processed_chunks
            
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {str(e)}")
            return [text] 