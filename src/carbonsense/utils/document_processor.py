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

    def chunk_tabular_data(self, df: pd.DataFrame, group_by: List[str] = None, max_rows_per_chunk: int = 10, 
                          semantic_context: Dict[str, List[str]] = None, query: str = None) -> List[Dict]:
        """
        Chunk tabular data while preserving context and relationships.
        
        Args:
            df: DataFrame containing the carbon footprint data
            group_by: List of column names to group by (e.g., ['facility', 'year'])
            max_rows_per_chunk: Maximum number of rows per chunk
            semantic_context: Dictionary mapping ambiguous terms to their context-specific meanings
                            e.g., {'apple': ['fruit', 'company', 'product']}
            query: The search query to help determine relevant context
            
        Returns:
            List of dictionaries containing chunks with metadata and relevance scores
        """
        try:
            chunks = []
            
            # Identify carbon-related columns
            carbon_columns = [col for col in df.columns if any(term in col.lower() 
                            for term in ['co2', 'carbon', 'emission', 'footprint'])]
            
            # Calculate basic statistics for carbon columns
            carbon_stats = {}
            for col in carbon_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    carbon_stats[col] = {
                        'mean': df[col].mean(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'total': df[col].sum()
                    }
            
            # Handle semantic context if provided
            if semantic_context and query:
                # Extract potential terms from the query
                query_terms = set(query.lower().split())
                
                # Score each context based on query relevance
                context_scores = {}
                for term, contexts in semantic_context.items():
                    if term.lower() in query_terms:
                        for context in contexts:
                            # Calculate context relevance score
                            score = self._calculate_context_relevance(query, term, context, df)
                            context_scores[(term, context)] = score
                
                # Sort contexts by relevance score
                sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Create chunks for the most relevant contexts
                for (term, context), score in sorted_contexts:
                    if score > 0:  # Only include contexts with positive relevance
                        # Filter rows that match both term and context
                        context_mask = df.apply(
                            lambda row: any(
                                term.lower() in str(val).lower() and 
                                context.lower() in str(val).lower() 
                                for val in row
                            ), axis=1
                        )
                        context_df = df[context_mask]
                        
                        if not context_df.empty:
                            # Create chunks for this specific context
                            for i in range(0, len(context_df), max_rows_per_chunk):
                                chunk_df = context_df.iloc[i:i + max_rows_per_chunk]
                                
                                # Calculate chunk-specific relevance
                                chunk_relevance = self._calculate_chunk_relevance(chunk_df, query)
                                
                                chunk_data = {
                                    'data': chunk_df,
                                    'metadata': {
                                        'semantic_context': {
                                            'term': term,
                                            'context': context,
                                            'context_score': score
                                        },
                                        'relevance_score': chunk_relevance,
                                        'start_index': i,
                                        'end_index': min(i + max_rows_per_chunk, len(context_df)),
                                        'row_count': len(chunk_df),
                                        'carbon_columns': carbon_columns,
                                        'matching_columns': [
                                            col for col in df.columns 
                                            if any(
                                                term.lower() in str(val).lower() and 
                                                context.lower() in str(val).lower() 
                                                for val in df[col]
                                            )
                                        ],
                                        'query_terms': list(query_terms)
                                    }
                                }
                                chunks.append(chunk_data)
                
                if chunks:  # If we found context-specific chunks, return them sorted by relevance
                    return sorted(chunks, key=lambda x: x['metadata']['relevance_score'], reverse=True)
            
            # If no semantic context or no matches found, proceed with regular chunking
            if not group_by:
                # Try to identify natural groupings from the data
                potential_group_cols = [col for col in df.columns 
                                     if df[col].nunique() < len(df) * 0.5]
                
                if potential_group_cols:
                    group_by = [max(potential_group_cols, 
                                  key=lambda x: df[x].nunique())]
                    logger.info(f"Using automatic grouping by: {group_by}")
            
            # Group by specified columns
            grouped = df.groupby(group_by)
            
            for group_key, group_df in grouped:
                # Calculate group-specific statistics
                group_carbon_stats = {}
                for col in carbon_columns:
                    if pd.api.types.is_numeric_dtype(group_df[col]):
                        group_carbon_stats[col] = {
                            'mean': group_df[col].mean(),
                            'min': group_df[col].min(),
                            'max': group_df[col].max(),
                            'total': group_df[col].sum()
                        }
                
                # If group is larger than max_rows_per_chunk, split it
                if len(group_df) > max_rows_per_chunk:
                    # Sort by date if available, otherwise by index
                    sort_col = next((col for col in group_df.columns 
                                   if 'date' in col.lower() or 'time' in col.lower()), 
                                  None)
                    if sort_col:
                        group_df = group_df.sort_values(by=sort_col)
                    
                    for i in range(0, len(group_df), max_rows_per_chunk):
                        chunk_df = group_df.iloc[i:i + max_rows_per_chunk]
                        chunk_data = {
                            'data': chunk_df,
                            'metadata': {
                                'group_key': group_key,
                                'start_index': i,
                                'end_index': min(i + max_rows_per_chunk, len(group_df)),
                                'row_count': len(chunk_df),
                                'carbon_columns': carbon_columns,
                                'group_stats': group_carbon_stats,
                                'time_range': {
                                    'start': chunk_df[sort_col].min() if sort_col else None,
                                    'end': chunk_df[sort_col].max() if sort_col else None
                                }
                            }
                        }
                        chunks.append(chunk_data)
                else:
                    # Keep the entire group together
                    chunk_data = {
                        'data': group_df,
                        'metadata': {
                            'group_key': group_key,
                            'start_index': 0,
                            'end_index': len(group_df),
                            'row_count': len(group_df),
                            'carbon_columns': carbon_columns,
                            'group_stats': group_carbon_stats,
                            'time_range': {
                                'start': group_df[sort_col].min() if sort_col else None,
                                'end': group_df[sort_col].max() if sort_col else None
                            }
                        }
                    }
                    chunks.append(chunk_data)
            
            # Add global statistics to each chunk
            for chunk in chunks:
                chunk['metadata']['global_stats'] = carbon_stats
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking tabular data: {str(e)}")
            return [{'data': df, 'metadata': {'error': str(e)}}]

    def _calculate_context_relevance(self, query: str, term: str, context: str, df: pd.DataFrame) -> float:
        """
        Calculate the relevance score for a specific context based on the query.
        
        Args:
            query: The search query
            term: The ambiguous term
            context: The specific context
            df: The dataframe containing the data
            
        Returns:
            float: Relevance score between 0 and 1
        """
        score = 0.0
        
        # Check for context-specific keywords in the query
        context_keywords = {
            'fruit': ['food', 'eat', 'nutrition', 'fresh', 'organic'],
            'company': ['business', 'corporate', 'enterprise', 'firm', 'inc', 'ltd'],
            'product': ['device', 'item', 'product', 'model', 'version']
        }
        
        # Add more context-specific keywords as needed
        
        # Check if query contains context-specific keywords
        if context.lower() in context_keywords:
            for keyword in context_keywords[context.lower()]:
                if keyword in query.lower():
                    score += 0.3
                    
        # Check column names for context relevance
        context_relevant_columns = [
            col for col in df.columns 
            if context.lower() in col.lower()
        ]
        if context_relevant_columns:
            score += 0.2
            
        # Check for context-specific patterns in the data
        context_patterns = {
            'fruit': r'\d+\s*(kg|g|lb|oz)',  # Weight units
            'company': r'(inc|llc|ltd|corp)',  # Company suffixes
            'product': r'(model|version|series)\s*[A-Z0-9]'  # Product identifiers
        }
        
        if context.lower() in context_patterns:
            pattern = context_patterns[context.lower()]
            matches = df.apply(
                lambda row: any(
                    re.search(pattern, str(val), re.IGNORECASE) 
                    for val in row
                ), axis=1
            ).sum()
            if matches > 0:
                score += 0.2
                
        return min(score, 1.0)  # Cap the score at 1.0
        
    def _calculate_chunk_relevance(self, chunk_df: pd.DataFrame, query: str) -> float:
        """
        Calculate the relevance score for a specific chunk based on the query.
        
        Args:
            chunk_df: The dataframe chunk
            query: The search query
            
        Returns:
            float: Relevance score between 0 and 1
        """
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Check for query terms in the chunk
        for term in query_terms:
            matches = chunk_df.apply(
                lambda row: any(
                    term in str(val).lower() 
                    for val in row
                ), axis=1
            ).sum()
            if matches > 0:
                score += 0.3
                
        # Check for carbon-related columns (higher weight)
        carbon_matches = chunk_df.apply(
            lambda row: any(
                any(term in str(val).lower() for term in ['co2', 'carbon', 'emission'])
                for val in row
            ), axis=1
        ).sum()
        if carbon_matches > 0:
            score += 0.4
            
        return min(score, 1.0)  # Cap the score at 1.0

    def handle_ambiguous_query(self, query: str, df: pd.DataFrame) -> Dict:
        """
        Handle ambiguous queries by analyzing possible contexts and requesting clarification.
        
        Args:
            query: The user's search query
            df: The dataframe containing the data
            
        Returns:
            Dict containing possible contexts and their data samples
        """
        try:
            # Extract potential ambiguous terms from the query
            query_terms = set(query.lower().split())
            
            # Define possible contexts and their indicators
            context_indicators = {
                'company': {
                    'keywords': ['corporate', 'business', 'company', 'inc', 'ltd', 'enterprise'],
                    'patterns': [r'(inc|llc|ltd|corp)', r'headquarters', r'business'],
                    'column_patterns': [r'company', r'corporate', r'business']
                },
                'fruit': {
                    'keywords': ['food', 'fruit', 'agricultural', 'farm', 'organic', 'fresh'],
                    'patterns': [r'\d+\s*(kg|g|lb|oz)', r'farm', r'orchard'],
                    'column_patterns': [r'food', r'fruit', r'agricultural']
                },
                'product': {
                    'keywords': ['product', 'device', 'item', 'manufacturing'],
                    'patterns': [r'(model|version|series)\s*[A-Z0-9]', r'manufacturing'],
                    'column_patterns': [r'product', r'manufacturing']
                }
            }
            
            # Find matching contexts
            matching_contexts = {}
            for context, indicators in context_indicators.items():
                score = 0
                matching_data = []
                
                # Check query keywords
                for keyword in indicators['keywords']:
                    if keyword in query.lower():
                        score += 0.3
                
                # Check data patterns
                for pattern in indicators['patterns']:
                    matches = df.apply(
                        lambda row: any(
                            re.search(pattern, str(val), re.IGNORECASE) 
                            for val in row
                        ), axis=1
                    ).sum()
                    if matches > 0:
                        score += 0.2
                        # Get sample data for this context
                        sample_data = df[df.apply(
                            lambda row: any(
                                re.search(pattern, str(val), re.IGNORECASE) 
                                for val in row
                            ), axis=1
                        )].head(2)
                        if not sample_data.empty:
                            matching_data.append(sample_data)
                
                # Check column patterns
                matching_columns = [
                    col for col in df.columns 
                    if any(re.search(pattern, col.lower()) for pattern in indicators['column_patterns'])
                ]
                if matching_columns:
                    score += 0.2
                
                if score > 0:
                    matching_contexts[context] = {
                        'score': score,
                        'sample_data': matching_data,
                        'matching_columns': matching_columns
                    }
            
            # If we have multiple matching contexts with similar scores
            if len(matching_contexts) > 1:
                max_score = max(ctx['score'] for ctx in matching_contexts.values())
                similar_contexts = [
                    ctx for ctx, data in matching_contexts.items() 
                    if data['score'] >= max_score * 0.8  # 80% of max score
                ]
                
                if len(similar_contexts) > 1:
                    return {
                        'is_ambiguous': True,
                        'possible_contexts': similar_contexts,
                        'context_data': matching_contexts,
                        'clarification_needed': True,
                        'suggested_clarification': f"Did you mean carbon footprint for apple as a {', '.join(similar_contexts)}?"
                    }
            
            return {
                'is_ambiguous': False,
                'contexts': matching_contexts,
                'clarification_needed': False
            }
            
        except Exception as e:
            logger.error(f"Error handling ambiguous query: {str(e)}")
            return {
                'is_ambiguous': False,
                'error': str(e),
                'clarification_needed': False
            }

    def process_query_with_clarification(self, query: str, df: pd.DataFrame) -> Dict:
        """
        Process a query with potential ambiguity and handle clarification.
        
        Args:
            query: The user's search query
            df: The dataframe containing the data
            
        Returns:
            Dict containing either clarification request or processed results
        """
        # First check for ambiguity
        ambiguity_check = self.handle_ambiguous_query(query, df)
        
        if ambiguity_check['clarification_needed']:
            return {
                'status': 'needs_clarification',
                'message': ambiguity_check['suggested_clarification'],
                'context_data': ambiguity_check['context_data']
            }
        
        # If no ambiguity or after clarification, process the query
        semantic_context = {
            term: list(context_indicators.keys())
            for term in query.lower().split()
            if term in context_indicators
        }
        
        return {
            'status': 'processed',
            'results': self.chunk_tabular_data(
                df=df,
                semantic_context=semantic_context,
                query=query
            )
        } 