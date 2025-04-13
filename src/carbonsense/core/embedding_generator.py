import os
import json
from typing import List, Dict, Any
from ..config.config_manager import ConfigManager
from ..services.cos_service import COSService
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..utils.document_processor import DocumentProcessor
from ..utils.logger import setup_logger
import pandas as pd
import traceback
from datetime import datetime
import re

# Set up logger
logger = setup_logger(__name__)

class EmbeddingGenerator:
    """Main class for generating and storing document embeddings."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the embedding generator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.cos = COSService(config)
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        self.document_processor = DocumentProcessor()
        
    def _is_file_processed(self, file_name: str, use_125m: bool = False, use_granite: bool = False) -> bool:
        """Check if a file has already been processed."""
        try:
            # Determine model type and directory
            model_type = "granite" if use_granite else "125m" if use_125m else "30m"
            model_dir = os.path.join("Embeddings", f"Embeddings_{model_type}")
            
            # Check local embeddings in model-specific directory
            local_path = os.path.join(model_dir, f"{file_name}.json")
            if os.path.exists(local_path):
                # Check if embeddings exist in Milvus
                collection_name = f"carbon_embeddings_{model_type}"
                
                # First check if collection exists
                if not self.milvus.collection_exists(collection_name):
                    logger.info(f"Collection '{collection_name}' does not exist yet, will be created during processing")
                    return False
                
                # If collection exists, check for file's embeddings
                query = f'source_file == "{file_name}"'
                results = self.milvus.query_vectors(collection_name, query, limit=1)
                if results:
                    logger.info(f"File '{file_name}' already processed and exists in both local storage and Milvus")
                    return True
                else:
                    logger.info(f"File '{file_name}' exists locally but not in Milvus, will be reprocessed")
                    return False
            
            # Check COS embeddings in model-specific directory
            cos_path = f"embeddings/{model_type}/{file_name}.json"
            if self.cos.file_exists(cos_path):
                logger.info(f"File '{file_name}' already processed and exists in COS")
                return True
            
            return False
        except Exception as e:
            error_msg = (
                f"Error checking file status for '{file_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            return False

    def process_file(self, file_path: str, use_125m: bool = False, use_granite: bool = False) -> None:
        """Process a single file and generate embeddings."""
        try:
            file_name = os.path.basename(file_path)
            logger.info(f"Starting to process file: {file_name}")
            
            if self._is_file_processed(file_name, use_125m, use_granite):
                logger.info(f"Skipping already processed file: {file_name}")
                return
            
            # Read the file content
            if file_path.endswith('.xlsx'):
                try:
                    logger.info(f"Reading Excel file: {file_name}")
                    df = pd.read_excel(file_path)
                    text = df.to_string(index=False)
                    logger.info(f"Successfully loaded Excel file: {file_name} with {len(df)} rows")
                except Exception as e:
                    error_msg = (
                        f"Failed to read Excel file '{file_name}'. Details:\n"
                        f"- Path: {file_path}\n"
                        f"- Error: {str(e)}"
                    )
                    logger.error(error_msg)
                    raise
            else:
                try:
                    logger.info(f"Reading text file: {file_name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    logger.info(f"Successfully loaded text file: {file_name} with {len(text)} characters")
                except Exception as e:
                    error_msg = (
                        f"Failed to read text file '{file_name}'. Details:\n"
                        f"- Path: {file_path}\n"
                        f"- Error: {str(e)}"
                    )
                    logger.error(error_msg)
                    raise
            
            # Preprocess text
            logger.info(f"Preprocessing text for {file_name}")
            text = self.document_processor.preprocess_text(text)
            
            # Split text into chunks
            logger.info(f"Splitting text into chunks for {file_name}")
            chunks = self._split_text(text)
            if not chunks:
                error_msg = (
                    f"No chunks extracted from '{file_name}'. Details:\n"
                    f"- File size: {len(text)} bytes\n"
                    f"- File type: {'Excel' if file_path.endswith('.xlsx') else 'Text'}"
                )
                logger.error(error_msg)
                return
            
            logger.info(f"Successfully split '{file_name}' into {len(chunks)} chunks")
            
            # Generate embeddings using Watsonx
            embeddings = []
            model_type = "granite" if use_granite else "125m" if use_125m else "30m"
            logger.info(f"Generating embeddings for '{file_name}' using {model_type} model")
            
            for i, chunk in enumerate(chunks, 1):
                try:
                    start_time = datetime.now()
                    embedding = self.watsonx.generate_embedding(chunk, model_type=model_type)
                    if embedding:
                        embeddings.append(embedding)
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        if i % 10 == 0:  # Log progress more frequently
                            logger.info(
                                f"Progress for '{file_name}': {i}/{len(chunks)} chunks processed "
                                f"(last chunk took {elapsed_time:.2f}s)"
                            )
                except Exception as e:
                    error_msg = (
                        f"Error processing chunk {i} of '{file_name}'. Details:\n"
                        f"- Chunk size: {len(chunk)} characters\n"
                        f"- Error: {str(e)}"
                    )
                    logger.error(error_msg)
                    continue
            
            if not embeddings:
                error_msg = (
                    f"No embeddings generated for '{file_name}'. Details:\n"
                    f"- Total chunks: {len(chunks)}\n"
                    f"- Model type: {model_type}"
                )
                logger.error(error_msg)
                return
            
            # Save embeddings and store in Milvus
            logger.info(f"Saving embeddings for '{file_name}'")
            try:
                self._save_embeddings(file_name, chunks, embeddings, use_125m, use_granite)
                self._store_in_milvus(file_name, chunks, embeddings, use_125m, use_granite)
                logger.info(f"Successfully processed '{file_name}'")
            except Exception as e:
                error_msg = (
                    f"Failed to save/store embeddings for '{file_name}'. Details:\n"
                    f"- Number of embeddings: {len(embeddings)}\n"
                    f"- Error: {str(e)}"
                )
                logger.error(error_msg)
                raise
            
        except Exception as e:
            error_msg = (
                f"Error processing file '{file_name}'. Details:\n"
                f"- Path: {file_path}\n"
                f"- Error: {str(e)}"
            )
            logger.error(error_msg)
            raise
    
    def process_all_files(self, use_125m: bool = False, use_granite: bool = False, specific_files: List[str] = None) -> None:
        """Process all files in the Data_processed directory.
        
        Args:
            use_125m: Whether to use 125M model
            use_granite: Whether to use Granite model
            specific_files: List of specific files to process (optional)
        """
        try:
            # Define categories to process
            categories = ['industry', 'electricity', 'regional']
            total_files = 0
            processed_files = 0
            
            # Process each category
            for category in categories:
                category_path = os.path.join("Data_processed", category)
                if not os.path.exists(category_path):
                    logger.warning(f"Category directory not found: {category_path}")
                    continue
                
                # Count total files to process
                files_in_category = [f for f in os.listdir(category_path) 
                                   if f.endswith(('.xlsx', '.txt')) and 
                                   (not specific_files or f in specific_files)]
                total_files += len(files_in_category)
                
                # Process each file in the category
                for file_name in files_in_category:
                    file_path = os.path.join(category_path, file_name)
                    try:
                        self.process_file(file_path, use_125m, use_granite)
                        processed_files += 1
                        logger.info(f"Processed {processed_files}/{total_files} files: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        continue
                            
            logger.info(f"Completed processing {processed_files}/{total_files} files")
                            
        except Exception as e:
            logger.error(f"Error in process_all_files: {str(e)}")
            raise
    
    def _save_embeddings(self, object_name: str, chunks: List[str], embeddings: List[List[float]], 
                        use_125m: bool = False, use_granite: bool = False) -> None:
        """Save embeddings to local file and COS.
        
        Args:
            object_name: Name of the object
            chunks: List of text chunks
            embeddings: List of embeddings
            use_125m: Whether to use the 125M parameter model (default: False)
            use_granite: Whether to use granite model
        """
        try:
            # Determine model type and directory
            model_type = "granite" if use_granite else "125m" if use_125m else "30m"
            model_dir = os.path.join("Embeddings", f"Embeddings_{model_type}")
            
            # Create model-specific directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save locally in model-specific directory
            local_path = os.path.join(model_dir, f"{object_name}.json")
            
            # Prepare data for saving
            data = {
                "chunks": chunks,
                "embeddings": embeddings,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to local file
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved embeddings locally to {local_path}")
            
            # Save to COS with model-specific path
            cos_path = f"embeddings/{model_type}/{object_name}.json"
            self.cos.upload_file(local_path, cos_path)
            logger.info(f"Uploaded embeddings to COS at {cos_path}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def _store_in_milvus(self, object_name: str, chunks: List[str], embeddings: List[List[float]], 
                        use_125m: bool = False, use_granite: bool = False) -> None:
        """Store embeddings in Milvus."""
        try:
            # Prepare entities for Milvus
            entities = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create entity with proper data types
                entity = {
                    "id": str(f"{object_name}_{i}"),
                    "text": str(chunk),
                    "embedding": [float(x) for x in embedding],
                    "source_file": str(object_name),
                    "model_type": str("granite" if use_granite else "125m" if use_125m else "30m"),
                    "timestamp": int(datetime.now().timestamp()),
                    "version": str("1.0"),
                    "chunk_index": int(i),
                    "total_chunks": int(len(chunks)),
                    "metadata": str("{}"),
                    "processing_info": str("{}")
                }
                entities.append(entity)
            
            # Get collection name
            collection_name = "carbon_embeddings_granite" if use_granite else "carbon_embeddings_125m" if use_125m else "carbon_embeddings_30m"
            
            # Create collection if it doesn't exist
            if not self.milvus.collection_exists(collection_name):
                if embeddings and len(embeddings) > 0:
                    dim = len(embeddings[0])
                    logger.info(f"Creating collection {collection_name} with dimension {dim}")
                    self.milvus.create_collection(collection_name, dim)
                else:
                    raise ValueError("No embeddings available to determine dimension")
            
            # Insert entities
            if entities:
                self.milvus.insert_vectors(collection_name, entities)
                logger.info(f"Successfully stored {len(entities)} embeddings in Milvus for {object_name}")
            else:
                logger.warning(f"No entities to store for {object_name}")
            
        except Exception as e:
            error_msg = (
                f"Error storing in Milvus: {str(e)}\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into meaningful chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        logger.info(f"Starting text splitting with total length: {len(text)} characters")
        
        # Get chunking parameters from config
        chunk_size = self.config.get_chunking_config()["chunk_size"]
        overlap = self.config.get_chunking_config()["overlap"]
        logger.info(f"Using chunk size: {chunk_size}, overlap: {overlap}")
        
        # If the text is from Excel, use specialized Excel chunking
        if '\n' in text and '\t' in text:  # Likely Excel data
            logger.info("Detected Excel-like data format, using specialized Excel chunking")
            return self._split_excel_text(text, chunk_size, overlap)
        
        # For non-Excel text, use the original splitting logic
        logger.info("Using regular text chunking")
        return self._split_regular_text(text, chunk_size, overlap)

    def _split_excel_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split Excel data into meaningful chunks while preserving data relationships.
        
        Args:
            text: Excel data as text
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Handle empty or very small text
        if not text.strip():
            logger.warning("Empty Excel text provided")
            return []
            
        # For very small Excel files, return the entire content as one chunk
        if len(text) < chunk_size:
            logger.info(f"Excel text is smaller than chunk size ({len(text)} < {chunk_size}), using single chunk")
            return [text]
            
        chunks = []
        rows = text.split('\n')
        logger.info(f"Processing Excel data with {len(rows)} rows")
        
        # Extract header and metadata
        header = None
        metadata = {}
        if rows and '\t' in rows[0]:  # First row might be header
            header = rows[0]
            rows = rows[1:]
            logger.info(f"Detected header row: {header}")
            
            # Extract column types and importance
            header_columns = header.split('\t')
            metadata['column_types'] = self._analyze_column_types(rows, header_columns)
            metadata['important_columns'] = self._identify_important_columns(header_columns)
        
        # For small number of rows, return as single chunk with header
        if len(rows) <= 5:  # If 5 or fewer rows, keep as one chunk
            logger.info(f"Small Excel file detected ({len(rows)} rows), using single chunk")
            if header:
                return [header + '\n' + '\n'.join(rows)]
            return ['\n'.join(rows)]
        
        # Group rows by their relationships
        row_groups = []
        current_group = []
        current_columns = None
        current_size = 0
        current_context = {}
        
        for row in rows:
            if not row.strip():  # Skip empty rows
                continue
                
            columns = row.split('\t')
            if current_columns is None:
                current_columns = len(columns)
                logger.info(f"Initial row structure detected: {current_columns} columns")
            
            row_size = len(row)
            row_context = self._extract_row_context(columns, header_columns, metadata)
            
            # Check if we should start a new group based on:
            # 1. Structure change
            # 2. Size limit
            # 3. Context change (e.g., different category, significant value change)
            if (len(columns) != current_columns or 
                current_size + row_size > chunk_size or
                self._should_start_new_group(row_context, current_context)):
                
                if current_group:
                    row_groups.append({
                        'rows': current_group,
                        'context': current_context,
                        'size': current_size
                    })
                    logger.debug(f"Created new row group with {len(current_group)} rows")
                
                current_group = []
                current_columns = len(columns)
                current_size = 0
                current_context = row_context
            
            current_group.append(row)
            current_size += row_size
            current_context = self._update_context(current_context, row_context)
        
        if current_group:
            row_groups.append({
                'rows': current_group,
                'context': current_context,
                'size': current_size
            })
        
        logger.info(f"Created {len(row_groups)} row groups")
        
        # Create chunks from row groups with proper context
        for i, group in enumerate(row_groups):
            chunk = []
            
            # Add context information
            if header:
                chunk.append(header)
            chunk.append(self._format_context(group['context']))
            
            # Add rows with proper formatting
            for row in group['rows']:
                if len(chunk) + len(row) > chunk_size and len(chunk) > 1:
                    chunks.append('\n'.join(chunk))
                    logger.debug(f"Created chunk {len(chunks)} with {len(chunk)} rows")
                    # Keep context and last few rows for overlap
                    overlap_rows = min(len(group['rows']), 3)
                    chunk = [header] if header else []
                    chunk.append(self._format_context(group['context']))
                    chunk.extend(group['rows'][-overlap_rows:])
                
                chunk.append(row)
            
            if chunk:
                chunks.append('\n'.join(chunk))
                logger.debug(f"Created final chunk for group {i+1} with {len(chunk)} rows")
        
        # If no chunks were created, return the entire content as one chunk
        if not chunks:
            logger.info("No chunks created, returning entire content as single chunk")
            if header:
                return [header + '\n' + '\n'.join(rows)]
            return ['\n'.join(rows)]
        
        logger.info(f"Excel chunking complete. Created {len(chunks)} chunks")
        return chunks

    def _analyze_column_types(self, rows: List[str], header_columns: List[str]) -> Dict[str, str]:
        """Analyze column types based on data patterns.
        
        Args:
            rows: List of data rows
            header_columns: List of column headers
            
        Returns:
            Dictionary mapping column names to their types
        """
        column_types = {}
        sample_size = min(100, len(rows))  # Analyze first 100 rows
        
        for i, col_name in enumerate(header_columns):
            try:
                values = [row.split('\t')[i].strip() for row in rows[:sample_size] if row and len(row.split('\t')) > i]
                if not values:
                    column_types[col_name] = 'text'
                    continue
                    
                # Check for numeric patterns
                if all(re.match(r'^-?\d+(\.\d+)?$', v) for v in values if v):
                    column_types[col_name] = 'numeric'
                # Check for date patterns
                elif all(re.match(r'^\d{4}-\d{2}-\d{2}$', v) for v in values if v):
                    column_types[col_name] = 'date'
                # Check for categorical patterns (less than 50% unique values)
                elif len(set(values)) < sample_size * 0.5:
                    column_types[col_name] = 'categorical'
                else:
                    column_types[col_name] = 'text'
            except Exception as e:
                logger.warning(f"Error analyzing column {col_name}: {str(e)}")
                column_types[col_name] = 'text'  # Default to text on error
        
        return column_types

    def _identify_important_columns(self, header_columns: List[str]) -> List[str]:
        """Identify important columns based on naming patterns.
        
        Args:
            header_columns: List of column headers
            
        Returns:
            List of important column names
        """
        important_patterns = [
            r'id$', r'key$', r'name$', r'code$',  # Identifier patterns
            r'date$', r'time$', r'year$',          # Temporal patterns
            r'amount$', r'value$', r'total$',      # Value patterns
            r'category$', r'type$', r'class$'      # Classification patterns
        ]
        
        important_columns = []
        for col in header_columns:
            if any(re.search(pattern, col.lower()) for pattern in important_patterns):
                important_columns.append(col)
        
        return important_columns

    def _extract_row_context(self, columns: List[str], header_columns: List[str], 
                           metadata: Dict) -> Dict[str, Any]:
        """Extract context from a row.
        
        Args:
            columns: Row data
            header_columns: Column headers
            metadata: Column metadata
            
        Returns:
            Dictionary containing row context
        """
        context = {}
        for i, (col, val) in enumerate(zip(header_columns, columns)):
            col_type = metadata['column_types'].get(col, 'text')
            if col in metadata['important_columns']:
                context[col] = {
                    'value': val,
                    'type': col_type
                }
        return context

    def _should_start_new_group(self, row_context: Dict[str, Any], 
                              current_context: Dict[str, Any]) -> bool:
        """Determine if a new group should be started based on context changes.
        
        Args:
            row_context: Context of the current row
            current_context: Context of the current group
            
        Returns:
            True if a new group should be started
        """
        if not current_context:
            return False
            
        # Check for significant changes in important columns
        for col, data in row_context.items():
            if col not in current_context:
                return True
                
            current_val = current_context[col]['value']
            new_val = data['value']
            
            # Different handling based on column type
            if data['type'] == 'numeric':
                try:
                    current = float(current_val)
                    new = float(new_val)
                    if abs(new - current) > current * 0.5:  # 50% change
                        return True
                except ValueError:
                    if current_val != new_val:
                        return True
            elif data['type'] == 'categorical':
                if current_val != new_val:
                    return True
        
        return False

    def _update_context(self, current_context: Dict[str, Any], 
                       row_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update group context with new row context.
        
        Args:
            current_context: Current group context
            row_context: New row context
            
        Returns:
            Updated context
        """
        if not current_context:
            return row_context
            
        updated = current_context.copy()
        for col, data in row_context.items():
            if col not in updated:
                updated[col] = data
            elif data['type'] == 'numeric':
                try:
                    current = float(updated[col]['value'])
                    new = float(data['value'])
                    updated[col]['value'] = str((current + new) / 2)  # Update with average
                except ValueError:
                    updated[col] = data
            else:
                updated[col] = data
                
        return updated

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for inclusion in chunk.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
            
        context_parts = []
        for col, data in context.items():
            context_parts.append(f"{col}: {data['value']} ({data['type']})")
        
        return "Context: " + "; ".join(context_parts)

    def _split_regular_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split regular text into meaningful chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        logger.info(f"Starting regular text splitting. Text length: {text_length}")
        
        # Adaptive chunk size based on text length
        if text_length < chunk_size:
            # For very small texts, use a single chunk
            logger.info(f"Text is smaller than chunk size ({text_length} < {chunk_size}), using single chunk")
            return [text]
        
        # Calculate maximum chunks based on text length and overlap
        # This ensures we have enough chunks while preventing excessive splitting
        max_chunks = min(
            (text_length // (chunk_size - overlap)) + 2,  # Base calculation
            text_length // 100  # Upper limit to prevent too many small chunks
        )
        chunk_count = 0
        
        while start < text_length and chunk_count < max_chunks:
            # Find the end of the chunk
            end = min(start + chunk_size, text_length)
            
            # If we're not at the end of the text, find the last sentence boundary
            if end < text_length:
                # Look for sentence boundaries
                sentence_boundaries = ['.', '!', '?', '\n']
                for boundary in sentence_boundaries:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary != -1:
                        end = last_boundary + 1
                        break
            
            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
                logger.debug(f"Created chunk {len(chunks)}: length={len(chunk)} characters, start={start}, end={end}")
            
            # Move the start pointer forward, accounting for overlap
            # Only apply overlap if we're not at the end of the text
            if end < text_length:
                start = end - overlap
            else:
                start = end
            
            if len(chunks) % 100 == 0:  # Log progress every 100 chunks
                logger.info(f"Processed {len(chunks)} chunks so far. Current position: {start}/{text_length}")
        
        if chunk_count >= max_chunks:
            logger.info(f"Created {len(chunks)} chunks for text of length {text_length} characters")
        
        logger.info(f"Regular text splitting complete. Created {len(chunks)} chunks")
        return chunks