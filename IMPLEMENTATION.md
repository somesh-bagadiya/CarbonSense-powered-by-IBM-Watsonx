# CarbonSense Implementation Details

## Technical Specifications

### 1. Data Processing Implementation

#### File Ingestion

```python
# Supported file types and their handlers
file_handlers = {
    '.xlsx': ExcelHandler,  # For Microsoft Excel spreadsheets
    '.txt': TextHandler,    # For plain text files
    '.csv': CSVHandler      # For comma-separated value files
}

# File validation
def validate_file(file_path):
    # Check file type - Ensures the file format is supported
    # Verify file integrity - Checks if the file is not corrupted
    # Validate content structure - Ensures the file contains expected data format
    # Check for required metadata - Verifies necessary information is present
```

#### Preprocessing Pipeline

```python
# Text standardization
def standardize_text(text):
    # Unit normalization - Converts different unit formats to standard forms
    units = {
        'kg': 'kilogram',      # Converts 'kg' to 'kilogram'
        'co2': 'carbon dioxide', # Standardizes CO2 references
        '%': 'percent'         # Converts percentage symbols to words
    }
  
    # Special character handling - Removes or standardizes special characters
    # Context marker addition - Adds markers to identify important information
    # Excel-specific formatting - Preserves Excel-specific data formatting
```

#### Chunking Implementation

```python
# Excel chunking configuration
excel_config = {
    'rows_per_chunk': 5,
    'include_headers': True,
    'preserve_relationships': True,
    'handle_merged_cells': True
}

# Text chunking configuration
text_config = {
    'chunk_size': 1000,
    'overlap': 200,
    'preserve_sentences': True,
    'context_window': 100
}
```

### 2. Embedding Generation

#### Model Configuration

```python
# Model specifications - Different AI models for processing text
models = {
    '30m': {  # Small model (30 million parameters)
        'name': 'ibm/slate-30m-english-rtrvr-v2',
        'max_tokens': 512,     # Maximum text length it can process
        'batch_size': 32       # Number of items processed at once
    },
    '125m': { # Medium model (125 million parameters)
        'name': 'ibm/slate-125m-english-rtrvr-v2',
        'max_tokens': 1024,    # Can handle longer text
        'batch_size': 16       # Smaller batch size due to larger model
    },
    'granite': { # Large multilingual model
        'name': 'ibm/granite-embedding-278m-multilingual',
        'max_tokens': 2048,    # Can process very long text
        'batch_size': 8        # Smallest batch size due to model size
    }
}

# Rate limiting configuration - Controls API request frequency
rate_limit = {
    'min_call_interval': 0.5,  # Minimum time between API calls (seconds)
    'batch_size': 5,          # Number of items processed in each batch
    'max_retries': 3,         # Maximum number of retry attempts
    'backoff_factor': 2       # Multiplier for retry delay
}
```

### 3. Storage Implementation

#### Milvus Schema

```python
# Collection schema definition - Structure of the database
schema = {
    'fields': [
        {'name': 'id', 'dtype': 'VARCHAR', 'is_primary': True},  # Unique identifier
        {'name': 'text', 'dtype': 'VARCHAR', 'max_length': 65535}, # Content text
        {'name': 'embedding', 'dtype': 'FLOAT_VECTOR', 'dim': 768}, # AI-generated vector
        {'name': 'source_file', 'dtype': 'VARCHAR'},  # Original file name
        {'name': 'model_type', 'dtype': 'VARCHAR'},   # AI model used
        {'name': 'version', 'dtype': 'VARCHAR'},      # Data version
        {'name': 'chunk_index', 'dtype': 'INT64'},    # Position in document
        {'name': 'total_chunks', 'dtype': 'INT64'},   # Total number of chunks
        {'name': 'timestamp', 'dtype': 'INT64'},      # Processing time
        {'name': 'metadata', 'dtype': 'VARCHAR'},     # Additional information
        {'name': 'processing_info', 'dtype': 'VARCHAR'}, # Processing details
        {'name': 'row_range', 'dtype': 'VARCHAR'},    # Excel row range
        {'name': 'column_info', 'dtype': 'VARCHAR'},  # Column metadata
        {'name': 'sheet_name', 'dtype': 'VARCHAR'}    # Excel sheet name
    ],
    'description': 'Carbon footprint data collection'
}

# Index configuration - Optimizes search performance
index_config = {
    'metric_type': 'L2',        # Distance calculation method
    'index_type': 'IVF_FLAT',   # Index structure type
    'params': {'nlist': 1024}   # Number of clusters for indexing
}
```

#### COS Integration

```python
# Cloud Object Storage configuration
cos_config = {
    'endpoint': 'COS_ENDPOINT',
    'api_key': 'COS_API_KEY',
    'instance_id': 'COS_INSTANCE_ID',
    'bucket_name': 'BUCKET_NAME'
}

# File operations
def upload_to_cos(file_path, object_name):
    # File upload with metadata
    # Error handling
    # Progress tracking
```

### 4. Query System Implementation

#### CarbonAgent

```python
class CarbonAgent:
    def __init__(self, config):
        self.config = config
        self.milvus = MilvusService(config)      # Vector database service
        self.watsonx = WatsonxService(config)    # AI model service
        self.discovery = DiscoveryService(config) # Web search service
        self.confidence_threshold = 0.6  # Minimum confidence score for results
  
    def process_query(self, query):
        # 1. Extract product and quantity - Identify key information
        # 2. Search Milvus - Find relevant data in database
        # 3. Evaluate confidence - Check result quality
        # 4. Web search fallback if needed - Search web if local data insufficient
        # 5. Generate response - Create final answer
```

#### Web Search Integration

```python
# Watson Discovery configuration - Web search service settings
discovery_config = {
    'api_key': 'WATSON_DISCOVERY_API_KEY',      # Authentication key
    'url': 'WATSON_DISCOVERY_URL',              # Service endpoint
    'project_id': 'WATSON_DISCOVERY_PROJECT_ID', # Project identifier
    'confidence_threshold': 0.7                  # Minimum confidence for web results
}
```

## Error Handling

### 1. Data Processing Errors

```python
# Error handling strategies - How different errors are managed
error_handlers = {
    'file_not_found': handle_file_not_found,     # Missing file handling
    'invalid_format': handle_invalid_format,     # Wrong file format handling
    'processing_error': handle_processing_error, # Processing failure handling
    'chunking_error': handle_chunking_error     # Text splitting error handling
}
```

### 2. API Errors

```python
# API error handling - Managing external service errors
def handle_api_error(error):
    # Rate limit handling - Managing request frequency limits
    # Authentication errors - Handling login/access issues
    # Network issues - Managing connection problems
    # Retry logic - Attempting failed operations again
```

### 3. Storage Errors

```python
# Storage error handling - Managing database issues
def handle_storage_error(error):
    # Connection issues - Database connection problems
    # Data consistency - Ensuring data integrity
    # Backup strategies - Data recovery plans
    # Recovery procedures - System restoration steps
```

## Performance Optimization

### 1. Batch Processing

```python
# Batch configuration - Group processing settings
batch_config = {
    'size': 100,        # Number of items per batch
    'timeout': 30,      # Maximum processing time (seconds)
    'max_retries': 3    # Maximum retry attempts
}
```

### 2. Caching

```python
# Cache configuration - Temporary storage settings
cache_config = {
    'ttl': 3600,        # Time to live (seconds)
    'max_size': 1000,   # Maximum cache size
    'strategy': 'LRU'   # Least Recently Used replacement policy
}
```

### 3. Parallel Processing

```python
# Parallel processing setup - Concurrent execution settings
parallel_config = {
    'max_workers': 4,    # Maximum concurrent processes
    'chunk_size': 100,   # Items per parallel batch
    'timeout': 300       # Maximum execution time (seconds)
}
```

## Monitoring and Logging

### 1. Logging Configuration

```python
# Logging setup - System activity recording
logging_config = {
    'level': 'INFO',     # Log detail level
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': [
        'console',       # Screen output
        'file'          # File storage
    ]
}
```

### 2. Metrics Collection

```python
# Metrics configuration - Performance measurement
metrics_config = {
    'endpoint': 'metrics_endpoint',  # Data collection point
    'interval': 60,                  # Collection frequency (seconds)
    'tags': ['environment', 'version'] # Data categorization
}
```

### 3. Alerting

```python
# Alert configuration - System monitoring
alert_config = {
    'thresholds': {
        'error_rate': 0.01,     # Maximum acceptable error percentage
        'latency': 1000,        # Maximum response time (milliseconds)
        'memory_usage': 0.8     # Maximum memory utilization
    },
    'notifications': ['email', 'slack']  # Alert delivery methods
}
```
