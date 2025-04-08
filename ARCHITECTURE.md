# CarbonSense Architecture

## System Overview

CarbonSense is a sophisticated RAG (Retrieval-Augmented Generation) system designed for carbon footprint analysis. It combines multiple AI models, vector databases, and web search capabilities to deliver accurate and context-aware responses to carbon-related queries.

## Core Components

### 1. Data Processing Pipeline

#### Document Processing
- **Smart Text Preprocessing**:
  - Standardizes units (kg, CO2, percentages)
  - Removes redundant information
  - Adds context markers for important data
  - Preserves structural information

- **Excel-Specific Processing**:
  - Maintains header information
  - Preserves data relationships
  - Handles numeric formatting
  - Extracts column context

#### Chunking Strategy
- **Excel-Optimized Chunking**:
  - **Row-Based Processing**:
    - Groups related rows together (default: 5 rows per chunk)
    - Preserves row relationships and context
    - Maintains data integrity across chunks
    - Handles merged cells and formatting

  - **Header Context**:
    - Includes column headers in each chunk
    - Preserves column relationships
    - Maintains data type information
    - Adds context markers for important columns

  - **Column Analysis**:
    - Identifies key columns (e.g., product, quantity, carbon footprint)
    - Preserves numeric formatting
    - Handles unit conversions
    - Maintains data relationships

  - **Metadata Enrichment**:
    - Adds sheet name context
    - Includes column descriptions
    - Preserves data validation rules
    - Tracks cell references

- **Text File Chunking** (for non-Excel files):
  - Configurable chunk sizes (default: 1000 characters)
  - Overlapping chunks (default: 200 characters)
  - Sentence boundary preservation
  - Context-aware grouping

### 2. Embedding Generation

#### Model Support
- **Multiple Model Options**:
  - **30M Parameter Model**:
    - `ibm/slate-30m-english-rtrvr-v2`
    - Default model for English text
    - Optimized for general retrieval tasks
    - 384-dimensional embeddings

  - **125M Parameter Model**:
    - `ibm/slate-125m-english-rtrvr-v2`
    - Enhanced performance model
    - Better for complex queries
    - Improved context understanding
    - 768-dimensional embeddings

  - **Granite Model**:
    - `ibm/granite-embedding-278m-multilingual`
    - Recommended for multilingual support
    - Advanced reasoning capabilities
    - Better handling of complex queries
    - Supports multiple languages
    - 768-dimensional embeddings

#### Rate Limiting
- Configurable cooldown periods (default: 0.5 seconds)
- Batch processing (default: 5 items per batch)
- Exponential backoff for retries
- Progress tracking and logging

### 3. Storage Systems

#### Milvus Integration
- **Collection Management**:
  - Schema-based organization
  - Version control
  - Data validation
  - Automatic collection creation

- **Indexing Strategy**:
  - Vector similarity search (L2 metric)
  - IVF_FLAT index type
  - Filtering indexes for metadata
  - Optimized query performance
  - 1024 clusters for indexing

#### Cloud Object Storage (COS)
- Secure file storage
- Efficient batch operations
- Error handling
- Metadata tracking
- Automatic backup and recovery

### 4. Query System

#### CarbonAgent Implementation
- **Query Processing**:
  1. Product and quantity extraction using Watsonx
  2. Milvus vector search with confidence scoring
  3. Confidence-based web search fallback (threshold: 0.6)
  4. Context-aware response generation

- **Response Generation**:
  - Carbon footprint per unit
  - Total impact calculation
  - Environmental considerations
  - Source attribution
  - Confidence scoring

#### Web Search Integration
- **Watson Discovery Service**:
  - Dynamic web search for latest data
  - Project-based organization
  - Version-controlled API
  - Confidence-based filtering

- **Integration Features**:
  - Real-time web search when local data is insufficient
  - Confidence-based fallback mechanism
  - Source attribution and verification
  - Automatic content filtering and relevance scoring

- **Trusted Sources**:
  - Government databases
  - Research institutions
  - Industry reports
  - Environmental organizations

## System Flow

1. **Data Ingestion**:
   - File type detection
   - Format validation
   - Metadata extraction
   - Directory structure management

2. **Processing Pipeline**:
   - Text preprocessing
   - Excel-specific handling
   - Chunking optimization
   - Embedding generation
   - Rate limit management

3. **Storage Management**:
   - Vector database updates
   - Cloud storage synchronization
   - Index maintenance
   - Version control
   - Backup procedures

4. **Query Processing**:
   - Query analysis
   - Multi-source search
   - Result aggregation
   - Response generation
   - Confidence evaluation

## Best Practices

### 1. Data Organization
- Keep files in appropriate categories
- Use consistent naming conventions
- Avoid consolidated files
- Include relevant metadata
- Maintain directory structure

### 2. Processing
- Choose appropriate model for your needs
- Monitor rate limits
- Regular cleanup
- Validate data consistency
- Track processing progress

### 3. Querying
- Be specific in queries
- Include quantities when relevant
- Use context when needed
- Verify results
- Check confidence scores

### 4. Maintenance
- Regular system verification
- Monitor API quotas
- Update dependencies
- Backup data regularly
- Check log files 