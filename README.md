# IBM CarbonSense powered by Watsonx

A RAG (Retrieval-Augmented Generation) system for carbon data analysis, powered by IBM watsonx.ai.

## Overview

CarbonSense is a sophisticated RAG system that processes and analyzes carbon-related data from various sources. It uses IBM watsonx.ai for generating embeddings and provides a robust framework for carbon data analysis.

## Architecture

### Core Components

1. **Data Processing Pipeline**

   - Processes Excel and text files from `Data_processed` directory
   - Organized into categories: industry, electricity, regional
   - Smart chunking strategy for optimal data processing
   - Enhanced metadata handling for better context preservation
2. **Embedding Generation**

   - Supports multiple models:
     - 30M parameter model (default)
     - 125M parameter model
     - Granite model (recommended for multilingual support)
   - Implements rate limiting and cooldown periods
   - Batch processing for efficiency
   - Improved error handling and retry mechanisms
3. **Storage Systems**

   - **Milvus**: Vector database for storing embeddings
     - Enhanced schema with versioning support
     - Improved indexing for faster searches
     - Data consistency checks and validation
   - **COS (Cloud Object Storage)**: Stores processed embeddings
   - **Local Storage**: Backup storage in `Embeddings` directory
4. **Query System**

   - Vector similarity search with configurable parameters
   - Context-aware results with source attribution
   - Support for multiple query modes
   - Enhanced result formatting and scoring
5. **Cleanup System**

   - Comprehensive cleanup of:
     - COS bucket contents
     - Milvus collections
     - Local embedding files
   - Model-specific cleanup options

## Implementation Details

### Data Processing

- **Smart Chunking**:

  - Excel-specific handling with header preservation
  - Structure-aware grouping of rows
  - Natural language boundaries for text files
  - Configurable chunk sizes and overlap
  - Enhanced metadata tracking
- **File Organization**:

  ```
  Data_processed/
  ├── electricity/
  │   ├── electricity EU.xlsx
  │   ├── electricity India.xlsx
  │   ├── electricity USA.xlsx
  │   ├── electricity Rest of the World.xlsx
  │   ├── Electricity_Global.xlsx
  │   └── electricity general industry.xlsx
  ├── industry/
  └── regional/
  ```

### Embedding Generation

- **Model Support**:

  - Multiple model options for different use cases
  - Configurable through command-line arguments
  - Automatic model selection based on requirements
  - Enhanced error handling and validation
- **Rate Limiting**:

  - Configurable cooldown periods
  - Batch processing for efficiency
  - Error handling with exponential backoff
  - Improved logging and monitoring

### Storage Implementation

- **Milvus Integration**:

  - Schema-based collection management
  - Efficient vector storage and retrieval
  - Automatic collection creation and cleanup
  - Enhanced data validation and consistency checks
  - Improved indexing for better search performance
- **COS Integration**:

  - Secure file storage and retrieval
  - Efficient batch operations
  - Error handling and retry mechanisms
  - Enhanced metadata tracking

## Setup and Configuration

1. **Environment Variables**

   ```env
   # IBM Cloud Object Storage
   COS_API_KEY=your_api_key
   COS_INSTANCE_ID=your_instance_id
   COS_ENDPOINT=your_endpoint
   BUCKET_NAME=your_bucket_name

   # Watsonx
   WATSONX_API_KEY=your_api_key
   WATSONX_URL=your_url
   WATSON_STUDIO_PROJECT_ID=your_project_id

   # Milvus
   MILVUS_GRPC_HOST=your_host
   MILVUS_GRPC_PORT=your_port
   MILVUS_CERT_PATH=your_cert_path
   ```
2. **Directory Structure**

   ```
   .
   ├── Data_processed/          # Source data files
   ├── Embeddings/             # Local embedding storage
   ├── src/                    # Source code
   ├── scripts/                # Utility scripts
   └── cleanup.py             # Cleanup utility
   ```

## Usage

### Generating Embeddings

1. **Process all files**:

   ```bash
   python -m carbonsense.main --mode generate
   ```
2. **Use specific model**:

   ```bash
   python -m carbonsense.main --mode generate --model granite
   ```
3. **Process individual files**:

   ```bash
   python -m carbonsense.main --mode generate --file path_to/file.xlsx
   ```

### Querying Data

1. **Basic query**:

   ```bash
   python -m carbonsense.main --mode query --query "your question here"
   ```
2. **Query with context**:

   ```bash
   python -m carbonsense.main --mode query --query "your question here" --show_context
   ```

### Verification

Check collection health and statistics:

```bash
python -m carbonsense.main --mode verify
```

### Cleanup

Run the cleanup script to remove all stored data:

```bash
python -m carbonsense.main --mode cleanup --model granite
```

## Logging

The system implements comprehensive logging:

- Console output with emojis for visual feedback
- File-based logging in `embedding_generation.log`
- Detailed progress tracking
- Error handling and reporting
- Enhanced debugging information

## Best Practices

1. **Data Organization**:

   - Keep files in appropriate categories
   - Maintain consistent naming conventions
   - Avoid consolidated files for better RAG performance
   - Use appropriate metadata for better context
2. **Processing**:

   - Use appropriate model for your needs
   - Monitor rate limits and adjust cooldown periods
   - Regular cleanup to maintain system health
   - Validate data consistency regularly
3. **Storage**:

   - Regular backups of important data
   - Monitor storage usage
   - Clean up unused embeddings
   - Verify data integrity periodically
4. **Querying**:

   - Use specific queries for better results
   - Include context when needed
   - Monitor query performance
   - Validate results for accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
