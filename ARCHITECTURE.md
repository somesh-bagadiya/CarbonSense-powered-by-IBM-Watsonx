# CarbonSense Architecture

## System Overview

CarbonSense is a sophisticated RAG (Retrieval-Augmented Generation) system designed for carbon footprint analysis. It combines multiple AI models, vector databases, and web search capabilities to deliver accurate and context-aware responses to carbon-related queries. The system now includes multi-agent workflows powered by CrewAI, voice input capabilities, and a web dashboard interface built with FastAPI.

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

### 4. Query Systems

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

#### CrewAI Multi-Agent System

- **Agent Architecture**:

  - Fully YAML-configured agents and workflows
  - IBM Watsonx foundation models for agent reasoning
  - Role-based agent design with specialized capabilities
  - Agent-specific tools for task execution
- **Agent Roles and Specialization**:

  - **Parser Agent**:

    - Extracts and normalizes product/quantity information
    - Handles different unit formats and conversions
    - Standardizes product names
    - Identifies query intent
  - **Research Agent**:

    - Collects data from Milvus vector database
    - Searches Watson Discovery for web results
    - Integrates domain knowledge with search results
    - Prioritizes trusted sources
  - **Harmonization Agent**:

    - Standardizes metrics across sources
    - Resolves conflicting information
    - Converts between different carbon measurement units
    - Calculates confidence scores for findings
  - **Writer Agent**:

    - Creates final user-friendly responses
    - Formats carbon calculations
    - Provides environmental context
    - Cites information sources
- **Workflow Orchestration**:

  - **Hierarchical Processing**:

    - YAML-defined task dependencies
    - Conditional execution paths
    - Sequential or parallel task execution
    - Error handling and fallback strategies
  - **Thought Process Management**:

    - Captures agent reasoning steps
    - Logs intermediate results
    - Enables transparency and debugging
    - Provides insights in the web interface
  - **Tool Integration**:

    - Custom tools for specific agent roles
    - Dynamic tool selection based on task
    - Tool-based semantic function calling
    - Error handling and retry mechanisms
- **YAML-Based Configuration**:

  - `agents.yaml`: Defines all agent roles, goals, backstories, and tools
  - `tasks.yaml`: Defines task workflow, dependencies, and execution logic
  - `prompts.yaml`: Contains system prompts for different agent roles
  - `common_schema.yaml`: Defines shared data schemas for consistency

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

  - Government databases (EPA, DOE)
  - Research institutions
  - Industry sustainability reports
  - Environmental organizations

### 5. Voice Input Processing

- **Audio Recording**:

  - Configurable recording duration
  - Multiple audio device support
  - Interactive device selection
  - Format standardization (WAV)
  - Powered by sounddevice library
- **Watson Speech-to-Text Integration**:

  - High-accuracy transcription
  - Background noise handling
  - Multiple language support
  - Confidence scoring
- **Stream Processing**:

  - Real-time audio handling
  - Error resilience
  - Resource cleanup
  - Temporary file management

### 6. Web Interface

- **Dashboard Implementation**:

  - Carbon footprint visualization
  - Goal tracking
  - Achievement badges
  - Weekly trends analysis
- **Interactive Query Interface**:

  - Real-time streaming responses
  - Agent thought visualization
  - Query history tracking
  - Source attribution display
- **FastAPI Backend**:

  - Asynchronous request handling
  - Server-sent events for streaming
  - Background task processing
  - Error handling and recovery
- **Responsive Frontend**:

  - Mobile-friendly design
  - Interactive visualizations with Chart.js
  - Real-time updates
  - Progressive web app capabilities

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

   - User query submission (text or voice)
   - Parser agent analysis and normalization
   - Research agent data collection from multiple sources
   - Harmonization agent data standardization
   - Writer agent response generation
   - Confidence evaluation and source attribution
5. **Voice Input Handling**:

   - Audio device selection
   - Recording management
   - Transcription processing
   - Text query conversion
   - Temporary file cleanup
6. **Web Interface Flow**:

   - User dashboard presentation
   - Query submission handling
   - Agent thought streaming
   - Result visualization
   - User activity tracking
   - Goal progress updates

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

### 4. Web Interface

- Optimize for mobile and desktop
- Implement progressive loading
- Handle network disruptions
- Provide clear user feedback
- Monitor server performance

### 5. Maintenance

- Regular system verification
- Monitor API quotas
- Update dependencies
- Backup data regularly
- Check log files
- Run periodic cleanup

### 6. Agent Development

- Follow YAML configuration standards
- Test individual agents before integration
- Keep prompts focused and specific
- Document agent capabilities
- Monitor agent performance metrics
