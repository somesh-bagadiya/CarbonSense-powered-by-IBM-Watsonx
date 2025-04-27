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
        # 2. Normalize query - Standardize product names and quantities
        # 3. Search Milvus - Find relevant data in database
        # 4. Evaluate confidence - Check result quality
        # 5. Web search fallback if needed - Search web if local data insufficient
        # 6. Generate response - Create final answer
```

#### CrewAI Multi-Agent System

##### Core Implementation

```python
# CrewAgent manager implementation
class CrewAgentManager:
    def __init__(self, config, debug_mode=False, use_hierarchical=True, store_thoughts=False):
        self.config = config
        self.debug_mode = debug_mode
        self.use_hierarchical = use_hierarchical
        self.store_thoughts = store_thoughts
        self.yaml_manager = YAMLManager(self._get_config_dir())
        self.tools = self._initialize_tools()
        
    def process_query(self, query, show_context=False):
        # 1. Load agent definitions from YAML
        # 2. Create agents with specialized tools
        # 3. Build task workflow from YAML configuration
        # 4. Execute task workflow (hierarchical or sequential)
        # 5. Collect and process agent outputs
        # 6. Return consolidated response with context if requested
```

##### YAML Configuration Examples

**Agent Definitions (agents.yaml)**

```yaml
agents:
  - id: parser
    name: Query Parser
    goal: Extract and normalize product and quantity from user queries
    backstory: |
      You are a specialist in understanding carbon footprint queries and extracting 
      key information. You have extensive knowledge of product names, quantity units, 
      and can normalize varied user inputs into standardized formats.
    verbose: false
    allow_delegation: false
    tools: [normalize_product, extract_quantity]
    
  - id: researcher
    name: Carbon Researcher
    goal: Find accurate carbon footprint information from multiple trusted sources
    backstory: |
      You are an environmental researcher specializing in carbon footprint data collection.
      You have access to a vector database of carbon metrics and can search the web for
      the latest information when needed.
    verbose: false
    allow_delegation: false
    tools: [search_milvus, search_discovery, search_web]
    
  - id: harmonizer
    name: Data Harmonizer
    goal: Standardize carbon metrics from different sources into consistent values
    backstory: |
      You are an expert in environmental data standardization. You can convert between
      different units, resolve conflicts between data sources, and provide confidence
      scores for the information.
    verbose: false
    allow_delegation: false
    tools: [convert_units, standardize_metrics]
    
  - id: writer
    name: Carbon Response Writer
    goal: Create clear, accurate and helpful responses about carbon footprint data
    backstory: |
      You are a science communicator specializing in environmental topics. You can
      explain complex carbon footprint concepts in simple terms, provide context,
      and format responses in a user-friendly way.
    verbose: false
    allow_delegation: false
    tools: [format_carbon_footprint]
```

**Task Workflow (tasks.yaml)**

```yaml
tasks:
  - id: parse_query
    agent: parser
    input: "{{ query }}"
    description: Extract product and quantity from user query
    async: false
    output: normalized_query
    
  - id: research_data
    agent: researcher
    description: Search for carbon footprint data from multiple sources
    depends_on: [parse_query]
    input: "{{ parse_query.output }}"
    async: false
    output: research_results
    
  - id: harmonize_data
    agent: harmonizer
    description: Standardize metrics across different data sources
    depends_on: [research_data]
    input: "{{ research_data.output }}"
    async: false
    output: harmonized_metrics
    
  - id: generate_response
    agent: writer
    description: Create a response based on the harmonized data
    depends_on: [harmonize_data, parse_query]
    input: "{{ harmonize_data.output }}"
    input_context: "{{ parse_query.output }}"
    async: false
    output: final_response
```

**Prompt Template (prompts.yaml)**

```yaml
prompts:
  parser:
    system: |
      You are a Carbon Query Parser specializing in extracting product and quantity information.
      Follow these guidelines:
      1. Identify the main product being queried (e.g., "beef", "driving", "electricity")
      2. Extract quantity information and units (e.g., "2 kg", "10 miles", "5 hours")
      3. Normalize product names to standard forms
      4. Identify query intent (e.g., comparison, single product, activity)
      
      Output format:
      ```json
      {
        "product": "normalized_product_name",
        "quantity": number,
        "unit": "standardized_unit",
        "query_type": "comparison|single|activity"
      }
      ```
      
  researcher:
    system: |
      You are a Carbon Footprint Researcher with access to multiple data sources.
      Your task is to find accurate carbon footprint information for the requested product.
      
      Follow these steps:
      1. Search the Milvus vector database for relevant carbon data
      2. If database results are insufficient, search Watson Discovery
      3. For each result, note the source and confidence level
      4. Prioritize recent, peer-reviewed, and official sources
      
      Output format:
      ```json
      {
        "results": [
          {
            "product": "product_name",
            "carbon_value": number,
            "unit": "kg_co2e_per_unit",
            "source": "source_name",
            "confidence": 0.0-1.0,
            "year": publication_year
          }
        ]
      }
      ```
```

##### YAML Manager Implementation

```python
# YAML configuration manager
class YAMLManager:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self._load_configuration()
        
    def _load_configuration(self):
        """Load all configuration files."""
        self.agents_config = self._load_yaml('agents.yaml')
        self.tasks_config = self._load_yaml('tasks.yaml')
        self.prompts_config = self._load_yaml('prompts.yaml')
        self.schemas_config = self._load_yaml('common_schema.yaml')
        
    def _load_yaml(self, filename):
        """Load and parse a YAML file."""
        file_path = os.path.join(self.config_dir, filename)
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
            
    def get_agent_config(self, agent_id):
        """Get configuration for a specific agent."""
        for agent in self.agents_config.get('agents', []):
            if agent.get('id') == agent_id:
                return agent
        return None
        
    def get_task_workflow(self):
        """Get the full task workflow configuration."""
        return self.tasks_config.get('tasks', [])
        
    def get_prompt_template(self, agent_id):
        """Get prompt template for a specific agent."""
        return self.prompts_config.get('prompts', {}).get(agent_id, '')
```

##### Tool Integration

```python
# Tool initialization
def initialize_tools(config):
    """Initialize all tools required by agents."""
    tools = {}
    
    # Parser tools
    tools['normalize_product'] = NormalizeProductTool()
    tools['extract_quantity'] = ExtractQuantityTool()
    
    # Researcher tools
    tools['search_milvus'] = MilvusSearchTool(config)
    tools['search_discovery'] = DiscoverySearchTool(config)
    tools['search_web'] = WebSearchTool(config)
    
    # Harmonizer tools
    tools['convert_units'] = UnitConversionTool()
    tools['standardize_metrics'] = MetricStandardizationTool()
    
    # Writer tools
    tools['format_carbon_footprint'] = FormatCarbonFootprintTool()
    
    return tools

# Example tool implementation
class MilvusSearchTool:
    def __init__(self, config):
        self.config = config
        self.milvus = MilvusService(config)
    
    def run(self, product, top_k=5):
        """Search Milvus for product carbon footprint data."""
        query = f"carbon footprint of {product}"
        results = self.milvus.semantic_search(query, top_k)
        
        processed_results = []
        for result in results:
            # Extract and process information from search results
            # Calculate confidence score based on relevance
            processed_results.append({
                "product": product,
                "carbon_value": extracted_value,
                "unit": extracted_unit,
                "source": result.get("source_file"),
                "confidence": result.get("score") / 100.0,
            })
            
        return processed_results
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

### 5. Voice Input System

```python
# Audio recording configuration
audio_config = {
    'sample_rate': 44100,       # Sample rate in Hz
    'channels': 1,              # Mono recording
    'default_duration': 10,     # Default recording duration in seconds
    'dtype': 'int16'            # Audio data type
}

# Speech-to-Text configuration
stt_config = {
    'api_key': 'IBM_STT_API_KEY',
    'url': 'IBM_STT_URL',
    'model': 'en-US_BroadbandModel',
    'content_type': 'audio/wav'
}

# Recording function
def record_audio(duration, sample_rate, channels, device_index=None):
    """Records audio from the microphone and saves it to a temporary file."""
    # Initialize recording settings
    device_info = f" using device index {device_index}" if device_index is not None else " using default device"
    num_frames = int(duration * sample_rate)
    
    # Start recording
    recording = sd.rec(num_frames, samplerate=sample_rate, channels=channels, dtype='int16', device=device_index)
    sd.wait()  # Wait until recording is finished
    
    # Save as WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file_path = temp_file.name
    temp_file.close()
    
    wav.write(temp_file_path, sample_rate, recording)
    return temp_file_path
    
# Transcription function
def transcribe_audio(config, audio_file_path):
    """Transcribes audio using IBM Watson Speech-to-Text."""
    # Initialize STT client
    authenticator = IAMAuthenticator(config.get_stt_config().get("api_key"))
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(config.get_stt_config().get("url"))
    
    # Transcribe the audio
    with open(audio_file_path, 'rb') as audio_file:
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            model='en-US_BroadbandModel'
        ).get_result()
    
    # Process transcription result
    if response['results']:
        transcript = response['results'][0]['alternatives'][0]['transcript']
        return transcript
    else:
        return ""
```

### 6. Web Interface Implementation

```python
# FastAPI app configuration
app = FastAPI(
    title="CarbonSense Dashboard",
    description="A dashboard for tracking and analyzing carbon footprint data",
    version="1.0.0"
)

# Template and static file configuration
templates_path = Path(__file__).parent / "templates"
static_path = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_path))
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Dashboard route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main dashboard page."""
    data = get_sample_data()
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "data": data}
    )

# API endpoint for streaming agent thoughts
@app.get("/api/stream-thoughts")
async def stream_thoughts(request: Request, query: str):
    """Stream agent thoughts as Server-Sent Events."""
    # Generate unique request ID
    request_id = f"request_{int(time.time() * 1000)}"
    
    # Create dedicated thought queue
    thought_queues[request_id] = queue.Queue()
    
    # Start background processing
    thread = threading.Thread(
        target=simulate_agent_thoughts,
        args=(request_id, query),
        daemon=True
    )
    thread.start()
    
    # Define the SSE generator
    async def event_generator():
        # Setup SSE connection
        yield "retry: 1000\n\n"
        yield f"id: {request_id}\n"
        yield f"data: {json.dumps({'type': 'info', 'content': 'Connection established'})}\n\n"
        
        # Process thoughts from queue
        while True:
            try:
                thought_data = thought_queues[request_id].get(timeout=0.5)
                
                if thought_data == "DONE":
                    yield f"id: {int(time.time() * 1000)}\n"
                    yield f"data: {json.dumps({'type': 'complete', 'content': 'Processing completed'})}\n\n"
                    break
                
                # Send the thought data as a server-sent event
                event_id = int(time.time() * 1000)
                yield f"id: {event_id}\n"
                yield f"data: {json.dumps(thought_data)}\n\n"
                
            except queue.Empty:
                # Send keep-alive ping
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)
    
    # Return streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"
        }
    )
    
# Query endpoint
@app.post("/api/query")
async def query_carbon(request: Request, background_tasks: BackgroundTasks):
    """Process a carbon footprint query."""
    data = await request.json()
    query = data.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Process with CrewAgentManager
    crew_manager = CrewAgentManager(config=ConfigManager())
    result = crew_manager.process_query(query, show_context=True)
    
    return {
        "answer": result["response"],
        "confidence": result.get("confidence", 0.8),
        "sources": result.get("context", {}).get("sources", []) if "context" in result else []
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

### 4. Audio Processing Errors

```python
# Audio recording error handling
def handle_audio_error(error):
    # Device access - Microphone permission issues
    # Recording timeout - Handle duration limits
    # Format conversion - Audio format problems
    # File I/O - Temporary file access issues
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

### 4. Web Interface Optimization

```python
# FastAPI optimization
fastapi_config = {
    'workers': 4,                # Number of uvicorn workers
    'log_level': "info",         # Logging level
    'timeout_keep_alive': 5,     # Keep-alive timeout
    'limit_concurrency': 100,    # Maximum concurrent connections
    'backlog': 128              # Connection queue size
}

# Frontend optimization
frontend_optimization = {
    'compression': True,         # Enable gzip compression
    'caching': {                 # Browser caching settings
        'static_files': 86400,   # Static files cache (1 day)
        'api_responses': 300     # API responses cache (5 minutes)
    },
    'lazy_loading': True,        # Enable lazy loading for images
    'minify': True               # Minify HTML/CSS/JS files
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

### 4. Agent Thought Logging

```python
# Agent thought logging configuration
thought_logging_config = {
    'enabled': True,            # Enable thought logging
    'log_dir': 'logs',          # Directory for thought logs
    'format': 'json',           # Log format (json or text)
    'include_timestamps': True, # Include timestamps
    'retention_days': 7,        # Keep logs for 7 days
    'log_level': 'INFO'         # Logging level
}

# Thought tracking function
def log_agent_thought(agent_id, thought_type, content):
    """Log an agent's thought or action."""
    # Create timestamp
    timestamp = datetime.now().isoformat()
    
    # Create log entry
    log_entry = {
        "agent_id": agent_id,
        "timestamp": timestamp,
        "type": thought_type,
        "content": content
    }
    
    # Determine log file path
    log_file = f"logs/agent_{agent_id}_{datetime.now().strftime('%Y%m%d')}.json"
    
    # Write log entry
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")
```
