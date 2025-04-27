# CarbonSense

An intelligent, AI-based carbon footprint assessment system powered by IBM Watsonx.

## System Overview

CarbonSense uses a modular, multi-agent RAG (Retrieval-Augmented Generation) system to analyze carbon footprints for a wide variety of products, activities, and scenarios. The system processes natural language queries, extracts relevant entities, searches multiple data sources, and generates standardized, accurate carbon footprint assessments.

## Key Features

- **Multiple Query Intents**: Supports estimation, comparison, suggestions, lifecycle analysis, and myth busting
- **Comprehensive Categorization**: Classifies all outputs into Food & Diet, Energy Use, Mobility, Purchases, or Miscellaneous
- **Multi-Source Research**: Utilizes vector database (Milvus), document search (Watson Discovery), and web search (Serper)
- **Standardized Output Format**: Consistent JSON response structure suitable for dashboard integration
- **Confidence Scoring**: Assigns confidence scores based on source credibility, data recency, and methodological rigor

## Usage Examples

### Carbon Estimation

```bash
python -m src.carbonsense.main --mode crew_agent --query "What's the carbon footprint of eating a cheeseburger?"
```

### Product Comparison

```bash
python -m src.carbonsense.main --mode crew_agent --query "Which has a higher carbon footprint, taking a bus or driving my car for 10 miles?"
```

### Suggestions/Recommendations

```bash
python -m src.carbonsense.main --mode crew_agent --query "How can I reduce the carbon footprint of my commute?"
```

### Lifecycle Analysis

```bash
python -m src.carbonsense.main --mode crew_agent --query "What's the lifecycle carbon footprint of an iPhone?"
```

### Myth Busting

```bash
python -m src.carbonsense.main --mode crew_agent --query "Is it true that local food always has a lower carbon footprint?"
```

## Response Format

All responses follow a standardized JSON format:

```json
{
  "emission": "12.5 kg CO₂e",
  "method": "Includes emissions from manufacturing, delivery, and disposal. Based on NREL lifecycle database.",
  "category": "Purchases",
  "sources": [
    {"title": "NREL Product LCA", "url": "https://nrel.gov/..."},
    {"title": "GHG Protocol", "url": "https://ghgprotocol.org/..."}
  ]
}
```

## Agentic Pipeline

The system uses a modular, multi-agent pipeline to process queries through several layers:

1. **Query Understanding Layer**
   - Query classification (intent & category)
   - Entity extraction (products, quantities, locations)
   - Unit normalization (standardization of measurements)

2. **Retrieval & Research Layer**
   - Milvus vector database search
   - Watson Discovery document search
   - Serper web search

3. **Carbon Estimation & Synthesis**
   - Unit harmonization (standardizing units)
   - Carbon footprint calculation
   - Metric ranking (source reliability evaluation)

4. **Intent-Specific Processing**
   - Comparison formatting
   - Recommendation generation
   - Explanation/myth busting

5. **Response Generation**
   - Standardized JSON output formatting
   - Usage logging

For a detailed visualization of the agentic pipeline, see [agentic_pipeline.md](docs/agentic_pipeline.md).

## Technical Configuration

Key configuration files:

- `config/agents.yaml`: Agent role definitions
- `config/tasks.yaml`: Task descriptions and workflows
- `config/common_schema.yaml`: JSON schema for carbon metrics
- `config/category_keywords.yaml`: Keyword mappings for categorization

## Running the System

Basic command:

```bash
python -m src.carbonsense.main --mode crew_agent --query "YOUR QUERY HERE"
```

Options:

- `--show_context`: Show context and sources
- `--debug`: Enable debug mode
- `--no_cache`: Disable caching
- `--sequential`: Use sequential process instead of hierarchical
- `--store_thoughts`: Store agent thoughts and reasoning in log files
- `--output_file`: Path to save results in JSON format 