# Redis Caching Implementation Guide

This guide explains how to test the implementation of Redis caching for the CarbonSense project.

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Start a Redis server:

   ```bash
   docker run -p 6379:6379 redis
   ```

   Alternatively, if you have Redis installed locally, start the Redis server with:

   ```bash
   redis-server
   ```
3. Run the unit tests to verify schema validation:

   ```bash
   pytest -q
   ```

## Testing the Caching

1. Run a query:

   ```bash
   python -m src.carbonsense.main --mode crew_agent --query "What is the carbon footprint of 2 kg beef?"
   ```
2. Check the logs directory for the generated files:

   - `logs/1_parse.json`: Contains the parsed and normalized query
   - `logs/2_cache.json`: Shows if there was a cache hit or miss
   - `logs/3_milvus.json`, `logs/4_discovery.json`, `logs/5_serper.json`: Research results
   - `logs/6_harmonised.json`: Harmonized metrics
   - `logs/7_ranked.json`: Best ranked metric
   - `logs/8_usage.txt`: Usage logging
   - `0_final_answer.txt`: The final answer
3. Run the same query again to test caching:

   ```bash
   python -m src.carbonsense.main --mode crew_agent --query "What is the carbon footprint of 2 kg beef?"
   ```

   This time, the response should be much faster as it retrieves the result from the Redis cache.

## Key Components

- **CacheService**: Provides a Redis wrapper with get/set methods in `src/carbonsense/core/services/cache_service.py`
- **JSON Schema**: Defines the structure of carbon metrics in `src/carbonsense/core/config/common_schema.yaml`
- **Task Flow**: The new DAG structure is defined in `src/carbonsense/core/config/tasks.yaml`
- **Agents Configuration**: Defines all agents in `src/carbonsense/core/config/agents.yaml`
- **Crew Agent**: The main implementation is in `src/carbonsense/core/crew_agent.py`

## Cache Implementation Details

1. **Parse & Normalize**: A query is parsed to extract product and quantity information
2. **Cache Lookup**: Checks if the product is already in the Redis cache
3. **Conditional Skip**: If a cache hit occurs, the research, harmonizing, and ranking tasks are skipped
4. **Answer Formatting**: The cached result is formatted considering the specific quantity in the query

## Schema Validation

The carbon metric schema enforces a consistent structure for all metrics:

- `value`: The numeric carbon footprint value
- `emission_unit`: The unit of measurement (kg CO2e or g CO2e)
- `product_unit`: The product unit (per kg, per item, etc.)
- `source`: The source of the data
- `confidence`: A confidence score between 0 and 1
