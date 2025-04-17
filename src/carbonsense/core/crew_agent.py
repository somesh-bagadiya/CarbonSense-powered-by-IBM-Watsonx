import logging
import os
import yaml
from typing import Dict, List, Any, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool
from langchain.callbacks.manager import CallbackManager
from langchain.llms.base import LLM as LangChainLLM

from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..services.litellm_watsonx_service import LiteLLMWatsonxService
from ..services.discovery_service import DiscoveryService
from ..utils.logger import setup_logger
from ..services.cache_service import CacheService
from jsonschema import validate
from ..config.common_schema import carbon_metric

# Import tools
from .tools import CarbonResearchTool, WebSearchTool, initialize_tools

# Set up logger with colored output
logger = setup_logger(__name__)

@CrewBase
class CarbonSenseCrew:
    """Carbon footprint analysis crew using CrewAI's recommended approach."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, 
                 use_cache: bool = False, use_hierarchical: bool = True,
                 store_thoughts: bool = False):
        """Initialize the CarbonSense crew.
        
        Args:
            config: Configuration manager instance
            debug_mode: Whether to enable debug mode
            use_cache: Whether to use caching
            use_hierarchical: Whether to use hierarchical task execution
            store_thoughts: Whether to store agent thoughts and reasoning in log files
        """
        logger.info("Initializing CarbonSenseCrew...")

        # Set up LiteLLM configuration for WatsonX
        os.environ["LITELLM_MODEL"] = "watsonx/meta-llama/llama-2-70b-instruct"
        os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL", "")
        os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY", "")
        os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID", "")

        # Important: All carbon footprint metrics must include the following required fields:
        # - value: numeric value of the carbon footprint
        # - emission_unit: unit of emission (kg CO2e or g CO2e)
        # - product_unit: unit of product (per kg, per item, etc.)
        # - source: source of the data (milvus, discovery, serper, etc.)
        # - confidence: confidence score (0-1)
        # - product_name: name of the product (beef, coffee, etc.)

        self.config = config
        self.use_cache = use_cache
        self.use_hierarchical = use_hierarchical
        self.current_query = ""
        self.debug_mode = debug_mode
        self.store_thoughts = store_thoughts
        
        # Initialize services
        self.milvus = MilvusService(config)
        
        # Always use LiteLLM-based WatsonX service for optimal CrewAI integration
        logger.info("Using LiteLLM-based WatsonX service (optimized for CrewAI)")
        self.watsonx = LiteLLMWatsonxService(config)
        self.discovery = DiscoveryService(config)
        
        # Initialize AI parameters
        self._load_config_files()
        
        # Get WatsonX credentials for LLM initialization
        watsonx_config = {
            "api_base": os.getenv("WATSONX_URL"),
            "api_key": os.getenv("WATSONX_APIKEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
            "version": os.getenv("WATSONX_VERSION", "2023-05-29")
        }
        
        # Initialize LLM configurations for each agent with reduced verbosity
        self.agent_llms = {}
        for agent_name, agent_config in self.ai_params['agents'].items():
            self.agent_llms[agent_name] = LLM(
                model=agent_config['model'],
                api_base=watsonx_config["api_base"],
                api_key=watsonx_config["api_key"],
                project_id=watsonx_config["project_id"],
                temperature=agent_config.get('temperature', 0.7),
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=True,
                verbose=debug_mode  # Only verbose in debug mode
            )
        
        # Initialize the cache manager if caching is enabled
        self.cache_manager = CacheService() if use_cache else None
        
        if not debug_mode:
            logger.setLevel(logging.INFO)
        
        logger.info("✅ CarbonSenseCrew initialized successfully")

    def _load_config_files(self):
        """Load agent and task configurations from YAML files."""
        try:
            # Define paths to config files
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "config")
            self.agents_config_path = os.path.join(config_dir, "agents.yaml")
            self.tasks_config_path = os.path.join(config_dir, "tasks.yaml")
            self.ai_params_path = os.path.join(config_dir, "ai_parameters.yaml")
            
            # Load agents config
            with open(self.agents_config_path, "r") as f:
                self.agents_config = yaml.safe_load(f)
                logger.info(f"Loaded agents configuration from {f.name}")
                
            # Load tasks config
            with open(self.tasks_config_path, "r") as f:
                self.tasks_config = yaml.safe_load(f)
                logger.info(f"Loaded tasks configuration from {f.name}")
                
            # Load AI parameters config
            with open(self.ai_params_path, "r") as f:
                self.ai_params = yaml.safe_load(f)
                logger.info(f"Loaded AI parameters from {f.name}")
                
            logger.info("Successfully loaded all configuration files")
        except Exception as e:
            logger.error(f"Error loading config files: {str(e)}")
            raise RuntimeError(f"Failed to load config files: {str(e)}")
    
    def validate_carbon_metric(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a carbon metric against the schema and ensure product_name is present.
        
        Args:
            metric: Dictionary containing carbon footprint metric data
            
        Returns:
            The validated metric
            
        Raises:
            ValueError: If the metric doesn't match the schema or is missing required fields
        """
        try:
            # Verify product_name is present, which is now required
            if 'product_name' not in metric:
                # Try to populate from parse_and_normalise_task output if available
                parse_output = self.get_task_output('parse_and_normalise_task')
                if parse_output and isinstance(parse_output, dict) and 'product' in parse_output:
                    metric['product_name'] = parse_output['product']
                else:
                    raise ValueError("Carbon metric is missing required 'product_name' field")
            
            # Verify category is present, which is now required
            if 'category' not in metric:
                # Try to determine category based on product name
                product = metric['product_name'].lower()
                
                # Food-related keywords
                food_keywords = ['food', 'fruit', 'vegetable', 'meat', 'dairy', 'milk', 'cheese',
                                'egg', 'bread', 'grain', 'rice', 'pasta', 'bean', 'fish', 'seafood',
                                'beef', 'pork', 'chicken', 'apple', 'banana', 'coffee', 'tea', 'water',
                                'beer', 'wine', 'juice', 'soda', 'drink', 'beverage']
                
                # Energy-related keywords
                energy_keywords = ['energy', 'electricity', 'gas', 'fuel', 'power', 'heating', 'cooling',
                                  'appliance', 'device', 'lamp', 'light', 'bulb', 'ac', 'air conditioning',
                                  'heater', 'furnace', 'stove', 'oven', 'refrigerator', 'fridge', 'tv',
                                  'television', 'computer', 'laptop', 'charger', 'battery']
                
                # Transport-related keywords
                transport_keywords = ['car', 'vehicle', 'bike', 'bicycle', 'motorcycle', 'bus', 'train',
                                     'plane', 'flight', 'ship', 'boat', 'truck', 'transport', 'travel',
                                     'commute', 'drive', 'ride', 'fly', 'transit', 'trip', 'journey',
                                     'kilometer', 'mile', 'distance', 'fuel', 'gasoline', 'diesel']
                
                # Determine category based on keywords
                if any(keyword in product for keyword in food_keywords):
                    metric['category'] = "Food and Beverages"
                elif any(keyword in product for keyword in energy_keywords):
                    metric['category'] = "Household Energy Use"
                elif any(keyword in product for keyword in transport_keywords):
                    metric['category'] = "Transport Related"
                else:
                    # Default to Food if we can't determine (most queries are about food)
                    metric['category'] = "Food and Beverages"
                    logger.info(f"Defaulting category to 'Food and Beverages' for product: {product}")
                
            # Validate against schema
            validate(metric, carbon_metric)
            return metric
        except Exception as e:
            logger.error(f"Carbon metric validation error: {str(e)}")
            raise ValueError(f"Invalid carbon metric: {str(e)}")
    
    def get_task_output(self, task_name: str) -> Any:
        """Get the output of a task by name from task output files.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task output or None if not found
        """
        try:
            if task_name not in self.tasks_config:
                return None
                
            output_file = self.tasks_config[task_name].get('output_file')
            if not output_file:
                return None
                
            # Handle relative paths
            if not os.path.isabs(output_file):
                output_file = os.path.join(os.getcwd(), output_file)
                
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    try:
                        # Try to parse as JSON first
                        import json
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Return as plain text if not JSON
                        return content
            return None
        except Exception as e:
            logger.warning(f"Error getting task output: {str(e)}")
            return None
    
    @agent
    def query_processor(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['query_processor'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['query_processor'],
            max_retry_limit=1
        )
    
    @agent
    def milvus_researcher(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['milvus_researcher'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['milvus_researcher'],
            tools=[tools['carbon_research']],
            max_retry_limit=1
        )
        
    @agent
    def discovery_researcher(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['discovery_researcher'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['discovery_researcher'],
            tools=[tools['web_search']],
            max_retry_limit=1
        )
        
    @agent
    def serper_researcher(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['serper_researcher'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['serper_researcher'],
            tools=[SerperDevTool(country="us", num_results=5)],
            max_retry_limit=1
        )
        
    @agent
    def answer_consolidator(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['answer_consolidator'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['answer_consolidator'],
            max_retry_limit=1
        )
        
    @agent
    def answer_formatter(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['answer_formatter'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['answer_formatter'],
            max_retry_limit=1
        )

    @agent
    def unit_normalizer(self) -> Agent:
        return Agent(config=self.agents_config['unit_normalizer'],
                     llm=self.agent_llms['unit_normalizer'],
                     max_retry_limit=1)

    @agent
    def footprint_cache(self) -> Agent:
        return Agent(config=self.agents_config['footprint_cache'],
                     llm=self.agent_llms['footprint_cache'],
                     max_retry_limit=1)

    @agent
    def unit_harmoniser(self) -> Agent:
        return Agent(config=self.agents_config['unit_harmoniser'],
                     llm=self.agent_llms['unit_harmoniser'],
                     max_retry_limit=1)

    @agent
    def metric_ranker(self) -> Agent:
        return Agent(config=self.agents_config['metric_ranker'],
                     llm=self.agent_llms['metric_ranker'],
                     max_retry_limit=1)

    @agent
    def usage_logger(self) -> Agent:
        return Agent(config=self.agents_config['usage_logger'],
                     llm=self.agent_llms['usage_logger'],
                     max_retry_limit=1)
        
    @task
    def parse_and_normalise_task(self) -> Task:
        return Task(config=self.tasks_config['parse_and_normalise_task'])

    @task
    def cache_lookup_task(self) -> Task:
        return Task(
            config=self.tasks_config['cache_lookup_task'],
            context=[self.parse_and_normalise_task()]
        )
    
    @task
    def milvus_research_task(self) -> Task:
        """Define the research task."""
        return Task(
            config=self.tasks_config['milvus_research_task'],
            context=[self.parse_and_normalise_task()],
            max_retry_limit=1
        )
    
    @task
    def discovery_research_task(self) -> Task:
        """Define the compilation task."""
        return Task(
            config=self.tasks_config['discovery_research_task'],
            context=[self.parse_and_normalise_task()],
            max_retry_limit=1
        )
    
    @task
    def serper_research_task(self) -> Task:
        """Define the research task."""
        return Task(
            config=self.tasks_config['serper_research_task'],
            context=[self.parse_and_normalise_task()],
            max_retry_limit=1
        )
    
    @task
    def harmonise_task(self) -> Task:
        return Task(
            config=self.tasks_config['harmonise_task'],
            context=[self.milvus_research_task(),
                     self.discovery_research_task(),
                     self.serper_research_task()]
        )

    @task
    def rank_task(self) -> Task:
        return Task(
            config=self.tasks_config['rank_task'],
            context=[self.harmonise_task()]
        )

    # @task
    # def usage_logging_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['usage_logging_task'],
    #         context=[self.answer_formatting_task()]
    #     )
    
    @task
    def answer_formatting_task(self) -> Task:
        """Define the research task."""
        return Task(
            config=self.tasks_config['answer_formatting_task'],
            context=[self.rank_task(), self.parse_and_normalise_task()],
            max_retry_limit=1
        )
    
    @before_kickoff
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs before the crew starts."""
        logger.info(f"Preparing inputs for query: {inputs.get('query', '')}")
        inputs["skip_research"] = False
        return inputs
    
    @after_kickoff
    def process_output(self, output):
        """Process output after the crew finishes."""
        logger.info("Processing crew output")
        
        # Check if the output is a carbon metric JSON that needs validation
        if isinstance(output, dict) and any(k in output for k in ['value', 'emission_unit', 'product_unit']):
            try:
                # Validate and potentially add product_name if missing
                output = self.validate_carbon_metric(output)
                logger.info("Validated carbon metric output")
            except ValueError as e:
                logger.warning(f"Output validation warning: {str(e)}")
                
        # For research task outputs (usually arrays of metrics)
        elif isinstance(output, list) and len(output) > 0:
            if all(isinstance(item, dict) for item in output):
                validated_items = []
                for item in output:
                    if any(k in item for k in ['value', 'emission_unit', 'product_unit']):
                        try:
                            validated_items.append(self.validate_carbon_metric(item))
                        except ValueError as e:
                            logger.warning(f"Item validation warning: {str(e)}")
                            validated_items.append(item)  # Include original item
                    else:
                        validated_items.append(item)
                output = validated_items
                
        return output
    
    @crew
    def crew(self) -> Crew:
        """Define the CarbonSense crew."""
        # Choose process type
        process_type = Process.hierarchical if self.use_hierarchical else Process.sequential
        
        # runtime pruning: if cache hit skip research tasks
        if self.cache_manager and self.cache_manager.get(self.current_query):
            # Skip research tasks if cache hit found
            logger.info("Cache hit found, skipping research tasks")
            self.tasks.remove(self.milvus_research_task())
            self.tasks.remove(self.discovery_research_task())
            self.tasks.remove(self.serper_research_task())
            self.tasks.remove(self.harmonise_task())
            self.tasks.remove(self.rank_task())
        else:
            # Check if cache lookup was performed and returned MISS
            cache_result = self.get_task_output('cache_lookup_task')
            if cache_result and isinstance(cache_result, dict) and cache_result.get('status') == 'MISS':
                logger.info("Cache miss detected, proceeding with full research workflow")
            else:
                logger.info("No cache lookup results found, proceeding with full research workflow")
        
        # Create logs directory if it doesn't exist
        if self.store_thoughts:
            os.makedirs("logs/thoughts", exist_ok=True)
        
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=process_type,
            verbose=self.debug_mode,  # Only verbose in debug mode
            # memory=True,         # Enable crew memory
            # cache=self.use_cache,
            manager_llm=self.agent_llms['manager'],  # Use manager LLM from ai_parameters.yaml
            output_log_file=self.store_thoughts,  # Save output logs if store_thoughts is enabled
            output_dir="logs/thoughts" if self.store_thoughts else None,  # Save logs in the thoughts subdirectory
            full_output=self.store_thoughts,  # Include agent thoughts and reasoning in the logs
        )

class CrewAgentManager:
    """Manager for creating and running CrewAI agents for carbon footprint analysis."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, use_cache: bool = True,
                use_hierarchical: bool = True, use_simple_logger: bool = True, 
                store_thoughts: bool = False):
        """Initialize the crew agent manager.
        
        Args:
            config: Configuration manager instance
            debug_mode: Whether to enable debug mode
            use_cache: Whether to use caching
            use_hierarchical: Whether to use hierarchical task execution
            use_simple_logger: Whether to use simple logger (legacy parameter)
            store_thoughts: Whether to store agent thoughts and reasoning in log files
        """
        logger.info("Initializing CrewAgentManager...")
        self.config = config
        self.debug_mode = debug_mode
        self.use_cache = use_cache
        self.use_hierarchical = use_hierarchical
        self.use_simple_logger = use_simple_logger
        self.store_thoughts = store_thoughts
        
        # Initialize CarbonSenseCrew with LiteLLM always enabled
        self.carbon_crew = CarbonSenseCrew(
            config=config,
            debug_mode=debug_mode,
            use_cache=use_cache,
            use_hierarchical=use_hierarchical,
            store_thoughts=store_thoughts
        )
        
        # Track current session ID
        self.current_session_id = "current_session"
        
        # Initialize shared tools using the tool initializer
        self.shared_tools = initialize_tools(config)
        
        logger.info("✅ CrewAgentManager initialized successfully with LiteLLM integration")

    def process_query(self, query: str, show_context: bool = False) -> Dict[str, Any]:
        """Process a query using the crew agents."""
        logger.info(f"Processing query: {query}")
        
        try:
            # Check for cached response if caching is enabled
            if self.use_cache and self.carbon_crew.cache_manager:
                cached_result = self.carbon_crew.cache_manager.get(query)
                if cached_result:
                    logger.info("Using cached response")
                    return cached_result
            
            # Set the current query in the crew
            self.carbon_crew.current_query = query
            
            # Run the crew with the query and task dependencies
            crew_result = self.carbon_crew.crew().kickoff(
                inputs={
                    "query": query,
                    "show_context": show_context
                }
            )
            
            # Extract the string result from CrewOutput object
            if hasattr(crew_result, 'raw_output'):
                # New versions of CrewAI return a CrewOutput object with raw_output attribute
                result = str(crew_result.raw_output)
            elif hasattr(crew_result, 'output'):
                # Some versions might have an output attribute
                result = str(crew_result.output)
            else:
                # Fallback to string conversion
                result = str(crew_result)
            
            # Format the response
            response_data = {
                "response": result,
                "context": {}
            }
            
            # Add sources if available and requested
            if show_context:
                sources = []
                # Look for SOURCES section in formatted output with standard marker
                if isinstance(result, str):
                    # Try the standard marker format first
                    if "SOURCES:" in result:
                        parts = result.split("SOURCES:")
                        if len(parts) > 1:
                            sources_text = parts[1].strip()
                            # Extract bullet points or numbered items
                            source_items = []
                            for line in sources_text.split("\n"):
                                line = line.strip()
                                # Match bullet points, numbered items, or other common formats
                                if line.startswith('-') or line.startswith('•') or (line and line[0].isdigit() and line[1:3] in ['. ', ') ']):
                                    source_items.append(line.lstrip('- •0123456789.) ').strip())
                            sources = [s for s in source_items if s]
                    
                    # If no sources found with standard marker, try to get from the metric
                    if not sources:
                        # Try to get the source from the rank_task output
                        rank_output = self.carbon_crew.get_task_output('rank_task')
                        if rank_output and isinstance(rank_output, dict) and 'source' in rank_output:
                            sources.append(rank_output['source'])
                
                response_data["context"] = {
                    "sources": sources if sources else ["No specific sources identified."],
                    "show_context": show_context
                }
            
            # Cache the response if caching is enabled
            if self.use_cache and self.carbon_crew.cache_manager:
                self.carbon_crew.cache_manager.set(query, response_data)
            
            logger.info("Query processed successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to process query: {str(e)}",
                "response": "Sorry, I encountered an error while processing your query."
            }

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.carbon_crew.cache_manager:
            self.carbon_crew.cache_manager.clear()
            logger.info("Query cache cleared")
        else:
            logger.warning("Caching is not enabled")