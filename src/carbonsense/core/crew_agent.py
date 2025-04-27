import logging
import os
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool
from langchain.callbacks.manager import CallbackManager
from langchain.llms.base import LLM as LangChainLLM
from jsonschema import validate
from pydantic import BaseModel, Field, ValidationError

from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..services.litellm_watsonx_service import LiteLLMWatsonxService
from ..services.discovery_service import DiscoveryService
from ..utils.logger import setup_logger
from ..services.cache_service import CacheService
from ..config.common_schema import carbon_metric

# Import tools
from .tools import CarbonResearchTool, WebSearchTool, initialize_tools

# Set up logger with colored output
logger = setup_logger(__name__)

# Pydantic model for final response validation
class CarbonResponse(BaseModel):
    """Pydantic model for validating and standardizing carbon response data."""
    answer: str
    method: str = Field(default="Based on environmental data analysis.")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    category: str = Field(default="Miscellaneous")
    sources: List[str] = Field(default_factory=list)

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
        # Commenting out Discovery service initialization
        # self.discovery = DiscoveryService(config)
        
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
                # Try to populate from entity extraction output if available
                entity_output = self.get_task_output('entity_extraction_task')
                if entity_output and isinstance(entity_output, dict) and 'products' in entity_output and entity_output['products']:
                    metric['product_name'] = entity_output['products'][0]['name']
                else:
                    raise ValueError("Carbon metric is missing required 'product_name' field")
            
            # Verify category is present, which is now required
            if 'category' not in metric:
                # Try to determine category based on query classification
                classification_output = self.get_task_output('query_classification_task')
                if classification_output and isinstance(classification_output, dict) and 'category' in classification_output:
                    metric['category'] = classification_output['category']
                else:
                    # Determine category based on product name
                    product = metric['product_name'].lower()
                    
                    # Food-related keywords
                    food_keywords = ['food', 'fruit', 'vegetable', 'meat', 'dairy', 'milk', 'cheese',
                                    'egg', 'bread', 'grain', 'rice', 'pasta', 'bean', 'fish', 'seafood',
                                    'beef', 'pork', 'chicken', 'apple', 'banana', 'coffee', 'tea', 'water',
                                    'beer', 'wine', 'juice', 'soda', 'drink', 'beverage', 'meal', 'diet']
                    
                    # Energy-related keywords
                    energy_keywords = ['energy', 'electricity', 'gas', 'fuel', 'power', 'heating', 'cooling',
                                      'appliance', 'device', 'lamp', 'light', 'bulb', 'ac', 'air conditioning',
                                      'heater', 'furnace', 'stove', 'oven', 'refrigerator', 'fridge', 'tv',
                                      'television', 'computer', 'laptop', 'charger', 'battery', 'kWh']
                    
                    # Mobility-related keywords
                    mobility_keywords = ['car', 'vehicle', 'bike', 'bicycle', 'motorcycle', 'bus', 'train',
                                         'plane', 'flight', 'ship', 'boat', 'truck', 'transport', 'travel',
                                         'commute', 'drive', 'ride', 'fly', 'transit', 'trip', 'journey',
                                         'kilometer', 'mile', 'distance', 'gasoline', 'diesel', 'EV', 'uber']
                    
                    # Purchase-related keywords
                    purchase_keywords = ['buy', 'purchase', 'shopping', 'amazon', 'item', 'product', 'good',
                                         'clothing', 'fashion', 'electronic', 'device', 'gadget', 'phone',
                                         'iphone', 'android', 'laptop', 'bag', 'shoe', 'furniture', 'toy',
                                         'book', 'online', 'delivery', 'packaging', 'streaming', 'subscription']
                    
                    # Determine category based on keywords
                    if any(keyword in product for keyword in food_keywords):
                        metric['category'] = "Food & Diet"
                    elif any(keyword in product for keyword in energy_keywords):
                        metric['category'] = "Energy Use"
                    elif any(keyword in product for keyword in mobility_keywords):
                        metric['category'] = "Mobility"
                    elif any(keyword in product for keyword in purchase_keywords):
                        metric['category'] = "Purchases"
                    else:
                        # Default to Miscellaneous if we can't determine
                        metric['category'] = "Miscellaneous"
                        logger.info(f"Defaulting category to 'Miscellaneous' for product: {product}")
                
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
    def query_classifier(self) -> Agent:
        """Query Intent and Category Classifier agent."""
        return Agent(
            config=self.agents_config['query_classifier'],
            verbose=self.debug_mode,
            llm=self.agent_llms['query_classifier'],
            max_retry_limit=1
        )
    
    @agent
    def entity_extractor(self) -> Agent:
        """Entity Extractor agent."""
        return Agent(
            config=self.agents_config['entity_extractor'],
            verbose=self.debug_mode,
            llm=self.agent_llms['entity_extractor'],
            max_retry_limit=1
        )
        
    @agent
    def unit_normalizer(self) -> Agent:
        """Unit Normalizer agent."""
        return Agent(
            config=self.agents_config['unit_normalizer'],
            verbose=self.debug_mode,
            llm=self.agent_llms['unit_normalizer'],
            max_retry_limit=1
        )
    
    @agent
    def footprint_cache(self) -> Agent:
        """Cache agent for carbon footprint data."""
        return Agent(
            config=self.agents_config['footprint_cache'],
            verbose=self.debug_mode,
            llm=self.agent_llms['footprint_cache'],
            max_retry_limit=1
        )
        
    @agent
    def milvus_researcher(self) -> Agent:
        """Milvus Research agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['milvus_researcher'],
            verbose=self.debug_mode,
            llm=self.agent_llms['milvus_researcher'],
            tools=[tools['carbon_research']],
            max_retry_limit=1
        )
        
    @agent
    def discovery_researcher(self) -> Agent:
        """Discovery Research agent."""
        # Instead of returning None, return a dummy agent that won't be used
        tools = initialize_tools(self.config)
        return Agent(
            config={"role": "Discovery Researcher (Disabled)", "goal": "Disabled", "backstory": "This agent is disabled"},
            verbose=False,
            llm=self.agent_llms['serper_researcher'],  # Reuse another LLM
            max_retry_limit=1
        )
        
    @agent
    def serper_researcher(self) -> Agent:
        """Web Search Research agent."""
        return Agent(
            config=self.agents_config['serper_researcher'],
            verbose=self.debug_mode,
            llm=self.agent_llms['serper_researcher'],
            tools=[SerperDevTool(country="us", num_results=5)],
            max_retry_limit=1
        )
    
    @agent
    def unit_harmoniser(self) -> Agent:
        """Unit Harmonization agent."""
        return Agent(
            config=self.agents_config['unit_harmoniser'],
            verbose=self.debug_mode,
            llm=self.agent_llms['unit_harmoniser'],
            max_retry_limit=1
        )
    
    @agent
    def carbon_estimator(self) -> Agent:
        """Carbon Estimation agent."""
        return Agent(
            config=self.agents_config['carbon_estimator'],
            verbose=self.debug_mode,
            llm=self.agent_llms['carbon_estimator'],
            max_retry_limit=1
        )
        
    @agent
    def metric_ranker(self) -> Agent:
        """Metric Ranking agent."""
        return Agent(
            config=self.agents_config['metric_ranker'],
            verbose=self.debug_mode,
            llm=self.agent_llms['metric_ranker'],
            max_retry_limit=1
        )
    
    @agent
    def comparison_formatter(self) -> Agent:
        """Comparison Formatting agent."""
        return Agent(
            config=self.agents_config['comparison_formatter'],
            verbose=self.debug_mode,
            llm=self.agent_llms['comparison_formatter'],
            max_retry_limit=1
        )
    
    @agent
    def recommendation_agent(self) -> Agent:
        """Recommendation agent."""
        return Agent(
            config=self.agents_config['recommendation_agent'],
            verbose=self.debug_mode,
            llm=self.agent_llms['recommendation_agent'],
            max_retry_limit=1
        )
    
    @agent
    def explanation_agent(self) -> Agent:
        """Explanation agent."""
        return Agent(
            config=self.agents_config['explanation_agent'],
            verbose=self.debug_mode,
            llm=self.agent_llms['explanation_agent'],
            max_retry_limit=1
        )
        
    @agent
    def answer_formatter(self) -> Agent:
        """Answer Formatting agent."""
        return Agent(
            config=self.agents_config['answer_formatter'],
            verbose=self.debug_mode,
            llm=self.agent_llms['answer_formatter'],
            max_retry_limit=1
        )
    
    @agent
    def answer_consolidator(self) -> Agent:
        """Answer Consolidation agent."""
        return Agent(
            config=self.agents_config['answer_consolidator'],
            verbose=self.debug_mode,
            llm=self.agent_llms['answer_consolidator'],
            max_retry_limit=1
        )
    
    @agent
    def manager(self) -> Agent:
        """Team Manager agent."""
        return Agent(
            config=self.agents_config['manager'],
            verbose=self.debug_mode,
            llm=self.agent_llms['manager'],
            allow_delegation=True,
            max_retry_limit=1
        )
    
    @task
    def query_classification_task(self) -> Task:
        """Query classification task."""
        return Task(
            config=self.tasks_config['query_classification_task'],
            max_retry_limit=1
        )
    
    @task
    def entity_extraction_task(self) -> Task:
        """Entity extraction task."""
        return Task(
            config=self.tasks_config['entity_extraction_task'],
            context=[self.query_classification_task()],
            max_retry_limit=1
        )
    
    @task
    def unit_normalization_task(self) -> Task:
        """Unit normalization task."""
        return Task(
            config=self.tasks_config['unit_normalization_task'],
            context=[self.entity_extraction_task(), self.query_classification_task()],
            max_retry_limit=1
        )
    
    @task
    def cache_lookup_task(self) -> Task:
        """Cache lookup task."""
        return Task(
            config=self.tasks_config['cache_lookup_task'],
            context=[self.unit_normalization_task()],
            max_retry_limit=1
        )
    
    @task
    def milvus_research_task(self) -> Task:
        """Milvus research task."""
        return Task(
            config=self.tasks_config['milvus_research_task'],
            context=[self.unit_normalization_task(), self.cache_lookup_task(), self.query_classification_task()],
            max_retry_limit=1
        )
    
    @task
    def discovery_research_task(self) -> Task:
        """Discovery research task."""
        # Instead of returning None, return a dummy task that won't be used
        return Task(
            description="This task is disabled",
            expected_output="Disabled",
            agent=self.discovery_researcher(),
            async_execution=False,
            output_file=None
        )
    
    @task
    def serper_research_task(self) -> Task:
        """Serper research task."""
        return Task(
            config=self.tasks_config['serper_research_task'],
            context=[self.unit_normalization_task(), self.cache_lookup_task(), self.query_classification_task()],
            max_retry_limit=1
        )
    
    @task
    def harmonise_task(self) -> Task:
        """Harmonization task."""
        # Remove discovery_research_task completely from the context list
        return Task(
            config=self.tasks_config['harmonise_task'],
            context=[
                self.milvus_research_task(),
                self.serper_research_task(),
                self.unit_normalization_task(),
                self.query_classification_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def carbon_estimation_task(self) -> Task:
        """Carbon estimation task."""
        return Task(
            config=self.tasks_config['carbon_estimation_task'],
            context=[
                self.harmonise_task(),
                self.unit_normalization_task(),
                self.query_classification_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def rank_metrics_task(self) -> Task:
        """Metric ranking task."""
        return Task(
            config=self.tasks_config['rank_metrics_task'],
            context=[
                self.carbon_estimation_task(),
                self.query_classification_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def comparison_task(self) -> Task:
        """Comparison task."""
        return Task(
            config=self.tasks_config['comparison_task'],
            context=[
                self.rank_metrics_task(),
                self.query_classification_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def recommendation_task(self) -> Task:
        """Recommendation task."""
        return Task(
            config=self.tasks_config['recommendation_task'],
            context=[
                self.rank_metrics_task(),
                self.query_classification_task(),
                self.carbon_estimation_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def explanation_task(self) -> Task:
        """Explanation task."""
        return Task(
            config=self.tasks_config['explanation_task'],
            context=[
                self.rank_metrics_task(),
                self.query_classification_task(),
                self.carbon_estimation_task()
            ],
            max_retry_limit=1
        )
    
    @task
    def answer_formatting_task(self) -> Task:
        """Answer formatting task."""
        return Task(
            config=self.tasks_config['answer_formatting_task'],
            context=[
                self.query_classification_task(),
                self.rank_metrics_task(),
                self.comparison_task(),
                self.recommendation_task(),
                self.explanation_task(),
                self.carbon_estimation_task()
            ],
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
        
        print("="*100)
        print("Output:")
        print(type(output))
        print(output)
        print("="*100)

        # Handle CrewOutput object (from newer versions of CrewAI)
        if hasattr(output, '__class__') and output.__class__.__name__ == 'CrewOutput':
            logger.info("Processing CrewOutput object")
            
            # Try to access properties in order of preference
            if hasattr(output, 'json_dict') and output.json_dict:
                logger.info("Using json_dict from CrewOutput")
                output = output.json_dict
            elif hasattr(output, 'pydantic') and output.pydantic:
                logger.info("Using pydantic model from CrewOutput")
                output = output.pydantic.dict()
            elif hasattr(output, 'raw'):
                raw_output = output.raw
                logger.info(f"Using raw output from CrewOutput: {type(raw_output)}")
                # Continue processing with the raw output
                output = raw_output
            else:
                # Convert to string and continue with processing
                output = str(output)
                logger.info(f"Converted CrewOutput to string: {output[:100]}...")
        
        # Initialize standard response structure
        response_dict = {
            "answer": "",
            "method": "Based on environmental data analysis.",
            "confidence": 0.7,
            "category": "Miscellaneous",
            "sources": []
        }
        
        # Handle JSON code blocks in string output
        if isinstance(output, str) and "```json" in output:
            try:
                import re
                import json
                match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    extracted = json.loads(json_str)
                    if isinstance(extracted, dict):
                        output = extracted
                        logger.info("Extracted JSON from code block")
            except Exception as e:
                logger.warning(f"JSON extraction failed: {str(e)}")
        
        # Case 1: Output is already a properly structured dict
        if isinstance(output, dict) and all(k in output for k in ["answer", "method", "confidence", "category"]):
            logger.info("Output already in correct format")
            
            # Try to parse nested JSON in 'answer'
            if isinstance(output["answer"], str) and output["answer"].startswith("{") and output["answer"].endswith("}"):
                try:
                    import json
                    nested = json.loads(output["answer"])
                    if isinstance(nested, dict) and all(k in nested for k in ["answer", "method", "confidence", "category"]):
                        logger.info("Extracted nested JSON from 'answer' field")
                        try:
                            # Validate with Pydantic
                            validated = CarbonResponse(**nested).dict()
                            return validated
                        except ValidationError as e:
                            logger.warning(f"Nested JSON validation error: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to parse nested JSON in answer: {str(e)}")
                    
                    # Try to parse Python-style dict with single quotes
                    try:
                        if "'" in output["answer"]:
                            import json
                            python_style = output["answer"].replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
                            nested = json.loads(python_style)
                            if isinstance(nested, dict) and all(k in nested for k in ["answer", "method", "confidence", "category"]):
                                logger.info("Extracted Python-style nested JSON from 'answer' field")
                                try:
                                    # Validate with Pydantic
                                    validated = CarbonResponse(**nested).dict()
                                    return validated
                                except ValidationError as e:
                                    logger.warning(f"Python-style nested JSON validation error: {str(e)}")
                    except Exception as e2:
                        logger.warning(f"Failed to parse Python-style nested JSON in answer: {str(e2)}")
            
            # Validate with Pydantic
            try:
                validated = CarbonResponse(**output).dict()
                return validated
            except ValidationError as e:
                logger.warning(f"Output validation error: {str(e)}")
                response_dict.update({k: v for k, v in output.items() if k in response_dict})
                return response_dict
            
        # Case 2: Output is a string
        if isinstance(output, str):
            # Try parsing as JSON if it looks like a JSON string
            if output.strip().startswith('{') and output.strip().endswith('}'):
                try:
                    import json
                    parsed = json.loads(output)
                    if isinstance(parsed, dict):
                        # Check if it has the required fields
                        if all(k in parsed for k in ["answer", "method", "confidence", "category"]):
                            logger.info("Successfully parsed JSON string")
                            
                            # Check for nested JSON in answer
                            if isinstance(parsed["answer"], str) and parsed["answer"].startswith("{") and parsed["answer"].endswith("}"):
                                try:
                                    nested = json.loads(parsed["answer"])
                                    if isinstance(nested, dict) and all(k in nested for k in ["answer", "method", "confidence", "category"]):
                                        logger.info("Extracted nested JSON from parsed answer field")
                                        try:
                                            # Validate with Pydantic
                                            validated = CarbonResponse(**nested).dict()
                                            return validated
                                        except ValidationError as e:
                                            logger.warning(f"Nested parsed JSON validation error: {str(e)}")
                                except Exception as e:
                                    logger.warning(f"Failed to parse nested JSON in parsed answer: {str(e)}")
                            
                            try:
                                # Validate with Pydantic
                                validated = CarbonResponse(**parsed).dict()
                                return validated
                            except ValidationError as e:
                                logger.warning(f"Parsed JSON validation error: {str(e)}")
                                response_dict.update({k: v for k, v in parsed.items() if k in response_dict})
                                return response_dict
                        else:
                            # Copy any matching fields
                            for key in parsed:
                                if key in response_dict:
                                    response_dict[key] = parsed[key]
                            
                            # If no answer, use the whole parsed object
                            if not response_dict["answer"]:
                                response_dict["answer"] = str(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON string: {str(e)}")
                    
                    # Try parsing as Python-style dict
                    try:
                        import json
                        python_style = output.strip().replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
                        parsed = json.loads(python_style)
                        if isinstance(parsed, dict):
                            if all(k in parsed for k in ["answer", "method", "confidence", "category"]):
                                logger.info("Successfully parsed Python-style dict")
                                try:
                                    # Validate with Pydantic
                                    validated = CarbonResponse(**parsed).dict()
                                    return validated
                                except ValidationError as e:
                                    logger.warning(f"Python-style dict validation error: {str(e)}")
                                    response_dict.update({k: v for k, v in parsed.items() if k in response_dict})
                                    return response_dict
                            else:
                                for key in parsed:
                                    if key in response_dict:
                                        response_dict[key] = parsed[key]
                                if not response_dict["answer"]:
                                    response_dict["answer"] = str(parsed)
                    except Exception as e2:
                        logger.warning(f"Failed to parse Python-style dict: {str(e2)}")
            
            response_dict["answer"] = output
            
            # Return validated model
            try:
                return CarbonResponse(**response_dict).dict()
            except ValidationError as e:
                logger.warning(f"String content validation error: {str(e)}")
                return response_dict
            
        # Case 3: Output is a dict but not with all required fields
        if isinstance(output, dict):
            # Copy any matching fields
            for key in output:
                if key in response_dict:
                    response_dict[key] = output[key]
            
            # If there's no answer field, try to create one
            if not response_dict["answer"] and output:
                if "value" in output or "emission_unit" in output:
                    response_dict["answer"] = f"The carbon footprint is approximately {output.get('value', '?')} {output.get('emission_unit', 'CO2e')}."
                elif "emission" in output:
                    response_dict["answer"] = output["emission"]
                else:
                    response_dict["answer"] = str(output)
            
            # Return validated model
            try:
                return CarbonResponse(**response_dict).dict()
            except ValidationError as e:
                logger.warning(f"Partial dict validation error: {str(e)}")
                return response_dict
            
        # Case 4: Output is a list
        if isinstance(output, list) and len(output) > 0:
            # Just take the first item for simplicity
            if all(isinstance(item, dict) for item in output):
                first_item = output[0]
                for key in first_item:
                    if key in response_dict:
                        response_dict[key] = first_item[key]
                
                if not response_dict["answer"]:
                    response_dict["answer"] = f"Found {len(output)} relevant carbon footprint data points."
            else:
                response_dict["answer"] = str(output)
            
            # Return validated model
            try:
                return CarbonResponse(**response_dict).dict()
            except ValidationError as e:
                logger.warning(f"List content validation error: {str(e)}")
                return response_dict
            
        # Default: Convert output to string
        response_dict["answer"] = str(output)
        
        # Return validated model
        try:
            return CarbonResponse(**response_dict).dict()
        except ValidationError as e:
            logger.warning(f"Default validation error: {str(e)}")
            return response_dict
    
    @crew
    def crew(self) -> Crew:
        """Define the CarbonSense crew."""
        # Choose process type
        process_type = Process.hierarchical if self.use_hierarchical else Process.sequential
        
        # Skip the discovery_research_task regardless of cache
        try:
            self.tasks.remove(self.discovery_research_task())
            logger.info("Removed discovery_research_task as it's disabled")
        except ValueError:
            # Task might not be in the list, which is fine
            pass
        
        # runtime pruning: if cache hit skip research tasks
        if self.cache_manager and self.cache_manager.get(self.current_query):
            # Skip research tasks if cache hit found
            logger.info("Cache hit found, skipping research tasks")
            self.tasks.remove(self.milvus_research_task())
            self.tasks.remove(self.serper_research_task())
            self.tasks.remove(self.harmonise_task())
            self.tasks.remove(self.rank_metrics_task())
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
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, use_cache: bool = False,
                use_hierarchical: bool = False, use_simple_logger: bool = False, 
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
                    "query": query
                }
            )
            
            # Extract the result from CrewOutput object
            if hasattr(crew_result, '__class__') and crew_result.__class__.__name__ == 'CrewOutput':
                logger.info("Processing CrewOutput object in process_query")
                
                # Try accessing properties in order of preference
                if hasattr(crew_result, 'json_dict') and crew_result.json_dict:
                    logger.info("Using json_dict from CrewOutput")
                    result = crew_result.json_dict
                elif hasattr(crew_result, 'pydantic') and crew_result.pydantic:
                    logger.info("Using pydantic model from CrewOutput")
                    result = crew_result.pydantic.dict()
                elif hasattr(crew_result, 'raw'):
                    logger.info("Using raw from CrewOutput")
                    result = crew_result.raw
                else:
                    logger.info("Using string representation of CrewOutput")
                    result = str(crew_result)
            elif hasattr(crew_result, 'raw_output'):
                # New versions of CrewAI return a CrewOutput object with raw_output attribute
                result = crew_result.raw_output
                logger.info(f"Using raw_output from crew_result: {type(result)}")
            elif hasattr(crew_result, 'output'):
                # Some versions might have an output attribute
                result = crew_result.output
                logger.info(f"Using output from crew_result: {type(result)}")
            else:
                # Fallback to string conversion
                result = str(crew_result)
                logger.info(f"Using string conversion for crew_result: {type(result)}")
            
            # Process the output to ensure it's a properly structured dictionary
            # Start by checking if it's already in the expected format
            if isinstance(result, dict) and all(k in result for k in ["answer", "method", "confidence", "category"]):
                processed_result = result
                logger.info("Result already in correct format")
            # Handle string results
            elif isinstance(result, str):
                # Check for JSON code blocks
                if "```json" in result:
                    try:
                        import re
                        import json
                        match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and all(k in parsed for k in ["answer", "method", "confidence", "category"]):
                                processed_result = parsed
                                logger.info("Extracted JSON from code block in result string")
                    except Exception as e:
                        logger.warning(f"Error extracting JSON from code block: {str(e)}")
                        processed_result = {
                            "answer": result,
                            "method": "Based on environmental data analysis.",
                            "confidence": 0.7,
                            "category": "Miscellaneous"
                        }
                # Try to parse JSON string
                elif result.strip().startswith('{') and result.strip().endswith('}'):
                    try:
                        import json
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and all(k in parsed for k in ["answer", "method", "confidence", "category"]):
                            processed_result = parsed
                            logger.info("Successfully parsed JSON string")
                        else:
                            # JSON object but wrong format
                            processed_result = {
                                "answer": result,
                                "method": "Based on environmental data analysis.",
                                "confidence": 0.7,
                                "category": "Miscellaneous"
                            }
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON string: {str(e)}")
                        processed_result = {
                            "answer": result,
                            "method": "Based on environmental data analysis.",
                            "confidence": 0.7,
                            "category": "Miscellaneous"
                        }
                else:
                    # Not JSON, use the string as the answer
                    processed_result = {
                        "answer": result,
                        "method": "Based on environmental data analysis.",
                        "confidence": 0.7,
                        "category": "Miscellaneous"
                    }
                    logger.info("Using plain string as answer")
            else:
                # For any other type, convert to a structured response
                logger.info(f"Converting {type(result).__name__} to structured format")
                processed_result = {
                    "answer": str(result),
                    "method": "Based on environmental data analysis.",
                    "confidence": 0.7,
                    "category": "Miscellaneous"
                }
            
            # Create the final response structure
            response_data = {
                "response": processed_result
            }
            
            # Add context if requested
            if show_context:
                sources = []
                # Get sources from various task outputs
                rank_output = self.carbon_crew.get_task_output('rank_metrics_task')
                if rank_output and isinstance(rank_output, list) and len(rank_output) > 0:
                    for metric in rank_output:
                        if isinstance(metric, dict) and 'source' in metric:
                            sources.append(metric['source'])
                
                answer_output = self.carbon_crew.get_task_output('answer_formatting_task')
                if answer_output and isinstance(answer_output, dict) and 'sources' in answer_output:
                    answer_sources = answer_output['sources']
                    if isinstance(answer_sources, list):
                        for source in answer_sources:
                            if isinstance(source, dict) and 'title' in source:
                                sources.append(f"{source.get('title')} - {source.get('url', 'No URL')}")
                
                # Deduplicate sources
                sources = list(set(sources))
                
                response_data["context"] = {
                    "sources": sources if sources else ["No specific sources identified."]
                }
            
            # Cache the response if caching is enabled
            if self.use_cache and self.carbon_crew.cache_manager:
                self.carbon_crew.cache_manager.set(query, response_data)
            
            logger.info("Query processed successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "response": {
                    "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                    "method": "No calculation could be performed due to an error.",
                    "confidence": 0.0,
                    "category": "Miscellaneous"
                }
            }

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.carbon_crew.cache_manager:
            self.carbon_crew.cache_manager.clear()
            logger.info("Query cache cleared")
        else:
            logger.warning("Caching is not enabled")