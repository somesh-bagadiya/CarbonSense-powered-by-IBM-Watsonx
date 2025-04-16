import logging
import os
import yaml
from typing import Dict, List, Any, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from langchain.callbacks.manager import CallbackManager
from langchain.llms.base import LLM as LangChainLLM

from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..services.litellm_watsonx_service import LiteLLMWatsonxService
from ..services.discovery_service import DiscoveryService
from ..utils.logger import setup_logger
from ..utils.cache_manager import CacheManager

# Import tools
from .tools import CarbonResearchTool, WebSearchTool, initialize_tools

# Set up logger with colored output
logger = setup_logger(__name__)

@CrewBase
class CarbonSenseCrew:
    """Carbon footprint analysis crew using CrewAI's recommended approach."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, 
                 use_cache: bool = True, use_hierarchical: bool = True):
        """Initialize the CarbonSense crew."""
        logger.info("Initializing CarbonSenseCrew...")

        # Set up LiteLLM configuration for WatsonX
        os.environ["LITELLM_MODEL"] = "watsonx/meta-llama/llama-2-70b-instruct"
        os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL", "")
        os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY", "")
        os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID", "")
        
        # # Configure LiteLLM to use WatsonX
        # os.environ["OPENAI_API_BASE"] = os.getenv("WATSONX_URL", "")
        # os.environ["OPENAI_API_KEY"] = os.getenv("WATSONX_APIKEY", "")
        # os.environ["OPENAI_API_VERSION"] = os.getenv("WATSONX_VERSION", "2023-05-29")

        self.config = config
        self.use_cache = use_cache
        self.use_hierarchical = use_hierarchical
        self.current_query = ""
        self.debug_mode = debug_mode
        
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
        self.cache_manager = CacheManager() if use_cache else None
        
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
            
            # AI parameters are in a different location
            ai_params_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
            self.ai_params_path = os.path.join(ai_params_dir, "ai_parameters.yaml")
            
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
    
    @agent
    def researcher(self) -> Agent:
        """Carbon Data Researcher agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['researcher'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['researcher'],
            tools=[tools['carbon_research']],
            max_retry_limit=1
        )
    
    @agent
    def analyst(self) -> Agent:
        """Carbon Data Analyst agent."""
        return Agent(
            config=self.agents_config['analyst'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['analyst'],
            max_retry_limit=1
        )
    
    @agent
    def formatter(self) -> Agent:
        """Information Formatter agent."""
        return Agent(
            config=self.agents_config['formatter'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['formatter'],
            max_retry_limit=1
        )
    
    @agent
    def compiler(self) -> Agent:
        """Report Compiler agent."""
        tools = initialize_tools(self.config)
        return Agent(
            config=self.agents_config['compiler'],
            verbose=self.debug_mode,  # Only verbose in debug mode
            llm=self.agent_llms['compiler'],
            tools=[tools['web_search']],
            max_retry_limit=1
        )
    
    @task
    def research_task(self) -> Task:
        """Define the research task."""
        return Task(
            config=self.tasks_config['research_task'],
            max_retry_limit=1
        )
    
    @task
    def analysis_task(self) -> Task:
        """Define the analysis task."""
        return Task(
            config=self.tasks_config['analysis_task'],
            max_retry_limit=1
        )
    
    @task
    def formatting_task(self) -> Task:
        """Define the formatting task."""
        return Task(
            config=self.tasks_config['formatting_task'],
            max_retry_limit=1
        )
    
    @task
    def compilation_task(self) -> Task:
        """Define the compilation task."""
        return Task(
            config=self.tasks_config['compilation_task'],
            max_retry_limit=1
        )
    
    @before_kickoff
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs before the crew starts."""
        logger.info(f"Preparing inputs for query: {inputs.get('query', '')}")
        return inputs
    
    @after_kickoff
    def process_output(self, output):
        """Process output after the crew finishes."""
        logger.info("Processing crew output")
        
        return output
    
    @crew
    def crew(self) -> Crew:
        """Define the CarbonSense crew."""
        # Choose process type
        process_type = Process.hierarchical if self.use_hierarchical else Process.sequential
        
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=process_type,
            verbose=self.debug_mode,  # Only verbose in debug mode
            # memory=True,         # Enable crew memory
            # cache=self.use_cache,
            manager_llm=self.agent_llms['manager'],  # Use manager LLM from ai_parameters.yaml
            output_log_file=self.debug_mode,  # Only output logs in debug mode
        )

class CrewAgentManager:
    """Manager for creating and running CrewAI agents for carbon footprint analysis."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, use_cache: bool = True,
                use_hierarchical: bool = True, use_simple_logger: bool = True):
        """Initialize the crew agent manager."""
        logger.info("Initializing CrewAgentManager...")
        self.config = config
        self.debug_mode = debug_mode
        self.use_cache = use_cache
        self.use_hierarchical = use_hierarchical
        self.use_simple_logger = use_simple_logger
        
        # Initialize CarbonSenseCrew with LiteLLM always enabled
        self.carbon_crew = CarbonSenseCrew(
            config=config,
            debug_mode=debug_mode,
            use_cache=use_cache,
            use_hierarchical=use_hierarchical
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
            result = self.carbon_crew.crew().kickoff(
                inputs={
                    "query": query,
                    "show_context": show_context
                }
            )
            
            # Format the response
            response_data = {
                "response": result,
                "context": {}
            }
            
            # Add sources if available and requested
            if show_context:
                sources = []
                if isinstance(result, str) and "SOURCES:" in result:
                    parts = result.split("SOURCES:")
                    if len(parts) > 1:
                        sources_text = parts[1].strip()
                        sources = [s.strip() for s in sources_text.split("\n") if s.strip()]
                
                response_data["context"] = {
                    "sources": sources,
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