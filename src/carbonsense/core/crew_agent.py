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
from ..utils.agent_debugger import AgentDebugger
from ..utils.cache_manager import CacheManager
from ..utils.simple_agent_logger import get_simple_agent_logger

# Import tools
from .tools import CarbonResearchTool, WebSearchTool

# Set up logger with colored output
logger = setup_logger(__name__)

@CrewBase
class CarbonSenseCrew:
    """Carbon footprint analysis crew using CrewAI's recommended approach."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, 
                 use_cache: bool = True, use_hierarchical: bool = True,
                 use_simple_logger: bool = True):
        """Initialize the CarbonSense crew."""
        logger.info("Initializing CarbonSenseCrew...")
        
        # Set default LLM configuration for CrewAI
        os.environ["OPENAI_API_MODEL"] = "watsonx/meta-llama/llama-3-3-70b-instruct"
        os.environ["OPENAI_API_BASE"] = os.getenv("WATSONX_URL", "")
        os.environ["OPENAI_API_KEY"] = os.getenv("WATSONX_APIKEY", "")
        
        self.config = config
        self.debug_mode = debug_mode
        self.use_cache = use_cache
        self.use_hierarchical = use_hierarchical
        self.use_simple_logger = use_simple_logger
        self.current_query = ""
        
        # Initialize services
        self.milvus = MilvusService(config)
        
        # Always use LiteLLM-based WatsonX service for optimal CrewAI integration
        logger.info("Using LiteLLM-based WatsonX service (optimized for CrewAI)")
        self.watsonx = LiteLLMWatsonxService(config)
            
        self.discovery = DiscoveryService(config)
        
        # Initialize AI parameters
        self._load_config_files()
        
        # Get WatsonX credentials from environment variables
        watsonx_config = {
            "api_base": os.getenv("WATSONX_URL"),  # LLM expects api_base instead of base_url
            "api_key": os.getenv("WATSONX_APIKEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID")
        }
        
        # Initialize LLM configurations for each agent from ai_parameters.yaml
        self.agent_llms = {}
        for agent_name, agent_config in self.ai_params['agents'].items():
            # Add the env variables as kwargs since they are required by WatsonX
            self.agent_llms[agent_name] = LLM(
                model=agent_config['model'],
                api_base=watsonx_config["api_base"],  # Use api_base consistently
                api_key=watsonx_config["api_key"],
                temperature=agent_config.get('temperature', 0.7),
                presence_penalty=0.0,
                frequency_penalty=0.0,
                project_id=watsonx_config["project_id"],  # Pass project_id as a custom parameter
                stream=True  # Enable streaming for better interaction
            )
        
        # Initialize debugger if debug mode is enabled
        self.debugger = AgentDebugger() if debug_mode else None
        
        # Initialize the cache manager if caching is enabled
        self.cache_manager = CacheManager() if use_cache else None
        
        # Initialize the simple logger if enabled
        if use_simple_logger:
            self.simple_logger = get_simple_agent_logger()
            # Use a fixed session ID
            self.current_session_id = "current_session"
        else:
            # Initialize standard logging
            self.current_session_id = f"session_{id(self)}"
        
        logger.info(f"✅ CarbonSenseCrew initialized successfully with session ID: {self.current_session_id}")
    
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
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            llm=self.agent_llms['researcher'],
            tools=[CarbonResearchTool(self.config, self.watsonx)],
            max_retry_limit=1
        )
    
    @agent
    def analyst(self) -> Agent:
        """Carbon Data Analyst agent."""
        return Agent(
            config=self.agents_config['analyst'],
            # verbose=True,
            llm=self.agent_llms['analyst'],
            max_retry_limit=1
        )
    
    @agent
    def formatter(self) -> Agent:
        """Information Formatter agent."""
        return Agent(
            config=self.agents_config['formatter'],
            # verbose=True,
            llm=self.agent_llms['formatter'],
            max_retry_limit=1
        )
    
    @agent
    def compiler(self) -> Agent:
        """Report Compiler agent."""
        return Agent(
            config=self.agents_config['compiler'],
            # verbose=True,
            llm=self.agent_llms['compiler'],
            tools=[WebSearchTool(self.config)],
            max_retry_limit=1
        )
    
    @task
    def research_task(self) -> Task:
        """Define the research task."""
        logger.info(f"{self.tasks_config['research_task']}")
        return Task(
            config=self.tasks_config['research_task'],
            max_retry_limit=1
            # context=["query:" + self.current_query]
        )
    
    @task
    def analysis_task(self) -> Task:
        """Define the analysis task."""
        return Task(
            config=self.tasks_config['analysis_task'],
            max_retry_limit=1,
            # context=["previous_task:research_task", "query:" + self.current_query]
        )
    
    @task
    def formatting_task(self) -> Task:
        """Define the formatting task."""
        return Task(
            config=self.tasks_config['formatting_task'],
            max_retry_limit=1,
            # context=[
            #     "previous_task": "research_task", "query:" + self.current_query,
            #     "previous_task": "analysis_task", "query:" + self.current_query
            # ]
        )
    
    @task
    def compilation_task(self) -> Task:
        """Define the compilation task."""
        return Task(
            config=self.tasks_config['compilation_task'],
            max_retry_limit=1,
            # context=[
            #     "previous_task research_task query" + self.current_query,
            #     "previous_task": "analysis_task", "query": self.current_query,
            #     "previous_task": "formatting_task", "query": self.current_query
            # ]
        )
    
    @before_kickoff
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs before the crew starts."""
        logger.info(f"Preparing inputs for query: {inputs.get('query', '')}")
        
        # Log the user query if using simple logger
        if self.use_simple_logger and 'query' in inputs:
            self.simple_logger.log_agent_input("user", inputs['query'])
        
        return inputs
    
    @after_kickoff
    def process_output(self, output):
        """Process output after the crew finishes."""
        logger.info("Processing crew output")
        
        # Log the final result if using simple logger
        if self.use_simple_logger:
            self.simple_logger.log_agent_output("system", f"FINAL RESULT: {output}")
        
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
            verbose=True,
            memory=True,         # Enable crew memory
            cache=self.use_cache,
            manager_llm=self.agent_llms['manager'],  # Use manager LLM from ai_parameters.yaml
            output_log_file=True if self.debug_mode else None,
            share_crew=False     # Don't share data with CrewAI team
        )

class CrewAgentManager:
    """Manager for creating and running CrewAI agents for carbon footprint analysis."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, use_cache: bool = True,
                use_hierarchical: bool = True, use_simple_logger: bool = True):
        """Initialize the crew agent manager.
        
        Args:
            config: Configuration manager instance
            debug_mode: Whether to enable detailed debugging for agent interactions
            use_cache: Whether to use caching for query results
            use_hierarchical: Whether to use hierarchical process for potentially faster execution
            use_simple_logger: Whether to use simple logging system with separate files
        """
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
            use_hierarchical=use_hierarchical,
            use_simple_logger=use_simple_logger
        )
        
        # Initialize the debugger if debug mode is enabled
        self.debugger = AgentDebugger() if debug_mode else None
        
        # Track current session ID
        self.current_session_id = "current_session"
        
        logger.info("✅ CrewAgentManager initialized successfully with LiteLLM integration")
    
    def process_query(self, query: str, show_context: bool = False) -> Dict[str, Any]:
        """Process a query using the CrewAI agents.
        
        This function takes a user query about carbon footprint data and processes it
        through the CrewAI agent workflow, returning a comprehensive response with
        optional context information.
        
        Args:
            query: The user's query about carbon footprint data
            show_context: Whether to include context information in the result
            
        Returns:
            Dictionary containing the response and optional context information
        """
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
            
            # Run the crew with the query
            result = self.carbon_crew.crew().kickoff(
                inputs={"query": query}
            )
            
            # Format the response
            response_data = {
                "response": result,
                "context": {}
            }
            
            # Add sources if available and requested
            if show_context:
                sources = []
                # Extract sources if available in the response
                if isinstance(result, str) and "SOURCES:" in result:
                    # Extract sources section from the response
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
    
    def _generate_debug_html(self) -> str:
        """Generate an HTML debug file for the current session.
        
        Returns:
            Path to the HTML debug file
        """
        if not self.debug_mode or not self.current_session_id or not self.debugger:
            return ""
        
        return self.debugger.export_session_to_html(self.current_session_id)
    
    def get_agent_interactions(self, agent_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get the interaction history for one or all agents.
        
        Args:
            agent_name: Optional name of the agent to get interactions for
            
        Returns:
            Dictionary mapping agent names to their interactions
        """
        if not self.debug_mode:
            logger.warning("Debug mode is not enabled, no interactions to return")
            return {}
        
        # In this implementation, agent interactions are not directly accessible
        # You would need to implement a mechanism to track these separately
        return {}
    
    def print_agent_debug_info(self, agent_name: Optional[str] = None, 
                             interaction_id: Optional[int] = None) -> None:
        """Print debug information for agent interactions.
        
        Args:
            agent_name: Optional name of the agent to print interactions for
            interaction_id: Optional ID of the specific interaction to print
        """
        if not self.debug_mode:
            print("Debug mode is not enabled. Enable with debug_mode=True when initializing CrewAgentManager.")
            return
        
        if not self.current_session_id:
            print("No agent session has been run yet.")
            return
        
        if self.debugger:
            self.debugger.print_interaction_details(
                self.current_session_id, 
                agent_name, 
                interaction_id
            )
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.carbon_crew.cache_manager:
            self.carbon_crew.cache_manager.clear()
            logger.info("Query cache cleared")
        else:
            logger.warning("Caching is not enabled")