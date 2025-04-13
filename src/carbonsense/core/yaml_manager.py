import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from ..utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class YamlManager:
    """Manager for loading and parsing YAML configuration for CrewAI."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the YAML manager.
        
        Args:
            config_dir: Optional path to the config directory
        """
        # If no config directory is provided, use the default location
        if config_dir is None:
            # Get the directory where this file is located
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the carbonsense directory
            carbonsense_dir = os.path.dirname(current_file_dir)
            # Go to the config directory
            config_dir = os.path.join(carbonsense_dir, 'config')
        
        self.config_dir = config_dir
        self.agents_dir = os.path.join(config_dir, 'agents')
        self.tasks_dir = os.path.join(config_dir, 'tasks')
        self.crews_dir = os.path.join(config_dir, 'crews')
        
        # Cache for loaded YAML to avoid repeated disk reads
        self._yaml_cache = {}
        
        logger.info(f"Initialized YamlManager with config directory: {config_dir}")
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load a YAML file and return its content as a dictionary.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary containing the YAML content
        """
        # Check if this file is already cached
        if file_path in self._yaml_cache:
            return self._yaml_cache[file_path]
            
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
                logger.info(f"Successfully loaded YAML file: {file_path}")
                
                # Cache the loaded YAML
                self._yaml_cache[file_path] = yaml_content
                
                return yaml_content
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
            raise
    
    def load_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Load an agent configuration from YAML.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary containing the agent configuration
        """
        agent_file = os.path.join(self.agents_dir, f"{agent_name}.yaml")
        return self._load_yaml(agent_file)
    
    def load_task_config(self, task_name: str) -> Dict[str, Any]:
        """Load a task configuration from YAML.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary containing the task configuration
        """
        task_file = os.path.join(self.tasks_dir, f"{task_name}.yaml")
        return self._load_yaml(task_file)
    
    def load_crew_config(self, crew_name: str) -> Dict[str, Any]:
        """Load a crew configuration from YAML.
        
        Args:
            crew_name: Name of the crew
            
        Returns:
            Dictionary containing the crew configuration
        """
        crew_file = os.path.join(self.crews_dir, f"{crew_name}.yaml")
        return self._load_yaml(crew_file)
    
    def create_agent(self, 
                    agent_name: str, 
                    llm_provider, 
                    tools_dict: Optional[Dict[str, BaseTool]] = None) -> Agent:
        """Create an agent from configuration.
        
        Args:
            agent_name: Name of the agent to create
            llm_provider: LLM provider instance
            tools_dict: Dictionary of available tools (optional)
            
        Returns:
            Created agent instance
        """
        try:
            # Load agent configuration
            config = self.load_agent_config(agent_name)
            
            # Create the agent with basic configuration
            agent = Agent(
                role=config['role'],
                goal=config['goal'],
                backstory=config['backstory'],
                llm=llm_provider
            )
            
            # After agent is created, we can set additional properties
            
            # Set verbose mode if specified
            if hasattr(agent, 'verbose') and 'verbose' in config:
                agent.verbose = config.get('verbose', True)
                
            # Set allow_delegation if specified
            if hasattr(agent, 'allow_delegation') and 'allow_delegation' in config:
                agent.allow_delegation = config.get('allow_delegation', False)
            
            # Add tools after agent creation
            if tools_dict and config.get('tools'):
                agent_tools = []
                for tool_name in config['tools']:
                    if tool_name in tools_dict:
                        agent_tools.append(tools_dict[tool_name])
                    else:
                        logger.warning(f"Tool '{tool_name}' specified in agent '{agent_name}' not found in tools dictionary")
                        
                # Only set tools if we found any valid ones
                if agent_tools:
                    agent.tools = agent_tools
                else:
                    logger.warning(f"No valid tools found for agent '{agent_name}'")
            
            logger.info(f"Created agent: {agent_name}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_name}: {str(e)}")
            raise
    
    def create_task(self, 
                   task_name: str, 
                   agents_dict: Dict[str, Agent],
                   tasks_dict: Optional[Dict[str, Task]] = None,
                   query: Optional[str] = None) -> Task:
        """Create a Task object from YAML configuration.
        
        Args:
            task_name: Name of the task
            agents_dict: Dictionary mapping agent names to agent objects
            tasks_dict: Dictionary mapping task names to task objects (for context)
            query: Optional query string to format into the description
            
        Returns:
            Configured Task object
        """
        config = self.load_task_config(task_name)
        
        # Get the agent for this task
        agent_name = config.get('agent')
        if not agent_name or agent_name not in agents_dict:
            logger.error(f"Agent '{agent_name}' specified in task '{task_name}' not found")
            raise ValueError(f"Agent '{agent_name}' not found for task '{task_name}'")
        
        agent = agents_dict[agent_name]
        
        # Format the description with the query if provided
        description = config.get('description', '')
        if query:
            description = description.format(query=query)
        
        # Get the context tasks
        context = []
        if config.get('context') and tasks_dict:
            for context_task_name in config['context']:
                if context_task_name in tasks_dict:
                    context.append(tasks_dict[context_task_name])
                else:
                    logger.warning(f"Context task '{context_task_name}' specified in task '{task_name}' not found")
        
        # Create the task
        task = Task(
            description=description,
            expected_output=config.get('expected_output', ''),
            agent=agent,
            context=context
        )
        
        logger.info(f"Created task: {task_name}")
        return task
    
    def create_crew(self, 
                   crew_name: str,
                   agents_dict: Dict[str, Agent],
                   tasks_list: List[Task],
                   manager_llm=None) -> Crew:
        """Create a Crew object from YAML configuration.
        
        Args:
            crew_name: Name of the crew
            agents_dict: Dictionary mapping agent names to agent objects
            tasks_list: List of task objects
            manager_llm: Optional LLM to use as the manager for hierarchical process
            
        Returns:
            Configured Crew object
        """
        config = self.load_crew_config(crew_name)
        
        # Get process type
        process_str = config.get('process', 'sequential')
        if process_str.lower() == 'hierarchical':
            process = Process.hierarchical
        else:
            process = Process.sequential
        
        # Create a properly formatted config dict that CrewAI 0.114.0 expects
        config_dict = {
            "max_retry_attempts": 3,  # Limit retries to prevent infinite loops
            "task_timeout": 300,      # Timeout tasks after 5 minutes (300 seconds)
            "dashify": False          # Disable dashboard to avoid spam
        }
        
        # Create the crew with the correct parameter format for 0.114.0
        crew = Crew(
            agents=list(agents_dict.values()),
            tasks=tasks_list,
            process=process,
            verbose=config.get('verbose', True),
            config=config_dict
        )
        
        # Set manager_llm separately if using hierarchical process
        if process == Process.hierarchical and manager_llm and hasattr(crew, 'manager_llm'):
            crew.manager_llm = manager_llm
            logger.info(f"Using hierarchical process with custom manager LLM")
        
        logger.info(f"Created crew with process type: {process_str}")
        return crew
        
    def clear_cache(self):
        """Clear the YAML file cache."""
        self._yaml_cache = {}
        logger.info("YAML cache cleared")