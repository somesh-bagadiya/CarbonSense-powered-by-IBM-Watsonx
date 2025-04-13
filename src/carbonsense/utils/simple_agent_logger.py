import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# Set up logger
logger = logging.getLogger(__name__)

class SimpleAgentLogger:
    """A simple logging system that overwrites files instead of creating new ones for each run."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls, log_dir: str = None):
        """Singleton pattern to ensure only one logger instance exists."""
        if cls._instance is None:
            cls._instance = super(SimpleAgentLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: str = None):
        """Initialize the simple agent logger.
        
        Args:
            log_dir: Directory to store logs
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        # Set up log directory
        if not log_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(base_dir, "logs")
            
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create context directory for transfers
        self.context_dir = os.path.join(log_dir, "context")
        os.makedirs(self.context_dir, exist_ok=True)
        
        # Track initialization time
        self.init_time = datetime.now()
        logger.info(f"SimpleAgentLogger initialized at {self.init_time.isoformat()}")
        logger.info(f"Logs will be stored in: {log_dir}")
        
        # Mark as initialized
        self._initialized = True
    
    def _ensure_agent_dir(self, agent_name: str) -> str:
        """Ensure agent directory exists and return path.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Path to the agent directory
        """
        agent_dir = os.path.join(self.log_dir, agent_name.lower())
        os.makedirs(agent_dir, exist_ok=True)
        return agent_dir
    
    def log_agent_input(self, agent_name: str, input_text: str) -> None:
        """Log agent input to file, overwriting existing file.
        
        Args:
            agent_name: Name of the agent
            input_text: Input text to log
        """
        agent_dir = self._ensure_agent_dir(agent_name)
        input_file = os.path.join(agent_dir, "input.txt")
        
        with open(input_file, 'w', encoding='utf-8') as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"--- {agent_name.upper()} INPUT ---\n")
            file.write(f"Time: {timestamp}\n\n")
            file.write(input_text)
    
    def log_agent_output(self, agent_name: str, output_text: str) -> None:
        """Log agent output to file, overwriting existing file.
        
        Args:
            agent_name: Name of the agent
            output_text: Output text to log
        """
        agent_dir = self._ensure_agent_dir(agent_name)
        output_file = os.path.join(agent_dir, "output.txt")
        
        with open(output_file, 'w', encoding='utf-8') as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"--- {agent_name.upper()} OUTPUT ---\n")
            file.write(f"Time: {timestamp}\n\n")
            file.write(output_text)
    
    def log_context_transfer(self, from_agent: str, to_agent: str, context: str) -> None:
        """Log context transfer between agents.
        
        Args:
            from_agent: Name of the source agent
            to_agent: Name of the target agent
            context: Context data to transfer
        """
        context_file = os.path.join(self.context_dir, f"{from_agent}_to_{to_agent}.txt")
        
        with open(context_file, 'w', encoding='utf-8') as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"--- CONTEXT TRANSFER: {from_agent} → {to_agent} ---\n")
            file.write(f"Time: {timestamp}\n\n")
            file.write(context)
    
    def log_tool_execution(self, agent_name: str, tool_name: str, input_data: str, output_data: str) -> None:
        """Log tool execution by an agent.
        
        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool
            input_data: Input data for the tool
            output_data: Output data from the tool
        """
        agent_dir = self._ensure_agent_dir(agent_name)
        tool_file = os.path.join(agent_dir, f"tool_{tool_name}.txt")
        
        with open(tool_file, 'w', encoding='utf-8') as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"--- {agent_name.upper()} TOOL: {tool_name} ---\n")
            file.write(f"Time: {timestamp}\n\n")
            file.write(f"INPUT:\n{'-' * 40}\n{input_data}\n\n")
            file.write(f"OUTPUT:\n{'-' * 40}\n{output_data}")

# Global singleton accessor
_global_instance = None

def get_simple_agent_logger(log_dir: str = None) -> SimpleAgentLogger:
    """Get the singleton SimpleAgentLogger instance.
    
    Args:
        log_dir: Optional directory to store logs
        
    Returns:
        SimpleAgentLogger instance
    """
    global _global_instance
    if _global_instance is None:
        _global_instance = SimpleAgentLogger(log_dir)
    return _global_instance