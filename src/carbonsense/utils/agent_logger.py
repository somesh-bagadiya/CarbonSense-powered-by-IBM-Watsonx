import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading

class AgentInteractionLogger:
    """
    Logger for AI agent interactions. Records prompts, completions, and other relevant data
    to a file that is replaced with each new run.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the interaction logger.
        
        Args:
            log_dir: Directory to store logs (defaults to 'logs' in project root)
        """
        if log_dir is None:
            # Get project root directory
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", ".."
            ))
            log_dir = os.path.join(project_root, "logs")
        
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Log file paths
        self.interaction_log_file = os.path.join(log_dir, "agent_interactions.log")
        self.debug_log_file = os.path.join(log_dir, "agent_debug.json")
        
        # Configure file logger
        self.file_logger = logging.getLogger("agent_interactions")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicate logging
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        # Create file handler that overwrites previous log
        file_handler = logging.FileHandler(self.interaction_log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Storage for interactions in memory
        self.interactions = []
        
        # Session ID counter
        self._session_counter = 0
        
        self.file_logger.info(f"AgentInteractionLogger initialized at {datetime.now().isoformat()}")
        self.file_logger.info(f"Logs stored at: {self.interaction_log_file}")
    
    def get_new_session_id(self) -> str:
        """Get a new unique session ID."""
        with self._lock:
            self._session_counter += 1
            return f"session_{self._session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_interaction(self, 
                      agent_name: str,
                      prompt: str, 
                      completion: str,
                      session_id: str,
                      parameters: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None) -> None:
        """
        Log a single interaction with an agent.
        
        Args:
            agent_name: Name of the agent
            prompt: The prompt sent to the agent
            completion: The completion received from the agent
            session_id: The session ID for grouping related interactions
            parameters: AI parameters used (temperature, tokens, etc.)
            metadata: Any additional metadata about the interaction
        """
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Create interaction record
        interaction = {
            "timestamp": timestamp,
            "session_id": session_id,
            "agent_name": agent_name,
            "parameters": parameters or {},
            "metadata": metadata or {},
            "prompt_length": len(prompt),
            "completion_length": len(completion)
        }
        
        # Create a full record for detailed logging
        full_record = interaction.copy()
        full_record["prompt"] = prompt
        full_record["completion"] = completion
        
        with self._lock:
            # Store in memory
            self.interactions.append(full_record)
            
            # Write to log file
            self.file_logger.info(f"Agent: {agent_name} | Session: {session_id}")
            self.file_logger.info(f"Parameters: {json.dumps(parameters or {})}")
            self.file_logger.info(f"Prompt length: {len(prompt)} chars | Completion length: {len(completion)} chars")
            
            # Log truncated prompt and completion in the log file
            max_log_length = 500  # Limit log file entries to prevent massive logs
            self.file_logger.info(f"Prompt (truncated): {prompt[:max_log_length]}...")
            self.file_logger.info(f"Completion (truncated): {completion[:max_log_length]}...")
            self.file_logger.info("-" * 80)
            
            # Write the complete data to the JSON debug file
            self._write_debug_json()
    
    def _write_debug_json(self):
        """Write the complete interaction history to a JSON file."""
        try:
            with open(self.debug_log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "interactions": self.interactions
                }, f, indent=2)
        except Exception as e:
            self.file_logger.error(f"Failed to write debug JSON: {str(e)}")
    
    def get_interactions_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all interactions for a specific session.
        
        Args:
            session_id: The session ID to filter by
            
        Returns:
            List of interaction records for the specified session
        """
        with self._lock:
            return [
                interaction for interaction in self.interactions
                if interaction["session_id"] == session_id
            ]
    
    def get_interactions_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get all interactions for a specific agent.
        
        Args:
            agent_name: The agent name to filter by
            
        Returns:
            List of interaction records for the specified agent
        """
        with self._lock:
            return [
                interaction for interaction in self.interactions
                if interaction["agent_name"] == agent_name
            ]
    
    def clear_interactions(self):
        """Clear all logged interactions."""
        with self._lock:
            self.interactions = []
            
            # Reset the log files
            with open(self.interaction_log_file, 'w') as f:
                f.write(f"Logs cleared at {datetime.now().isoformat()}\n")
            
            with open(self.debug_log_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "interactions": []
                }, f, indent=2)
            
            self.file_logger.info("Interaction logs cleared")