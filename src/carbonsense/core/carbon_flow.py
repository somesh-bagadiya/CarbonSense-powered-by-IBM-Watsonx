import logging
import os
import yaml
import json
import time
import traceback
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from crewai import LLM
from crewai.flow.flow import Flow, listen, start, or_

from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.litellm_watsonx_service import LiteLLMWatsonxService
from ..utils.logger import setup_logger
from ..core.crew_agent import CarbonSenseCrew, CrewAgentManager

# Set up logger with colored output
logger = setup_logger(__name__)

class CarbonSenseState(BaseModel):
    """State model for the CarbonSense flow."""
    id: str = ""  # Auto-populated by Flow
    query: str = ""
    query_classification: Dict[str, Any] = Field(default_factory=dict)
    entities: Dict[str, Any] = Field(default_factory=dict)
    normalized_entities: Dict[str, Any] = Field(default_factory=dict)
    cache_results: List[Dict[str, Any]] = Field(default_factory=list)
    milvus_research: List[Dict[str, Any]] = Field(default_factory=list)
    serper_research: List[Dict[str, Any]] = Field(default_factory=list)
    harmonized_data: List[List[Dict[str, Any]]] = Field(default_factory=list)
    estimated_footprints: List[Dict[str, Any]] = Field(default_factory=list)
    ranked_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    comparison_result: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    final_answer: Dict[str, Any] = Field(default_factory=dict)

class CarbonSenseFlow(Flow[CarbonSenseState]):
    """Carbon footprint analysis flow using CrewAI's Flow approach."""
    
    def __init__(self, config: ConfigManager, debug_mode: bool = False, 
                 use_cache: bool = False, store_thoughts: bool = False,
                 agent_callback=None):
        """Initialize the CarbonSense flow.
        
        Args:
            config: Configuration manager instance
            debug_mode: Whether to enable debug mode
            use_cache: Whether to use caching (ignored now)
            store_thoughts: Whether to store agent thoughts and reasoning in log files
            agent_callback: Optional callback function to notify about current agent step
        """
        super().__init__()
        logger.info("Initializing CarbonSenseFlow...")
        
        # Store configuration
        self.config = config
        self.debug_mode = debug_mode
        self.use_cache = False
        self.store_thoughts = store_thoughts
        self.agent_callback = agent_callback
        
        # Initialize services
        self.milvus = MilvusService(config)
        self.watsonx = LiteLLMWatsonxService(config)
        # Removing cache_manager initialization
        
        # Set up LiteLLM configuration for WatsonX
        os.environ["LITELLM_MODEL"] = "watsonx/meta-llama/llama-2-70b-instruct"
        os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL", "")
        os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY", "")
        os.environ["WATSONX_PROJECT_ID"] = os.getenv("WATSONX_PROJECT_ID", "")
        
        # Initialize the crew agent manager which contains all our agents
        self.crew_manager = CrewAgentManager(
            config=config,
            debug_mode=debug_mode,
            use_cache=False,  # Always set to False
            use_hierarchical=True,
            store_thoughts=store_thoughts
        )
        
        # Create logs directory if it doesn't exist and store_thoughts is enabled
        if self.store_thoughts:
            os.makedirs("logs/thoughts", exist_ok=True)
            
        # Create visualizations directory
        os.makedirs("logs/visualizations", exist_ok=True)
            
        logger.info("✅ CarbonSenseFlow initialized successfully")
        
    @start()
    def process_query(self):
        """Starting point: Process the user query from the state"""
        logger.info(f"Processing query: {self.state.query}")
        
        # Removed caching logic
        
        return self.state.query
        
    @listen(process_query)
    def classify_query(self, query):
        """Classify the query intent and category"""
        # Removed cache hit check
            
        logger.info("Classifying query intent and category")
        
        # Execute the classification task using the crew_manager
        result = self.execute_task("query_classification_task", {"query": query})
        
        # Store in state
        self.state.query_classification = result
        logger.info(f"Query classified as: {result}")
        
        return result
        
    @listen(classify_query)
    def extract_entities(self, classification):
        """Extract entities from the query"""
        # Removed cache hit check
            
        logger.info("Extracting entities from query")
        
        # Execute the entity extraction task
        result = self.execute_task("entity_extraction_task", {
            "query": self.state.query,
            "classification": classification
        })
        
        # Store in state
        self.state.entities = result
        logger.info(f"Extracted entities: {result}")
        
        return result
        
    @listen(extract_entities)
    def normalize_units(self, entities):
        """Normalize extracted entities to standard units"""
        # Removed cache hit check
            
        # Skip if no entities were extracted
        if not entities:
            logger.info("No entities to normalize, skipping normalization")
            return None
            
        logger.info("Normalizing units")
        
        # Execute the unit normalization task
        result = self.execute_task("unit_normalization_task", {
            "entities": entities,
            "query": self.state.query
        })
        
        # Store in state
        self.state.normalized_entities = result
        logger.info(f"Normalized entities: {result}")
        
        return result
        
    @listen(normalize_units)
    def check_cache(self, normalized_entities):
        """Check if results are in cache - now always returns MISS"""
        # Removed cache hit check
            
        # Skip if no normalization
        if not normalized_entities:
            logger.info("Skipping cache lookup due to missing entities")
            return None
            
        logger.info("Cache lookup disabled, proceeding with research")
        
        # Always return MISS since caching is disabled
        return {"status": "MISS"}
        
    @listen(check_cache)
    def research_milvus(self, cache_results):
        """Research using Milvus vector database"""
        # Removed cache hit check
            
        logger.info("Researching with Milvus vector database")
        
        # Execute the Milvus research task
        result = self.execute_task("milvus_research_task", {
            "normalized_entities": self.state.normalized_entities,
            "cache_results": cache_results,
            "query": self.state.query
        })
        
        # Store in state
        self.state.milvus_research = result
        logger.info(f"Milvus research completed with {len(result) if isinstance(result, list) else 0} results")
        
        return result
        
    @listen(check_cache)
    def research_serper(self, cache_results):
        """Research using Serper web search"""
        # Removed cache hit check
            
        logger.info("Researching with Serper web search")
        
        # Execute the Serper research task
        result = self.execute_task("serper_research_task", {
            "normalized_entities": self.state.normalized_entities,
            "cache_results": cache_results,
            "query": self.state.query
        })
        
        # Store in state
        self.state.serper_research = result
        logger.info(f"Serper research completed with {len(result) if isinstance(result, list) else 0} results")
        
        return result
    
    @listen(or_(research_milvus, research_serper))
    def harmonize_data(self, *research_results):
        """Harmonize data from multiple research sources"""
        # Removed cache hit check
            
        # Check if we have any research results
        if not self.state.milvus_research and not self.state.serper_research:
            logger.info("No research results to harmonize")
            return None
            
        logger.info("Harmonizing research data")
        
        # Execute the harmonization task
        result = self.execute_task("harmonise_task", {
            "milvus_data": self.state.milvus_research,
            "serper_data": self.state.serper_research,
            "normalized_entities": self.state.normalized_entities,
            "query_classification": self.state.query_classification
        })
        
        # Store in state
        self.state.harmonized_data = result
        logger.info(f"Data harmonization completed")
        
        return result
        
    @listen(harmonize_data)
    def estimate_carbon(self, harmonized_data):
        """Calculate carbon footprint estimates"""
        # Removed cache hit check
            
        # Skip if no harmonized data
        if not harmonized_data:
            logger.info("Skipping carbon estimation due to missing data")
            return None
            
        logger.info("Estimating carbon footprints")
        
        # Execute the carbon estimation task
        result = self.execute_task("carbon_estimation_task", {
            "harmonized_data": harmonized_data,
            "normalized_entities": self.state.normalized_entities,
            "query_classification": self.state.query_classification
        })
        
        # Store in state
        self.state.estimated_footprints = result
        logger.info(f"Carbon estimation completed")
        
        return result
        
    @listen(estimate_carbon)
    def rank_metrics(self, estimated_footprints):
        """Rank and select the best metrics"""
        # Removed cache hit check
            
        # Skip if no estimated footprints
        if not estimated_footprints:
            logger.info("Skipping metric ranking due to missing estimates")
            return None
            
        logger.info("Ranking carbon footprint metrics")
        
        # Execute the metric ranking task
        result = self.execute_task("rank_metrics_task", {
            "estimated_footprints": estimated_footprints,
            "query_classification": self.state.query_classification
        })
        
        # Store in state
        self.state.ranked_metrics = result
        logger.info(f"Metric ranking completed")
        
        return result
    
    @listen(or_(rank_metrics, process_query))
    def intent_specific_processing(self, *previous_results):
        """Route to intent-specific processing based on classification"""
        # Removed cache hit check
            
        # Safely handle query_classification which might be None
        if not hasattr(self.state, 'query_classification') or self.state.query_classification is None:
            logger.warning("Query classification is missing or None, defaulting to estimate intent")
            intent = "estimate"
        else:
            # Get the query intent, default to estimate
            intent = self.state.query_classification.get("query_intent", "estimate")
        
        logger.info(f"Processing intent-specific logic for: {intent}")
        
        # Route based on intent
        if intent == "compare":
            return self.process_comparison()
        elif intent == "suggest":
            return self.process_recommendations()
        elif intent == "myth_bust":
            return self.process_explanation()
        elif intent == "lifecycle":
            # Lifecycle handling is part of the carbon estimation process
            return self.state.ranked_metrics
        else:  # Default for estimate
            return self.state.ranked_metrics
    
    def process_comparison(self):
        """Process comparison intent"""
        logger.info("Processing comparison")
        
        # Check if we have ranked metrics to compare
        if not self.state.ranked_metrics:
            logger.warning("No ranked metrics available for comparison")
            return None
            
        # Execute the comparison task
        result = self.execute_task("comparison_task", {
            "ranked_metrics": self.state.ranked_metrics,
            "query_classification": self.state.query_classification
        })
        
        # Store in state
        self.state.comparison_result = result
        logger.info(f"Comparison processing completed")
        
        return result
        
    def process_recommendations(self):
        """Process suggestion intent"""
        logger.info("Processing recommendations")
        
        # Check if we have ranked metrics for recommendations
        if not self.state.ranked_metrics:
            logger.warning("No ranked metrics available for recommendations")
            return None
            
        # Execute the recommendation task
        result = self.execute_task("recommendation_task", {
            "ranked_metrics": self.state.ranked_metrics,
            "query_classification": self.state.query_classification,
            "estimated_footprints": self.state.estimated_footprints
        })
        
        # Store in state
        self.state.recommendations = result
        logger.info(f"Recommendation processing completed")
        
        return result
        
    def process_explanation(self):
        """Process myth-busting intent"""
        logger.info("Processing explanation/myth-busting")
        
        # Check if we have ranked metrics for explanation
        if not self.state.ranked_metrics:
            logger.warning("No ranked metrics available for explanation")
            return None
            
        # Execute the explanation task
        result = self.execute_task("explanation_task", {
            "ranked_metrics": self.state.ranked_metrics,
            "query_classification": self.state.query_classification,
            "estimated_footprints": self.state.estimated_footprints
        })
        
        # Store in state
        self.state.explanation = result
        logger.info(f"Explanation processing completed")
        
        return result
    
    @listen(intent_specific_processing)
    def format_answer(self, intent_result):
        """Format the final answer"""
        # Removed cache hit check
            
        logger.info("Formatting final answer")
        
        # Safely handle query_classification
        if not hasattr(self.state, 'query_classification') or self.state.query_classification is None:
            logger.warning("Query classification is missing or None, defaulting to estimate intent")
            intent = "estimate"
        else:
            # Determine which processed result to use based on intent
            intent = self.state.query_classification.get("query_intent", "estimate")
        
        # Prepare formatter input with safe access to state fields
        formatter_input = {
            "query": getattr(self.state, 'query', ''),
            "query_classification": getattr(self.state, 'query_classification', {}),
            "ranked_metrics": getattr(self.state, 'ranked_metrics', [])
        }
        
        # Add intent-specific results
        if intent == "compare" and hasattr(self.state, 'comparison_result') and self.state.comparison_result:
            formatter_input["comparison"] = self.state.comparison_result
        elif intent == "suggest" and hasattr(self.state, 'recommendations') and self.state.recommendations:
            formatter_input["recommendations"] = self.state.recommendations
        elif intent == "myth_bust" and hasattr(self.state, 'explanation') and self.state.explanation:
            formatter_input["explanation"] = self.state.explanation
        elif intent == "estimate" or intent == "lifecycle":
            if hasattr(self.state, 'estimated_footprints'):
                formatter_input["carbon_estimation"] = self.state.estimated_footprints
            
        # Execute the answer formatting task
        result = self.execute_task("answer_formatting_task", formatter_input)
        
        # If we have no result, return an error message
        if result is None:
            logger.warning("Failed to format answer")
            result = {
                "error": "Could not process query",
                "response": "I'm sorry, but I couldn't process your carbon footprint query. Please try again or rephrase your question."
            }
        
        # Store in state
        if hasattr(self.state, 'final_answer'):
            self.state.final_answer = result
            logger.info(f"Answer formatting completed")
        
        # Removed caching logic
        
        return result
    
    def _process_task_output(self, task_output):
        """Helper method to process task output consistently"""
        try:
            # Try to parse as JSON if it's a string
            if isinstance(task_output, str):
                if task_output.strip().startswith('{') and task_output.strip().endswith('}'):
                    import json
                    return json.loads(task_output)
                elif task_output.strip().startswith('[') and task_output.strip().endswith(']'):
                    import json
                    return json.loads(task_output)
            return task_output
        except Exception as e:
            logger.warning(f"Error parsing task output: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            return task_output
            
    def _save_task_result_to_file(self, task_name, result):
        """Save task result to its corresponding output file.
        
        Args:
            task_name: Name of the task
            result: Result to save
        """
        try:
            # Get the output file from the task config
            crew = self.crew_manager.carbon_crew
            task_config = crew.tasks_config.get(task_name, {})
            
            if not task_config or 'output_file' not in task_config:
                logger.warning(f"No output file configured for task: {task_name}")
                return
                
            output_file = task_config['output_file']
            
            # Handle relative paths
            if not os.path.isabs(output_file):
                output_file = os.path.join(os.getcwd(), output_file)
                
            # Make sure the directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Write the result to the file
            with open(output_file, 'w') as f:
                if isinstance(result, (dict, list)):
                    import json
                    json.dump(result, f, indent=2)
                else:
                    f.write(str(result))
                    
            logger.info(f"Saved task result to output file: {output_file}")
        except Exception as e:
            logger.error(f"Error saving task result to file: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
    
    def execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Any:
        """Execute a task using the crew manager and handle the output.
        
        Args:
            task_name (str): The name of the task to execute
            inputs (Dict[str, Any]): The inputs to the task
            
        Returns:
            Any: The processed task output
        """
        task_start_time = time.time()
        logger.info(f"Executing task: {task_name}")
        
        try:
            # Notify via callback if provided
            if self.agent_callback:
                try:
                    # Get the agent associated with this task
                    agent_info = self.crew_manager.get_agent_for_task(task_name)
                    if agent_info:
                        self.agent_callback(agent_info)
                except Exception as cb_error:
                    logger.error(f"Error in agent callback: {cb_error}")
                    
            # Execute the task
            result = self.crew_manager.execute_task(task_name, inputs)
            
            # Log execution time
            task_end_time = time.time()
            execution_time = task_end_time - task_start_time
            logger.info(f"Task {task_name} executed in {execution_time:.2f} seconds")
            
            # Process the result
            processed_result = self._process_task_output(result)
            
            # Save the result to a file if store_thoughts is enabled
            if self.store_thoughts:
                self._save_task_result_to_file(task_name, processed_result)
                
            # Try to capture detailed agent output for thought streaming
            try:
                # Import here to avoid circular imports
                from ..web.app import capture_agent_output
                
                # Get the agent associated with this task
                agent_info = self.crew_manager.get_agent_for_task(task_name)
                if agent_info and hasattr(agent_info, 'role'):
                    # Extract the full agent output including thoughts
                    agent_role = agent_info.role
                    agent_output = str(result)
                    
                    # Capture the detailed output
                    capture_agent_output(agent_role, agent_output)
            except Exception as capture_error:
                logger.warning(f"Error capturing detailed agent output: {capture_error}")
                
            return processed_result
            
        except Exception as e:
            logger.error(f"Error executing task {task_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a partial result or error indicator
            return {"error": str(e)}
    
    def process_query_with_flow(self, query: str, save_visualization: bool = True) -> Dict[str, Any]:
        """Process a query using the flow.
        
        Args:
            query: User query string
            save_visualization: Whether to save a visualization of the flow
            
        Returns:
            Processed result as a dictionary
        """
        try:
            # Create state with the query
            state = CarbonSenseState(query=query)
            
            # Start the flow with the initial state
            # Use model_dump instead of dict for Pydantic v2 compatibility
            if hasattr(state, 'model_dump'):
                # Pydantic v2
                inputs = state.model_dump()
            else:
                # Pydantic v1 fallback
                inputs = state.dict()
            
            # This is a blocking call that shouldn't be used in an async context
            # For async contexts, use process_query_with_flow_async instead
            result = self.kickoff(inputs=inputs)
            
            # Save visualization if requested
            if save_visualization:
                self.save_flow_visualization(query)
            
            # Return the final answer from state
            if hasattr(self.state, 'final_answer'):
                return self.state.final_answer
            else:
                logger.warning("No final answer in state after flow execution")
                return {"error": "No final answer generated", "response": "Failed to process query"}
        except Exception as e:
            logger.error(f"Error processing query with flow: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"error": str(e), "response": f"An error occurred while processing your query: {str(e)}"}
            
    async def process_query_with_flow_async(self, query: str, save_visualization: bool = True, agent_callback=None) -> Dict[str, Any]:
        """Process a query using the flow asynchronously.
        
        Args:
            query: User query string
            save_visualization: Whether to save a visualization of the flow
            agent_callback: Optional callback function to notify about current agent step
            
        Returns:
            Processed result as a dictionary
        """
        try:
            # Store the callback
            self.agent_callback = agent_callback
            
            # Create state with the query
            state = CarbonSenseState(query=query)
            
            # Start the flow with the initial state
            # Use model_dump instead of dict for Pydantic v2 compatibility
            if hasattr(state, 'model_dump'):
                # Pydantic v2
                inputs = state.model_dump()
            else:
                # Pydantic v1 fallback
                inputs = state.dict()
                
            # Run in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.kickoff(inputs=inputs))
            
            # Save visualization if requested (non-blocking)
            if save_visualization:
                await loop.run_in_executor(None, lambda: self.save_flow_visualization(query))
            
            # Return the final answer from state
            if hasattr(self.state, 'final_answer'):
                return self.state.final_answer
            else:
                logger.warning("No final answer in state after flow execution")
                return {"error": "No final answer generated", "response": "Failed to process query"}
        except Exception as e:
            logger.error(f"Error processing query with flow asynchronously: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"error": str(e), "response": f"An error occurred while processing your query: {str(e)}"}
        finally:
            # Clear the callback
            self.agent_callback = None
    
    def save_flow_visualization(self, query: str = None) -> str:
        """Generate and save a visualization of the flow.
        
        Args:
            query: Optional query string to include in the filename (ignored now - using consistent filename)
            
        Returns:
            Path to the saved visualization file
        """
        try:
            # Use a consistent filename that will be replaced on each run
            output_path = os.path.join("logs", "visualizations", "carbonsense_flow")
            image_path = f"{output_path}.png"
            
            # Generate and save the visualization as PNG image
            logger.info(f"Saving flow visualization to {image_path}")
            
            # First save as HTML (required by crewai Flow API)
            self.plot(output_path)
            
            logger.info(f"Flow visualization saved successfully")
            return image_path
        except Exception as e:
            logger.error(f"Error saving flow visualization: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None 