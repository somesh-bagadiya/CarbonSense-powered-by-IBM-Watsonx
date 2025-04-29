from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Response, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import json
import time
import tempfile
import shutil
from typing import Optional, List, Dict, Any
import random
import asyncio
import functools
import threading
import logging
import traceback
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import sys
import queue
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from ..config.config_manager import ConfigManager
from ..core.crew_agent import CrewAgentManager
from ..core.carbon_flow import CarbonSenseFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("carbonsense")

# Add the project root to the Python path to help with imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

# Import the core carbon sense functionality
from src.carbonsense.config.config_manager import ConfigManager
from src.carbonsense.core.crew_agent import CrewAgentManager

# Constants for audio recording
SAMPLE_RATE = 44100  # Sample rate in Hz
CHANNELS = 1  # Mono recording
global_transcript=""

# Define a global variable for the current query ID and thoughts file path
CURRENT_QUERY_ID = None
# Create logs/thoughts directory if it doesn't exist
thoughts_dir = os.path.join(project_root, "logs", "thoughts")
os.makedirs(thoughts_dir, exist_ok=True)
THOUGHTS_FILE_PATH = os.path.join(thoughts_dir, "persistant_thoughts.json")

# Initialize a global state for the thought process
current_thought_state = {
    "query_id": None,
    "thoughts": [],
    "status": "idle"  # idle, processing, complete, error
}

# Define functions that will be used by endpoints
def select_input_device_interactively() -> int | None:
    """Lists available input devices and prompts the user to select one."""
    print("\nAvailable Input Devices:")
    print("="*25)
    try:
        devices = sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

        if not input_devices:
            logger.error(" No input devices found.")
            return None

        for i, (idx, device) in enumerate(input_devices):
            default_marker = " (default)" if idx == sd.default.device[0] else ""
            print(f"  {i+1}: {device['name']}{default_marker} (Index: {idx})")

        while True:
            try:
                choice = input(f"\nEnter the number of the microphone to use (1-{len(input_devices)}): ")
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(input_devices):
                    selected_device_index = input_devices[choice_index][0]
                    logger.info(f"Selected device: {input_devices[choice_index][1]['name']} (Index: {selected_device_index})")
                    return selected_device_index
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(input_devices)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle case where input stream is closed (e.g., piping)
                 logger.error("Input stream closed. Cannot select device interactively.")
                 return None
            except KeyboardInterrupt:
                logger.info("\nDevice selection cancelled by user.")
                return None

    except Exception as e:
        logger.error(f" Could not query or select audio devices: {e}")
        return None

def record_audio(duration: int, sample_rate: int, channels: int, device_index: int = None) -> str:
    """Records audio from the microphone for a specified duration and saves it to a temporary WAV file."""
    try:
        device_info = f" using device index {device_index}" if device_index is not None else " using default device"
        num_frames = int(duration * sample_rate)
        logger.info(f" Attempting to record for {duration} seconds ({num_frames} frames) at {sample_rate} Hz{device_info}... Speak now!")

        start_time = time.time()
        recording = sd.rec(num_frames, samplerate=sample_rate, channels=channels, dtype='int16', device=device_index)
        sd.wait()  # Wait until recording is finished
        end_time = time.time()

        actual_duration = end_time - start_time
        logger.info(f" Recording finished. Actual duration: {actual_duration:.2f} seconds.")

        # Check if actual duration significantly deviates from requested duration
        if abs(actual_duration - duration) > 2: # Allow a 2-second buffer
             logger.warning(f" Recording duration ({actual_duration:.2f}s) significantly differs from requested ({duration}s).")

        # Create a temporary file to save the recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file_path = temp_file.name
        temp_file.close() # Close the file handle so wav.write can open it

        # Save as WAV file using scipy
        wav.write(temp_file_path, sample_rate, recording)
        logger.info(f" Audio saved temporarily to: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f" Error during audio recording: {str(e)}")
        # Attempt to clean up if file was created
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
                 logger.info(f" Cleaned up temporary file: {temp_file_path}")
             except OSError as rm_err:
                 logger.error(f" Error cleaning up temp file {temp_file_path}: {rm_err}")
        raise

def transcribe_audio(config: ConfigManager, audio_file_path: str) -> str:
    """Transcribes audio using IBM Watson Speech to Text."""
    stt_config = config.get_stt_config()
    api_key = stt_config.get("api_key")
    service_url = stt_config.get("url")

    if not api_key or not service_url:
        logger.error(" IBM Speech to Text API key or URL not found in configuration.")
        raise ValueError("IBM STT configuration missing.")

    authenticator = IAMAuthenticator(api_key)
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )
    speech_to_text.set_service_url(service_url)

    try:
        with open(audio_file_path, 'rb') as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/wav', # Adjust content type based on your audio file format
                model='en-US_BroadbandModel' # Specify the appropriate model
            ).get_result()
        
        if response['results']:
            transcript = response['results'][0]['alternatives'][0]['transcript']
            logger.info(" Transcription successful.")
            return transcript
        else:
            logger.warning(" No transcription results returned.")
            return ""
            
    except Exception as e:
        logger.error(f" Error during transcription: {str(e)}")
        raise
    finally:
        # Clean up the temporary audio file after transcription attempt
        if os.path.exists(audio_file_path) and tempfile.gettempdir() in os.path.dirname(audio_file_path):
            try:
                os.remove(audio_file_path)
                logger.info(f"🧹 Cleaned up temporary file: {audio_file_path}")
            except OSError as e:
                logger.error(f" Error cleaning up temp file {audio_file_path}: {str(e)}")

def process_stt_crew_query(config: ConfigManager, audio_file_path: str, show_context: bool = False, 
                          debug_mode: bool = False, use_hierarchical: bool = True, 
                          store_thoughts: bool = False) -> dict:
    """Process a speech query using transcription and CrewAgentManager with full CrewAI features."""
    try:
        # Transcribe the audio
        transcript = transcribe_audio(config, audio_file_path)
        if not transcript:
            return {
                "error": "Could not transcribe audio or transcription is empty.",
                "response": "Failed to transcribe audio. Please try again.",
                "transcription": ""
            }
            
        print(f"\nTranscribed Query: {transcript}")
        
        # Use the already initialized crew_manager instead of creating a new one
        global crew_manager
        
        print(f"\nProcessing query with CrewAI using hierarchical as {use_hierarchical}...")
        print("=" * 80)
        
        # Process the query and return results
        result = crew_manager.process_query(transcript, show_context)
        
        # Add the transcript to the result
        result["transcription"] = transcript
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing STT CrewAI query: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to process speech query: {str(e)}",
            "response": "Sorry, I encountered an error while processing your speech query.",
            "transcription": ""
        }

# Define allowed origins for CORS
# In production, this should be restricted to specific origins
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")

# Create the FastAPI app
app = FastAPI(
    title="CarbonSense Dashboard",
    description="A dashboard for tracking and analyzing carbon footprint data",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Setup template and static directories
templates_path = Path(__file__).parent / "templates"
static_path = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_path))
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize ConfigManager and set up our engines at startup 
config = ConfigManager()

# Initialize carbon flow as the primary engine
carbon_flow = CarbonSenseFlow(
    config=config,
    debug_mode=True,
    use_cache=False,
    store_thoughts=True
)

# Ensure all logs directories exist
for log_dir in ["logs/thoughts", "logs/visualizations", "logs/communication"]:
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Created log directory: {log_dir}")

# Create the numbered log files if they don't exist
for i in range(1, 15):
    if i == 1:
        suffix = "classification.json"
    elif i == 2:
        suffix = "entities.json"
    elif i == 3:
        suffix = "normalized.json"
    elif i == 4:
        suffix = "cache.json"
    elif i == 5:
        suffix = "milvus.json"
    elif i == 6:
        suffix = "discovery.json"
    elif i == 7:
        suffix = "serper.json"
    elif i == 8:
        suffix = "harmonised.json"
    elif i == 9:
        suffix = "estimation.json"
    elif i == 10:
        suffix = "ranked.json"
    elif i == 11:
        suffix = "comparison.json"
    elif i == 12:
        suffix = "recommendation.json"
    elif i == 13:
        suffix = "explanation.json"
    elif i == 14:
        suffix = "formatted_answer.json"
    
    log_file = f"logs/{i}_{suffix}"
    # Create empty file if it doesn't exist
    if not os.path.exists(log_file):
        try:
            with open(log_file, 'w') as f:
                f.write('')
            logger.info(f"Created log file: {log_file}")
        except Exception as e:
            logger.error(f"Error creating log file {log_file}: {e}")

# Keep crew_manager for backward compatibility
crew_manager = None

# Global thought queue to store agent thoughts during processing
# We'll use a thread-safe queue to handle multiple concurrent requests
thought_queues = {}

# Recording state management class to replace global variables
class RecordingManager:
    def __init__(self):
        self.processes = {}
        self.temp_files = {}
        self.device_indices = {}
    
    def get_process(self, session_id):
        return self.processes.get(session_id)
    
    def set_process(self, session_id, process):
        self.processes[session_id] = process
    
    def get_temp_file(self, session_id):
        return self.temp_files.get(session_id)
    
    def set_temp_file(self, session_id, temp_file):
        self.temp_files[session_id] = temp_file
    
    def get_device_index(self, session_id):
        return self.device_indices.get(session_id)
    
    def set_device_index(self, session_id, device_index):
        self.device_indices[session_id] = device_index
    
    def cleanup_session(self, session_id):
        """Cleanup session resources but don't delete temp file.
        The temp file should be managed by stop_recording and check_processing functions.
        """
        # Stop any running process
        if session_id in self.processes and self.processes[session_id].poll() is None:
            try:
                self.processes[session_id].terminate()
                logger.info(f"Terminated process for session {session_id}")
            except Exception as e:
                logger.warning(f"Error terminating process for session {session_id}: {str(e)}")
        
        # Remove process entry
        self.processes.pop(session_id, None)
        
        # Keep track of the temp file reference but don't delete it
        # The actual file deletion should be handled by stop_recording/check_processing
        # only when processing is complete
        
        # Remove device index entry
        self.device_indices.pop(session_id, None)
        
        logger.info(f"Cleaned up session resources for {session_id}, keeping temp file reference")

# Initialize recording manager
recording_manager = RecordingManager()

# Add after other global variables
TRACKED_DATA_FILE = Path("tracked_queries.json")

# Initialize tracked data structure or load from file if exists
def load_tracked_data():
    if TRACKED_DATA_FILE.exists():
        with open(TRACKED_DATA_FILE, 'r') as f:
            return json.load(f)
    return {
        "Food & Diet": {"count": 0, "total": 0},
        "Energy Use": {"count": 0, "total": 0},
        "Mobility": {"count": 0, "total": 0},
        "Purchases": {"count": 0, "total": 0},
        "Miscellaneous": {"count": 0, "total": 0}
    }

tracked_data = load_tracked_data()

# Sample data for the dashboard (in a real application, this would come from a database)
def get_sample_data():
    # Calculate total carbon from tracked data
    total_carbon = sum(cat["total"] for cat in tracked_data.values())
    
    return {
        "total_carbon": round(total_carbon, 2),
        "food_carbon": round(tracked_data["Food & Diet"]["total"], 2),
        "household_carbon": round(tracked_data["Energy Use"]["total"], 2),
        "transportation_carbon": round(tracked_data["Mobility"]["total"], 2),
        "goods_carbon": round(tracked_data["Purchases"]["total"], 2),
        "misc_carbon": round(tracked_data["Miscellaneous"]["total"], 2),
        "goal_percentage": 30,  # Keep static for now
        "weekly_trend": [4.5, 5.2, 6.0, 6.8, 5.9, 7.1, 8.2],  # Keep static for now
        "badges": ["Eco Starter", "Committed", "Carbon Pro"]  # Keep static for now
    }

# Sample personalized tips (in a real app, these would be generated dynamically based on user data)
def get_personalized_tips():
    return [
        {
            "id": 1,
            "category": "food",
            "title": "Quick Win",
            "description": "Eat one vegetarian meal today to save 1.8 kg CO₂",
            "impact": 1.8,
            "difficulty": "easy"
        },
        {
            "id": 2,
            "category": "household",
            "title": "Energy Saver",
            "description": "Reduce your thermostat by 1°C to save 0.3 kg CO₂ per day",
            "impact": 0.3,
            "difficulty": "easy"
        },
        {
            "id": 3,
            "category": "transportation",
            "title": "Transport Tip",
            "description": "Cycle instead of driving for trips under 2 miles to save 0.8 kg CO₂ per trip",
            "impact": 0.8,
            "difficulty": "medium"
        },
        {
            "id": 4,
            "category": "goods",
            "title": "Shopping Smarter",
            "description": "Buy second-hand clothing instead of new to save 6 kg CO₂ per item",
            "impact": 6.0,
            "difficulty": "medium"
        },
        {
            "id": 5,
            "category": "misc",
            "title": "Digital Cleanup",
            "description": "Clean up your email inbox to reduce server storage and save 0.2 kg CO₂",
            "impact": 0.2,
            "difficulty": "easy"
        }
    ]

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    data = get_sample_data()
    tips = get_personalized_tips()
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "data": data, "tips": tips}
    )

# API endpoint for getting personalized tips
@app.get("/api/tips")
async def get_tips(request: Request, category: Optional[str] = None):
    tips = get_personalized_tips()
    
    # Filter by category if specified
    if category:
        tips = [tip for tip in tips if tip["category"] == category.lower()]
    
    return {
        "tips": tips
    }

# API endpoint for applying a tip
@app.post("/api/tips/apply")
async def apply_tip(request: Request):
    data = await request.json()
    tip_id = data.get("tip_id")
    
    if not tip_id:
        raise HTTPException(status_code=400, detail="Tip ID not provided")
    
    # In a real app, we would record this action in the user's profile
    # and update their carbon footprint accordingly
    
    # Find the tip by ID
    tips = get_personalized_tips()
    selected_tip = next((tip for tip in tips if tip["id"] == tip_id), None)
    
    if not selected_tip:
        raise HTTPException(status_code=404, detail=f"Tip with ID {tip_id} not found")
    
    # Return the impact details
    return {
        "status": "success",
        "message": f"Applied tip: {selected_tip['description']}",
        "impact": {
            "category": selected_tip["category"],
            "reduction": selected_tip["impact"]
        }
    }

@app.post("/api/query")
async def query_carbon(request: Request):
    """
    Process a carbon footprint query and return the result.
    
    This endpoint accepts a query text and optional request_id and processes it using 
    the CarbonSenseFlow engine. It returns the result or error information.
    """
    try:
        # Parse request data
        data = await request.json()
        query = data.get('query', '')
        
        # Use provided request_id if available, otherwise generate a new one
        request_id = data.get('request_id')
        if not request_id:
            request_id = f"query_{int(time.time() * 1000)}"
        
        logger.info(f"Received query request: '{query}' with ID: {request_id}")
        
        # Validate the query
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query cannot be empty", "message": "Please provide a query"}
            )
        
        # Process the query with flow
        result = await process_query_with_flow(query, request_id)
        
        # Ensure the result is properly formatted (already done in process_query_with_flow)
        # This is a double check in case any changes bypass the formatting in process_query_with_flow
        formatted_result = ensure_proper_json_format(result)
        
        # Return the result
        logger.info(f"Successfully processed query with ID: {request_id}")
        return JSONResponse(content={
            "result": formatted_result, 
            "request_id": request_id,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in query_carbon endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing query: {str(e)}",
                "message": "An error occurred while processing your request. Please try again.",
                "status": "error"
            }
        )

# Add after other global variables
agent_step_queues = {}

# Define agent messages for streaming to the frontend
AGENT_MESSAGES = {
    "query_classifier": "Classifying your query to understand your intent and category...",
    "entity_extractor": "Extracting key entities and details from your question...",
    "unit_normalizer": "Normalizing all quantities and units for consistency...",
    "footprint_cache": "Checking the cache for existing carbon footprint data...",
    "milvus_researcher": "Searching the Milvus database for relevant carbon data...",
    "discovery_researcher": "Searching Watson Discovery for scientific carbon data...",
    "serper_researcher": "Searching the web for recent carbon footprint information...",
    "unit_harmoniser": "Harmonizing all carbon metrics into a standard format...",
    "carbon_estimator": "Calculating precise carbon footprints using the best data...",
    "metric_ranker": "Evaluating and ranking all carbon metrics for reliability...",
    "comparison_formatter": "Comparing carbon footprints to highlight differences...",
    "recommendation_agent": "Generating personalized recommendations to reduce your footprint...",
    "explanation_agent": "Providing clear explanations and debunking myths...",
    "answer_formatter": "Formatting the final answer for you...",
    "answer_consolidator": "Consolidating all information into a cohesive response...",
    "manager": "Managing the analysis workflow...",
    "intent_classifier": "Determining the intention behind your query...",
    "entity_recognition": "Identifying key entities in your question...",
    "activity_processor": "Processing your activity details...",
    "footprint_calculator": "Calculating accurate carbon footprints...",
    "recommendations_generator": "Generating tailored recommendations...",
    "data_collector": "Collecting relevant carbon data sources...",
    "done": "Analysis complete! Preparing final response..."
}

# Custom agent callback that emits events for streaming
def agent_callback_with_emit(agent):
    logger.info(f"Agent callback triggered for: {agent}")
    
    # Extract the agent name/role from potentially complex agent objects
    agent_name = "system"  # Default value
    agent_thought = None   # To capture detailed thoughts
    agent_task = None      # To capture task description
    agent_final_answer = None  # To capture final answer
    agent_code_blocks = []  # To capture code blocks
    
    try:
        # Extract agent name based on the type of object
        if isinstance(agent, str):
            agent_name = agent.strip()
        elif hasattr(agent, 'role'):
            agent_name = agent.role.strip()
            # Try to extract detailed thoughts if available
            if hasattr(agent, 'last_output'):
                agent_thought = agent.last_output
        elif isinstance(agent, dict) and 'role' in agent:
            agent_name = agent['role'].strip()
            # Try to get detailed thoughts
            if 'last_output' in agent:
                agent_thought = agent['last_output']
        elif hasattr(agent, '__dict__') and 'role' in agent.__dict__:
            agent_name = agent.__dict__['role'].strip()
            # Try to get detailed thoughts
            if 'last_output' in agent.__dict__:
                agent_thought = agent.__dict__['last_output']
        # Additional check for common agent object structure seen in logs
        elif str(agent).startswith("id=") and "role=" in str(agent):
            # Try to extract role from string representation
            role_part = str(agent).split("role=")[1].split("'")[1]
            if role_part:
                agent_name = role_part.strip()
        
        # Extract detailed content from string representation - works with any agent type
        agent_str = str(agent)
        
        # Extract task description if present
        if "## Task:" in agent_str:
            task_parts = agent_str.split("## Task:")[1].split("\n\n")[0].strip()
            agent_task = f"Task: {task_parts}"
        
        # Extract thought content if present
        if "Thought:" in agent_str:
            thought_parts = agent_str.split("Thought:")[1]
            if "Final Answer:" in thought_parts:
                thought_parts = thought_parts.split("Final Answer:")[0]
            elif "\n\n" in thought_parts:
                thought_parts = thought_parts.split("\n\n")[0]
            agent_thought = f"Thought: {thought_parts.strip()}"
        
        # Extract final answer if present
        if "## Final Answer:" in agent_str:
            answer_parts = agent_str.split("## Final Answer:")[1].strip()
            agent_final_answer = f"Final Answer: {answer_parts}"
        elif "Final Answer:" in agent_str:
            answer_parts = agent_str.split("Final Answer:")[1].strip()
            agent_final_answer = f"Final Answer: {answer_parts}"
            
        # Extract JSON code blocks - commonly found in output
        # Match patterns like ```json ... ``` or just plain JSON objects
        if "```json" in agent_str:
            json_blocks = agent_str.split("```json")
            for block in json_blocks[1:]:  # Skip the first part before any json block
                if "```" in block:
                    json_content = block.split("```")[0].strip()
                    agent_code_blocks.append(f"```json\n{json_content}\n```")
        
        # For any JSON that's not in code blocks but directly in the answer or thought
        if agent_final_answer and "```" not in agent_final_answer:
            # Try to detect JSON in the final answer
            try:
                # Check for common JSON patterns like "{"answer": ..."
                if ("{" in agent_final_answer and "}" in agent_final_answer) or \
                   ("[" in agent_final_answer and "]" in agent_final_answer):
                    # Find the JSON structure from { to }
                    json_start = agent_final_answer.find("{")
                    if json_start >= 0:
                        # Find the balanced closing bracket
                        open_count = 0
                        for i, char in enumerate(agent_final_answer[json_start:]):
                            if char == '{':
                                open_count += 1
                            elif char == '}':
                                open_count -= 1
                                if open_count == 0:
                                    json_content = agent_final_answer[json_start:json_start+i+1]
                                    # Format it nicely if it's valid JSON
                                    try:
                                        parsed = json.loads(json_content)
                                        formatted_json = json.dumps(parsed, indent=2)
                                        agent_code_blocks.append(f"```json\n{formatted_json}\n```")
                                    except json.JSONDecodeError:
                                        # Not valid JSON, just add as is
                                        agent_code_blocks.append(f"```\n{json_content}\n```")
                                    break
            except Exception as e:
                logger.warning(f"Error parsing JSON in answer: {e}")
            
    except Exception as e:
        logger.warning(f"Error extracting agent details: {e}, using basic information", exc_info=True)
    
    # Clean up agent name by removing any newlines and excessive whitespace
    if agent_name and isinstance(agent_name, str):
        agent_name = agent_name.replace('\n', ' ').strip()
        
    logger.info(f"Extracted agent name: {agent_name}")
    
    # Build content with all available parts
    content_parts = [f"Processing with {agent_name}..."]
    
    if agent_task:
        content_parts.append(agent_task)
        logger.info(f"Extracted task: {agent_task[:100]}...")
        
    if agent_thought:
        content_parts.append(agent_thought)
        logger.info(f"Extracted thought: {agent_thought[:100]}...")
        
    if agent_final_answer:
        content_parts.append(agent_final_answer)
        logger.info(f"Extracted final answer: {agent_final_answer[:100]}...")
    
    # Add any code blocks if they're not already in the final answer
    for block in agent_code_blocks:
        if not any(block in part for part in content_parts):
            content_parts.append(block)
            logger.info(f"Added code block: {block[:50]}...")
    
    # Join all parts with newlines
    content = "\n\n".join(content_parts)
    
    # Add thought with the extracted content
    add_thought(
        thought_type="agent_step",
        content=content,
        query_id=CURRENT_QUERY_ID
    )
    
    # Emit the step for event streaming
    emit_agent_step(CURRENT_QUERY_ID, agent_name)

# Update the emit_agent_step function to ensure it correctly handles agent names
def emit_agent_step(request_id: str, agent: str):
    """Emit the current agent step to the frontend."""
    if request_id in agent_step_queues:
        # Ensure agent is not None or empty
        if not agent or not isinstance(agent, str) or agent.strip() == "":
            agent = "system"
            
        agent_message = AGENT_MESSAGES.get(agent, f"Processing step: {agent}")
        
        agent_step_queues[request_id].put({
            "agent": agent,
            "message": agent_message
        })
        
        # Also update the thought status with this agent
        set_thought_status("processing", request_id, agent)

# Add the streaming endpoint
@app.get("/api/stream-agent-step")
async def stream_agent_step(request: Request, query: str, request_id: Optional[str] = None):
    """Stream the current agent step as each task is executed.
    
    This endpoint creates a Server-Sent Events stream that emits agent steps as they occur
    during query processing. The frontend can use this to show real-time progress to the user.
    """
    # Use the provided request_id if available, otherwise generate a new one
    if not request_id:
        request_id = f"query_{int(time.time() * 1000)}"
    
    # Create a queue for this request
    agent_step_queues[request_id] = queue.Queue()
    
    # Log that we're creating a stream for this request ID
    logger.info(f"Starting agent step stream for request: {request_id}")
    
    try:
        async def event_generator():
            try:
                while True:
                    try:
                        # Non-blocking get with timeout
                        data = agent_step_queues[request_id].get(timeout=0.5)
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        # Log the step for debugging
                        logger.info(f"Emitted agent step: {data['agent']} for request: {request_id}")
                        
                        # If this was the final step, break
                        if data["agent"] == "done":
                            logger.info(f"Agent steps complete for request: {request_id}")
                            break
                    except queue.Empty:
                        # Send keep-alive
                        yield ": keep-alive\n\n"
                        await asyncio.sleep(0.5)
                        
                        # Check if request is still active
                        try:
                            is_disconnected = await request.is_disconnected()
                            if is_disconnected:
                                logger.info(f"Client disconnected from SSE stream for request: {request_id}")
                                break
                        except Exception as e:
                            logger.error(f"Error checking connection status: {e}")
                            break
            except Exception as e:
                logger.error(f"Error in event generator for request {request_id}: {e}")
            finally:
                # Cleanup
                if request_id in agent_step_queues:
                    logger.info(f"Cleaning up agent step queue for request: {request_id}")
                    del agent_step_queues[request_id]
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        logger.error(f"Error in stream_agent_step for request {request_id}: {e}")
        if request_id in agent_step_queues:
            del agent_step_queues[request_id]
        raise

# Add a helper function after the other helper functions, around line 407
def ensure_proper_json_format(result):
    """
    Ensures the result is in the expected JSON format for the frontend.
    
    This function cleans the result to remove any markdown code blocks or formatting
    and ensures it contains the expected fields: answer, method, confidence, and category.
    
    Args:
        result: The result from carbon flow processing, could be a dict, string, or formatted string
        
    Returns:
        dict: A properly formatted dictionary ready for JSON serialization
    """
    logger.info(f"Ensuring proper JSON format for result: {type(result)}")
    
    # Initialize default response structure
    formatted_response = {
        "answer": "",
        "method": "No methodology information provided.",
        "confidence": 0.5,
        "category": "Miscellaneous"
    }
    
    try:
        # If result is already a dictionary, use it as base
        if isinstance(result, dict):
            # Extract relevant fields
            if "answer" in result:
                formatted_response["answer"] = str(result["answer"]).strip()
            elif "response" in result:
                formatted_response["answer"] = str(result["response"]).strip()
                
            # Copy other fields if they exist
            for field in ["method", "confidence", "category"]:
                if field in result:
                    formatted_response[field] = result[field]
            
            # Special handling for when the entire response is in a string format inside the result
            if isinstance(formatted_response["answer"], str) and formatted_response["answer"].startswith("```json"):
                json_str = formatted_response["answer"]
                # Remove code block markers
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                try:
                    # Parse the JSON string
                    parsed_json = json.loads(json_str)
                    # Update formatted_response with values from parsed JSON
                    if isinstance(parsed_json, dict):
                        for key in ["answer", "method", "confidence", "category"]:
                            if key in parsed_json:
                                formatted_response[key] = parsed_json[key]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON string in answer: {json_str}")
        
        # If result is a string, try to extract JSON from it if it contains code blocks
        elif isinstance(result, str):
            result_str = result.strip()
            
            # Check if the result contains a JSON code block
            if "```json" in result_str:
                # Extract JSON from code block
                json_block = result_str.split("```json")[1].split("```")[0].strip()
                try:
                    parsed_json = json.loads(json_block)
                    if isinstance(parsed_json, dict):
                        # Update formatted_response with values from parsed JSON
                        for key in ["answer", "method", "confidence", "category"]:
                            if key in parsed_json:
                                formatted_response[key] = parsed_json[key]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON in code block: {json_block}")
                    formatted_response["answer"] = result_str
            else:
                # If it's a simple string, use it as the answer
                formatted_response["answer"] = result_str
        
        # Ensure fields have the correct types
        if "confidence" in formatted_response and not isinstance(formatted_response["confidence"], (int, float)):
            try:
                formatted_response["confidence"] = float(formatted_response["confidence"])
            except (ValueError, TypeError):
                formatted_response["confidence"] = 0.5
        
        # If confidence is out of range, fix it
        if "confidence" in formatted_response:
            if formatted_response["confidence"] > 1.0:
                formatted_response["confidence"] = float(formatted_response["confidence"]) / 100 if formatted_response["confidence"] <= 100 else 1.0
            elif formatted_response["confidence"] < 0:
                formatted_response["confidence"] = 0.0
        
        logger.info(f"Formatted response: {formatted_response}")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error formatting JSON response: {str(e)}", exc_info=True)
        # Return a basic response in case of errors
        return {
            "answer": str(result) if result is not None else "No answer available.",
            "method": "Error in result formatting.",
            "confidence": 0.5,
            "category": "Miscellaneous"
        }

# Update process_query_with_flow to format the response correctly, around line 842
async def process_query_with_flow(query: str, request_id: str):
    """Process a query using the Flow-based approach with event-driven architecture."""
    try:
        # Initialize thoughts file with starting state
        initial_state = {
            "query_id": request_id,
            "thoughts": [],
            "status": "processing",
            "response": "",
            "transcript": query,
            "last_updated": time.time(),
            "current_agent": "system",  # Set a default agent to avoid null
            "error": None
        }
        
        with open(THOUGHTS_FILE_PATH, 'w') as f:
            json.dump(initial_state, f, indent=2)
        
        logger.info(f"Initialized thought state for request {request_id}")
        
        def update_thoughts(agent_name=None, thought_content=None, status=None):
            try:
                with open(THOUGHTS_FILE_PATH, 'r') as f:
                    current_state = json.load(f)
                
                # Only update the agent name if provided and not None
                if agent_name is not None and agent_name.strip() != "":
                    current_state["current_agent"] = agent_name
                
                if thought_content:
                    current_state["thoughts"].append({
                        "type": "agent_step",
                        "content": thought_content,
                        "timestamp": time.time()
                    })
                
                if status:
                    current_state["status"] = status
                
                current_state["last_updated"] = time.time()
                
                with open(THOUGHTS_FILE_PATH, 'w') as f:
                    json.dump(current_state, f, indent=2)
                
                logger.info(f"Updated thoughts for {current_state.get('query_id')}: {thought_content}")
            except Exception as e:
                logger.error(f"Error updating thoughts: {str(e)}")
        
        # Process query with flow
        update_thoughts(thought_content="Starting query processing...")
        
        # Start with the first update
        emit_agent_step(request_id, "query_classifier")
        
        result = await carbon_flow.process_query_with_flow_async(
            query,
            agent_callback=agent_callback_with_emit
        )
        
        # Format the response to ensure proper JSON structure
        formatted_result = ensure_proper_json_format(result)
        
        # Update final state but preserve current agent
        try:
            with open(THOUGHTS_FILE_PATH, 'r') as f:
                current_state = json.load(f)
            current_agent = current_state.get("current_agent")
        except Exception:
            current_agent = None
        
        # Update final state
        final_state = {
            "query_id": request_id,
            "thoughts": [],
            "status": "complete",
            "response": formatted_result,
            "transcript": query,
            "last_updated": time.time(),
            "current_agent": current_agent,  # Preserve the current agent
            "error": None
        }
        
        with open(THOUGHTS_FILE_PATH, 'w') as f:
            json.dump(final_state, f, indent=2)
            
        # Send the 'done' signal to indicate completion
        emit_agent_step(request_id, "done")
        
        logger.info(f"Completed processing for {request_id}")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in process_query_with_flow: {e}")
        
        # Get the current agent if exists
        try:
            with open(THOUGHTS_FILE_PATH, 'r') as f:
                current_state = json.load(f)
            current_agent = current_state.get("current_agent")
        except Exception:
            current_agent = None
        
        # Update error state
        error_state = {
            "query_id": request_id,
            "thoughts": [],
            "status": "error",
            "response": str(e),
            "transcript": query,
            "last_updated": time.time(),
            "current_agent": current_agent,  # Preserve the current agent
            "error": str(e)
        }
        
        with open(THOUGHTS_FILE_PATH, 'w') as f:
            json.dump(error_state, f, indent=2)
            
        # Send the 'done' signal even in case of error
        emit_agent_step(request_id, "done")
        
        return {
            "error": str(e),
            "response": f"An error occurred: {str(e)}"
        }

@app.post("/api/voice-query")
async def process_voice_query(request: Request, audio_data: UploadFile = File(...)):
    """
    Process an audio query using the Speech-to-Text and CarbonSenseFlow functionality.
    
    This endpoint:
    1. Receives audio data from the frontend
    2. Saves it to a temporary file
    3. Transcribes it with IBM Watson STT
    4. Processes it with the CarbonSenseFlow
    5. Returns the results
    """
    temp_file_path = None
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Write the audio data to the temp file
            content = await audio_data.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Log the received audio file
        logger.info(f"Received audio file, saved to {temp_file_path}")
        
        # Transcribe the audio using IBM Watson STT
        transcript = transcribe_audio(config, temp_file_path)
        if not transcript:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Could not transcribe audio or transcription is empty.",
                    "message": "Failed to transcribe audio. Please try again."
                }
            )
            
        logger.info(f"\nTranscribed Query: {transcript}")
        
        # Generate a request ID for tracking
        request_id = f"voice_api_{int(time.time() * 1000)}"
        
        # Setup thought tracking as with text queries
        write_thoughts_to_file([])
        set_thought_status('PROCESSING', request_id)
        
        # Process the query using the same function as text queries
        result = await process_query_with_flow(transcript, request_id)
        
        # Format the result to match expected structure
        formatted_result = ensure_proper_json_format(result)
        
        # Add transcription to the formatted result
        formatted_result["transcription"] = transcript
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            set_thought_status("ERROR", request_id)
            return JSONResponse(
                status_code=400,
                content={"error": result["error"], "message": result.get("response", "An error occurred")}
            )
        
        # Set thought processing as complete
        set_thought_status("COMPLETE", request_id)
        
        # Return the formatted response
        return formatted_result
    except Exception as e:
        logger.error(f"Error processing voice query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Server error", "message": f"Failed to process audio: {str(e)}"}
        )
    finally:
        # Clean up the temporary file if it exists
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"Deleted temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.post("/api/activity")
async def add_activity(request: Request):
    data = await request.json()
    activity = data.get("activity", "")
    
    if not activity:
        raise HTTPException(status_code=400, detail="Activity cannot be empty")
    
    try:
        # Process the activity query using CarbonSenseFlow async method
        result = await carbon_flow.process_query_with_flow_async(f"Record this activity: {activity}")
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            return {
                "status": "warning",
                "message": f"Activity recorded with warning: {result['error']}"
            }
        
        return {
            "status": "success",
            "message": f"Activity '{activity}' recorded successfully.",
            "analysis": result.get("response", str(result)) if isinstance(result, dict) else str(result)
        }
    except Exception as e:
        logger.error(f"Error processing activity: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to record activity: {str(e)}",
            "analysis": None
        }

@app.post("/api/start-recording")
async def start_recording():
    """Start recording audio using the system's microphone."""
    global global_transcript
    session_id = f"session_{int(time.time() * 1000)}"
    
    try:
        # Create a temporary file for the recording result
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        recording_manager.set_temp_file(session_id, temp_file)
        
        # Create an empty file immediately so we know the process has started
        try:
            with open(temp_file, 'w') as f:
                # Write an empty JSON object to indicate initialization
                f.write('{"status": "initializing"}')
            logger.info(f"Created initialization marker file: {temp_file}")
        except Exception as e:
            logger.error(f"Failed to create initialization marker file: {str(e)}")
        
        # Check if we have microphones available
        devices = sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        if not input_devices:
            return JSONResponse(
                status_code=500,
                content={"error": "No microphone found", "message": "No input devices available"}
            )
        
        # Try to find Lenovo Performance Audio microphone (index 4)
        device_index = None
        for idx, device in input_devices:
            if "Lenovo Performance" in device.get('name', '') and idx == 4:
                device_index = idx
                logger.info(f"Selected preferred microphone: Lenovo Performance Audio (index: {idx})")
                break
        
        # If preferred microphone not found, use the first available input device
        if device_index is None:
            device_index = input_devices[0][0]  # Get the index of the first available input device
            logger.info(f"Preferred microphone not found, using: {input_devices[0][1].get('name', '')} (index: {device_index})")
        
        recording_manager.set_device_index(session_id, device_index)
        
        # Store references to the functions needed in the thread to avoid NameError
        _record_audio = record_audio
        _transcribe_audio = transcribe_audio
        _config = config
        _carbon_flow = carbon_flow
        
        # Start recording in a background thread
        def record_and_process():
            try:
                # Record audio directly using the stored reference
                record_duration = 15  # 15 seconds recording
                try:
                    temp_audio_path = _record_audio(record_duration, SAMPLE_RATE, CHANNELS, device_index=device_index)
                    logger.info(f"Recorded audio to {temp_audio_path}, now transcribing and processing")
                except Exception as rec_err:
                    logger.error(f"Error during audio recording: {str(rec_err)}")
                    with open(temp_file, 'w') as f:
                        json.dump({
                            "error": f"Recording failed: {str(rec_err)}",
                            "response": "Failed to record audio. Please check your microphone and try again.",
                            "transcription": "",
                            "confidence": 0.0
                        }, f)
                    return
                
                # Transcribe and process
                try:
                    transcript = _transcribe_audio(_config, temp_audio_path)
                    if not transcript:
                        logger.error("Transcription failed or returned empty result")
                        result = {
                            "error": "Transcription failed",
                            "response": "Could not transcribe the audio. Please try again.",
                            "transcription": "",
                            "confidence": 0.0
                        }
                    else:
                        logger.info(f"Successfully transcribed: '{transcript}'")
                        global_transcript = transcript
                        print("--------------------------------")
                        print("")
                        print(f"Global transcript: {global_transcript}")
                        print("")
                        print("--------------------------------")
                        # write global transcript to file atomically
                        write_transcript_to_file(global_transcript)
                        try:
                            # Generate a request ID similar to text queries
                            request_id = f"voice_{int(time.time() * 1000)}"
                            
                            # Setup thought tracking as in text queries
                            write_thoughts_to_file([])
                            set_thought_status('PROCESSING', request_id)
                            
                            # Process using the same flow as text queries
                            # We need to run it synchronously since we're in a thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                # Use the same processing function as text queries
                                query_result = loop.run_until_complete(process_query_with_flow(transcript, request_id))
                                
                                # Format the result - should already be properly formatted from process_query_with_flow
                                result = query_result
                                if not isinstance(result, dict):
                                    result = {
                                        "response": str(result),
                                        "transcription": transcript,
                                        "confidence": 0.8
                                    }
                                else:
                                    # Ensure we have the transcript
                                    result["transcription"] = transcript
                                    
                                    # Ensure confidence exists
                                    if "confidence" not in result:
                                        result["confidence"] = 0.8
                            finally:
                                loop.close()
                            
                            # Set thought processing as complete
                            set_thought_status("COMPLETE", request_id)
                        except Exception as query_err:
                            logger.error(f"Error processing query: {str(query_err)}")
                            
                            # Create a fallback response that acknowledges the transcript
                            result = {
                                "error": f"Processing error: {str(query_err)}",
                                "response": f"I heard your question about '{transcript}', but I'm having trouble processing it right now. Please try again or rephrase your question.",
                                "transcription": transcript,
                                "confidence": 0.7
                            }
                            
                            # Set error status in thought tracking
                            set_thought_status("ERROR", request_id)
                except Exception as trans_err:
                    logger.error(f"Error during transcription: {str(trans_err)}")
                    result = {
                        "error": f"Transcription error: {str(trans_err)}",
                        "response": "Failed to process audio. Please try again.",
                        "transcription": "",
                        "confidence": 0.0
                    }
                
                # Save result to temp file - wrapped in another try/except to ensure we always save something
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(result, f)
                    logger.info(f"Saved processing result to {temp_file}")
                except Exception as write_err:
                    logger.error(f"Failed to write result to temp file: {str(write_err)}")
                    try:
                        # Last attempt to save at least an error message
                        with open(temp_file, 'w') as f:
                            json.dump({
                                "error": "Failed to save results",
                                "response": "An error occurred while saving results. Please try again.",
                                "transcription": "",
                                "confidence": 0.0
                            }, f)
                    except:
                        logger.error("Critical error: Could not write error message to temp file")
                
            except Exception as e:
                logger.error(f"Error in record_and_process: {str(e)}", exc_info=True)
                # Save error to temp file
                try:
                    with open(temp_file, 'w') as f:
                        json.dump({
                            "error": str(e),
                            "response": f"An error occurred: {str(e)}",
                            "transcription": "",
                            "confidence": 0.0
                        }, f)
                except Exception as write_err:
                    logger.error(f"Failed to write error to temp file: {str(write_err)}")

        # Start recording in a thread with a name and exception handler
        thread = threading.Thread(target=record_and_process, name="AudioRecordingThread", daemon=True)
        # This makes uncaught exceptions in the thread log an error but not crash the app
        thread.setDaemon(True)
        thread.start()
        
        return {
            "status": "recording_started",
            "message": "Started recording audio. Will record for 15 seconds.",
            "session_id": session_id
        }
    
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Server error", "message": f"Failed to start recording: {str(e)}"}
        )

@app.post("/api/stop-recording")
async def stop_recording(request: Request):
    """
    Stop the recording if it's running and get the transcription results.
    """
    try:
        try:
            data = await request.json()
            session_id = data.get("session_id")
        except:
            session_id = None
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing session_id", "message": "Session ID is required"}
            )
        
        # First check if we have results in the temporary file
        temp_file = recording_manager.get_temp_file(session_id)
        if temp_file and os.path.exists(temp_file):
            try:
                logger.info(f"Reading results from {temp_file}")
                try:
                    with open(temp_file, 'r') as f:
                        file_content = f.read().strip()
                        if file_content:
                            results = json.loads(file_content)
                        else:
                            logger.warning(f"Results file is empty, using default empty results")
                            results = {"transcription": "", "response": "No results were recorded.", "confidence": 0.0}
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error reading results: {str(json_err)}")
                    results = {"transcription": "", "response": "Error reading results: Invalid JSON format.", "confidence": 0.0}
                
                # Check if we have complete results with content
                has_complete_results = (
                    "transcription" in results and 
                    "response" in results and 
                    results.get("transcription") and  # Has actual transcription content
                    results.get("response")  # Has actual response content
                )
                
                # Only delete the file if we have complete results
                if has_complete_results:
                    try:
                        os.unlink(temp_file)
                        logger.info(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file: {str(e)}")
                    recording_manager.cleanup_session(session_id)
                else:
                    logger.info(f"Keeping temp file {temp_file} as processing is still ongoing")
                
                # If we have complete results, return them
                if has_complete_results:
                    return {
                        "status": "success",
                        "result": {
                            "transcription": results.get("transcription", ""),
                            "response": results.get("response", ""),
                            "confidence": results.get("confidence", 0.5)
                        }
                    }
            except Exception as e:
                logger.error(f"Error reading results from temp file: {str(e)}", exc_info=True)
        
        # Next check thought tracking, which is used by process_query_with_flow
        try:
            thoughts_data = read_thoughts_from_file()
            if thoughts_data:
                status = thoughts_data.get("status", "").upper()
                
                # Get the transcript either from global variable or from file
                transcript = global_transcript
                if not transcript:
                    try:
                        transcript_path = os.path.join(project_root, "logs/transcript.txt")
                        if os.path.exists(transcript_path):
                            with open(transcript_path, 'r') as f:
                                transcript = f.read().strip()
                    except Exception as e:
                        logger.error(f"Error reading transcript file: {str(e)}")
                
                # If processing is complete, return the final thought as the response
                if status == "COMPLETE":
                    # Find the most recent completion thought
                    completion_thoughts = [t for t in thoughts_data.get("thoughts", []) if t.get("type") == "completion"]
                    if completion_thoughts:
                        # Sort by timestamp and get the most recent
                        latest_completion = sorted(completion_thoughts, key=lambda x: x.get("timestamp", 0))[-1]
                        return {
                            "status": "success",
                            "result": {
                                "transcription": transcript,
                                "response": latest_completion.get("content", "Processing completed."),
                                "confidence": 0.8
                            }
                        }
                
                # If there was an error, return it
                elif status == "ERROR":
                    # Find the most recent error thought
                    error_thoughts = [t for t in thoughts_data.get("thoughts", []) if t.get("type") == "error"]
                    if error_thoughts:
                        # Sort by timestamp and get the most recent
                        latest_error = sorted(error_thoughts, key=lambda x: x.get("timestamp", 0))[-1]
                        return {
                            "status": "error",
                            "result": {
                                "transcription": transcript,
                                "response": latest_error.get("content", "An error occurred."),
                                "confidence": 0.5
                            }
                        }
                
                # If we have a transcript but processing is ongoing, return as partial result
                elif transcript:
                    # Get the most recent thought to show progress
                    if thoughts_data.get("thoughts"):
                        thoughts = sorted(thoughts_data["thoughts"], key=lambda x: x.get("timestamp", 0))
                        latest_thought = thoughts[-1]
                        return {
                            "status": "processing",
                            "result": {
                                "transcription": transcript,
                                "response": f"Processing query: {transcript}\nCurrent status: {latest_thought.get('content', 'Processing...')}",
                                "confidence": 0.5
                            }
                        }
        except Exception as e:
            logger.error(f"Error checking thoughts tracking: {str(e)}")
        
        # If no results yet, return no_results status
        return {
            "status": "no_results",
            "message": "No transcription results available. Try recording again."
        }
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Server error", "message": f"Failed to stop recording: {str(e)}"}
        )

@app.post("/api/stop-recording/{session_id}")
async def stop_recording_by_path(request: Request, session_id: str):
    """
    Alternative endpoint that accepts session_id as a path parameter.
    """
    logger.info(f"Stopping recording via path parameter for session: {session_id}")
    
    # First check if we have results in the temporary file
    temp_file = recording_manager.get_temp_file(session_id)
    if temp_file and os.path.exists(temp_file):
        try:
            logger.info(f"Reading results from {temp_file}")
            try:
                with open(temp_file, 'r') as f:
                    file_content = f.read().strip()
                    if file_content:
                        results = json.loads(file_content)
                    else:
                        logger.warning(f"Results file is empty, using default empty results")
                        results = {"transcription": "", "response": "No results were recorded.", "confidence": 0.0}
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error reading results: {str(json_err)}")
                results = {"transcription": "", "response": "Error reading results: Invalid JSON format.", "confidence": 0.0}
                
                # Check if we have complete results with content
                has_complete_results = (
                    "transcription" in results and 
                    "response" in results and 
                    results.get("transcription") and  # Has actual transcription content
                    results.get("response")  # Has actual response content
                )
                
                # Only delete the file if we have complete results
                if has_complete_results:
                    try:
                        os.unlink(temp_file)
                        logger.info(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file: {str(e)}")
                    recording_manager.cleanup_session(session_id)
                else:
                    logger.info(f"Keeping temp file {temp_file} as processing is still ongoing")
                
                return {
                    "status": "success",
                    "result": {
                        "transcription": results.get("transcription", ""),
                        "response": results.get("response", ""),
                        "confidence": results.get("confidence", 0.5)
                    }
                }
        except Exception as e:
            logger.error(f"Error reading results: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to read transcription results: {str(e)}"
                }
    return {
        "status": "no_results",
        "message": "No transcription results available. Try recording again."
    }

@app.post("/api/track-query")
async def track_query(request: Request):
    data = await request.json()
    query = data.get("query")
    category = data.get("category", "Miscellaneous")
    carbon_value = float(data.get("carbon_value", 0))
    
    logger.info(f"Tracking query: '{query}' in category: {category} with value: {carbon_value}")
    
    # Validate carbon value
    if carbon_value <= 0:
        logger.warning(f"Invalid carbon value: {carbon_value}")
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid carbon value", "message": "Carbon value must be greater than 0"}
        )
    
    # Update tracked data
    if category in tracked_data:
        tracked_data[category]["count"] += 1
        tracked_data[category]["total"] += carbon_value
    else:
        tracked_data[category] = {"count": 1, "total": carbon_value}
    
    # Save to file
    with open(TRACKED_DATA_FILE, 'w') as f:
        json.dump(tracked_data, f, indent=2)
    
    logger.info(f"Updated tracked data: {tracked_data}")
    
    # Calculate updated dashboard data
    total_carbon = sum(cat["total"] for cat in tracked_data.values())
    
    # Format the response data to match the expected structure
    updated_data = {
        "total_carbon": round(total_carbon, 2),
        "food_carbon": round(tracked_data.get("Food & Diet", {"total": 0})["total"], 2),
        "household_carbon": round(tracked_data.get("Energy Use", {"total": 0})["total"], 2),
        "transportation_carbon": round(tracked_data.get("Mobility", {"total": 0})["total"], 2),
        "goods_carbon": round(tracked_data.get("Purchases", {"total": 0})["total"], 2),
        "misc_carbon": round(tracked_data.get("Miscellaneous", {"total": 0})["total"], 2)
    }
    
    logger.info(f"Returning updated dashboard data: {updated_data}")
    
    return JSONResponse(content=updated_data)

@app.get("/api/check-processing/{session_id}")
async def check_processing(session_id: str):
    """
    Check if the audio processing has completed for a session.
    Returns status and results if available.
    """
    global global_transcript

    _transcript = global_transcript

    # If global transcript is empty, try to read from file
    if not _transcript:
        try:
            transcript_path = os.path.join(project_root, "logs/transcript.txt")
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    _transcript = f.read().strip()
                    if _transcript:
                        logger.info(f"Read transcript from file: '{_transcript}'")
        except Exception as e:
            logger.error(f"Error reading transcript file: {str(e)}")

    # First check if we have a temp file result from the recording process
    temp_file = recording_manager.get_temp_file(session_id)
    if temp_file and os.path.exists(temp_file):
        try:
            with open(temp_file, 'r') as f:
                content = f.read().strip()
                if content:
                    try:
                        result = json.loads(content)
                        # Format the result properly
                        formatted_result = ensure_proper_json_format(result)
                        # Add transcription if available
                        if "transcription" in result:
                            formatted_result["transcription"] = result["transcription"]
                        elif _transcript:
                            formatted_result["transcription"] = _transcript
                            
                        # Return the result directly if we have valid content
                        if formatted_result["answer"]:
                            logger.info(f"Found complete result in temp file for session {session_id}")
                            return {
                                "status": "complete",
                                "result": formatted_result
                            }
                    except json.JSONDecodeError:
                        logger.warning(f"Couldn't parse JSON in temp file: {temp_file}")
        except Exception as e:
            logger.error(f"Error reading temp file: {str(e)}")
    
    # Next check thought tracking which is used for process_query_with_flow
    try:
        thoughts_data = read_thoughts_from_file()
        if thoughts_data:
            status = thoughts_data.get("status", "").upper()
            
            # If processing is complete, return the final thought as the response
            if status == "COMPLETE":
                # Find the most recent completion thought
                completion_thoughts = [t for t in thoughts_data.get("thoughts", []) if t.get("type") == "completion"]
                if completion_thoughts:
                    # Sort by timestamp and get the most recent
                    latest_completion = sorted(completion_thoughts, key=lambda x: x.get("timestamp", 0))[-1]
                    # Create a formatted response
                    formatted_result = ensure_proper_json_format(latest_completion.get("content", "Processing completed."))
                    formatted_result["transcription"] = _transcript
                    return {
                        "status": "complete",
                        "result": formatted_result
                    }
            
            # If there was an error, return it
            elif status == "ERROR":
                # Find the most recent error thought
                error_thoughts = [t for t in thoughts_data.get("thoughts", []) if t.get("type") == "error"]
                if error_thoughts:
                    # Sort by timestamp and get the most recent
                    latest_error = sorted(error_thoughts, key=lambda x: x.get("timestamp", 0))[-1]
                    error_content = latest_error.get("content", "An error occurred.")
                    # Create a formatted response for the error
                    formatted_result = ensure_proper_json_format({
                        "answer": f"Error: {error_content}",
                        "method": "Processing encountered an error.",
                        "confidence": 0.1,
                        "category": "Error"
                    })
                    formatted_result["transcription"] = _transcript
                    formatted_result["error"] = error_content
                    return {
                        "status": "error",
                        "result": formatted_result
                    }
            
            # If we have a transcript but processing is ongoing, return as partial result
            elif _transcript:
                # Get the most recent thought to show progress
                if thoughts_data.get("thoughts"):
                    thoughts = sorted(thoughts_data["thoughts"], key=lambda x: x.get("timestamp", 0))
                    latest_thought = thoughts[-1]
                    # Create a formatted progress response
                    progress_result = {
                        "transcription": _transcript,
                        "answer": f"Processing query: {_transcript}",
                        "method": f"Current status: {latest_thought.get('content', 'Processing...')}",
                        "confidence": 0.5,
                        "category": "Processing"
                    }
                    return {
                        "status": "processing",
                        "result": progress_result
                    }
    except Exception as e:
        logger.error(f"Error checking thoughts tracking: {str(e)}")
    
    # Finally, if we just have a transcript but no other information, return it
    print("--------------------------------")
    print()
    print("frontend check-processing")
    print(f"Transcript: {_transcript}")
    print()
    print("--------------------------------")

    if _transcript:
        # Create a formatted response for just transcription
        basic_result = ensure_proper_json_format({
            "answer": f"Processing query: {_transcript}",
            "method": "Transcription received, awaiting processing.",
            "confidence": 0.5,
            "category": "Pending"
        })
        basic_result["transcription"] = _transcript
        return {
            "status": "transcribed",
            "result": basic_result
        }
    else:
        return {
            "status": "pending",
            "message": "No transcript available yet"
        }

@app.get("/api/check-transcript-file")
async def check_transcript_file():
    """
    Check if the transcript file exists and return its content if available.
    """
    global global_transcript
    logger.info("Checking transcript file")
    try:
        # Use absolute path for transcript file
        transcript_path = os.path.join(project_root, "logs/transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
                if transcript:
                    logger.info(f"Found transcript: '{transcript}'")
                    # Update global transcript to ensure consistency
                    global_transcript = transcript
                    return {
                        "status": "success",
                        "transcript": transcript
                    }
                else:
                    logger.warning("Transcript file exists but is empty")
                    return {
                        "status": "pending",
                        "message": "Transcript file exists but is empty"
                    }
        else:
            logger.warning("Transcript file does not exist yet")
            return {
                "status": "pending",
                "message": "Transcript file does not exist yet"
            }
    except Exception as e:
        logger.error(f"Error checking transcript file: {str(e)}")
        return {
            "status": "error",
            "message": f"Error checking transcript file: {str(e)}"
        }

# Add this function to clean up transcript file
def cleanup_transcript_file():
    """Clean up the transcript file if it exists."""
    transcript_path = os.path.join(project_root, "logs/transcript.txt")
    temp_path = transcript_path + ".tmp"
    
    try:
        # First try to remove the main transcript file
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
            logger.info(f"Cleaned up transcript file: {transcript_path}")
    except Exception as e:
        logger.error(f"Error cleaning up transcript file: {str(e)}")
    
    try:
        # Also try to clean up any temporary transcript file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary transcript file: {temp_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary transcript file: {str(e)}")
    
    # Reset the global transcript variable
    global global_transcript
    global_transcript = ""

# In the checkForResults function in the frontend, after displaying the response,
# add a call to an endpoint to clean up the transcript file
@app.post("/api/cleanup-transcript")
async def cleanup_transcript():
    """Clean up the transcript file."""
    cleanup_transcript_file()
    return {"status": "success"}

@app.post("/api/reset-transcript")
async def reset_transcript():
    """Reset the global transcript variable."""
    global global_transcript
    global_transcript = ""
    logger.info("Reset global transcript variable")
    return {"status": "success"}

# Add this function to write transcript to file atomically
def write_transcript_to_file(transcript_text):
    """Write transcript to file in an atomic operation."""
    try:
        transcript_path = os.path.join(project_root, "logs/transcript.txt")
        temp_path = transcript_path + ".tmp"
        
        # Write to temporary file first
        with open(temp_path, "w") as f:
            f.write(transcript_text)
        
        # Then rename to final location (atomic operation)
        os.replace(temp_path, transcript_path)
        logger.info(f"Successfully wrote transcript to file: {transcript_path}")
    except Exception as e:
        logger.error(f"Error writing transcript to file: {str(e)}")

# Add function to reset the CrewManager
def reset_crew_manager():
    """Reset the global crew manager instance."""
    global crew_manager
    try:
        logger.info("Attempting to reset CrewAgentManager")
        # Re-initialize the crew manager
        crew_manager = CrewAgentManager(
            config=config,
            debug_mode=True,
            use_cache=False,
            use_hierarchical=True,
            store_thoughts=True
        )
        logger.info("Successfully reset CrewAgentManager")
        return True
    except Exception as e:
        logger.error(f"Failed to reset CrewAgentManager: {str(e)}")
        return False

def reset_carbon_flow():
    """Reset the global carbon flow instance."""
    global carbon_flow
    try:
        # Reinitialize the carbon flow with original settings
        carbon_flow = CarbonSenseFlow(
            config=config,
            debug_mode=True,
            use_cache=False,
            store_thoughts=True
        )
        logger.info("Carbon flow has been reset")
        return True
    except Exception as e:
        logger.error(f"Error resetting Carbon Flow: {str(e)}")
        return False
    
@app.post("/api/reset")
async def reset_endpoint():
    """Reset the carbon flow instance."""
    try:
        success = reset_carbon_flow()
        return {"status": "success" if success else "error", 
                "message": "CarbonSenseFlow has been reset" if success else "Error resetting CarbonSenseFlow"}
    except Exception as e:
        return {"status": "error", "message": f"Error resetting CarbonSenseFlow: {str(e)}"}

# Add a function to process with fallback to non-hierarchical mode
def process_query_with_fallback(crew_manager, transcript, show_context=False):
    """Process a query with fallback to non-hierarchical mode if hierarchical fails."""
    try:
        # First try with current settings
        result = crew_manager.process_query(transcript, show_context)
        return result, False  # No error
    except Exception as e:
        logger.error(f"Error in hierarchical processing: {str(e)}")
        
        # If it's the manager tools error, try non-hierarchical mode
        if "Manager agent should not have tools" in str(e):
            logger.info("Falling back to non-hierarchical processing")
            try:
                # Set use_hierarchical to False temporarily
                orig_hierarchical = crew_manager.use_hierarchical
                crew_manager.use_hierarchical = False
                
                # Process the query with non-hierarchical mode
                try:
                    result = crew_manager.process_query(transcript, show_context)
                    return result, True  # Processed with fallback
                finally:
                    # Restore the original setting
                    crew_manager.use_hierarchical = orig_hierarchical
            except Exception as fallback_err:
                logger.error(f"Fallback processing also failed: {str(fallback_err)}")
                raise fallback_err
        else:
            # For other errors, re-raise
            raise e

# Add this function to write thoughts to file atomically
def write_thoughts_to_file(thoughts_data):
    """Write thoughts to file in an atomic operation."""
    try:
        # Ensure the thoughts_data has all required fields
        if isinstance(thoughts_data, dict):
            # Add default fields if missing
            thoughts_data.setdefault("query_id", None)
            thoughts_data.setdefault("thoughts", [])
            thoughts_data.setdefault("status", "idle")
            thoughts_data.setdefault("response", "")
            thoughts_data.setdefault("transcript", "")
            thoughts_data.setdefault("last_updated", time.time())
            
            # Always ensure current_agent exists and is not None
            if "current_agent" not in thoughts_data or thoughts_data["current_agent"] is None:
                thoughts_data["current_agent"] = "system"
                
            thoughts_data.setdefault("error", None)
        else:
            # If not a dict, create a default structure
            thoughts_data = {
                "query_id": None,
                "thoughts": thoughts_data if isinstance(thoughts_data, list) else [],
                "status": "idle",
                "response": "",
                "transcript": "",
                "last_updated": time.time(),
                "current_agent": "system",
                "error": None
            }
            
        temp_path = THOUGHTS_FILE_PATH + ".tmp"
        
        # Write to temporary file first
        with open(temp_path, "w") as f:
            json.dump(thoughts_data, f, indent=2)
        
        # Then rename to final location (atomic operation)
        os.replace(temp_path, THOUGHTS_FILE_PATH)
        logger.info(f"Successfully wrote thoughts to file: {THOUGHTS_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error writing thoughts to file: {str(e)}")
        return False

# Add this function to read thoughts from file
def read_thoughts_from_file():
    """Read thoughts from file."""
    try:
        if os.path.exists(THOUGHTS_FILE_PATH):
            try:
                with open(THOUGHTS_FILE_PATH, 'r') as f:
                    thoughts_data = json.load(f)
                    
                    # Ensure all required fields exist
                    if "current_agent" not in thoughts_data or thoughts_data["current_agent"] is None:
                        thoughts_data["current_agent"] = "system"
                    
                    logger.info(f"Successfully read thoughts from file")
                    return thoughts_data
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in thoughts file - creating new empty file")
                # Create a fresh thoughts file
                default_thoughts = {
                    "query_id": None,
                    "thoughts": [],
                    "status": "idle",
                    "response": "",
                    "transcript": "",
                    "last_updated": time.time(),
                    "current_agent": "system",
                    "error": None
                }
                write_thoughts_to_file(default_thoughts)
                return default_thoughts
        else:
            logger.warning(f"Thoughts file does not exist: {THOUGHTS_FILE_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error reading thoughts from file: {str(e)}")
        return None

# Add this function to clean up thoughts file
def cleanup_thoughts_file():
    """Reset the thoughts file state to idle without deleting the file."""
    try:
        global current_thought_state
        
        # Read existing thoughts data to preserve history
        existing_data = None
        if os.path.exists(THOUGHTS_FILE_PATH):
            try:
                with open(THOUGHTS_FILE_PATH, 'r') as f:
                    existing_data = json.load(f)
                    logger.info(f"Read existing thoughts from file: {THOUGHTS_FILE_PATH}")
            except Exception as e:
                logger.error(f"Error reading existing thoughts: {str(e)}")
                existing_data = None
        
        # Reset the current state
        current_thought_state = {
            "query_id": None,
            "thoughts": existing_data.get("thoughts", []) if existing_data else [],
            "status": "idle"
        }
        
        # Write updated state to file
        write_thoughts_to_file(current_thought_state)
        
        logger.info(f"Reset thoughts file state: {THOUGHTS_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error resetting thoughts file: {str(e)}")
        return False

# Update function to add thought to the state and file
def add_thought(thought_type, content, query_id=None):
    """Add a thought to the current state and write to file.
    
    Args:
        thought_type (str): Type of thought (e.g., "agent_step", "agent_detail", "thinking", "completion", "error")
        content (str): The content of the thought
        query_id (str, optional): The query ID to associate with this thought
    """
    global current_thought_state
    
    try:
        # Log detailed debugging info
        logger.info(f"Adding thought - Type: {thought_type}, Content length: {len(content)}, Query ID: {query_id}")
        
        # Create thought object
        timestamp = time.time()
        
        # Check if this is a detailed agent output that should be preserved as is
        preserve_formatting = thought_type == "agent_detail" or "```" in content
        
        thought = {
            "type": thought_type,
            "content": content,
            "timestamp": timestamp,
            "preserve_formatting": preserve_formatting
        }
        
        # If query_id is provided, update the current state
        if query_id is not None:
            if current_thought_state["query_id"] != query_id:
                # First, read any existing thoughts to preserve history
                existing_thoughts = []
                if os.path.exists(THOUGHTS_FILE_PATH):
                    try:
                        with open(THOUGHTS_FILE_PATH, 'r') as f:
                            existing_data = json.load(f)
                            existing_thoughts = existing_data.get("thoughts", [])
                    except Exception as e:
                        logger.error(f"Error reading existing thoughts: {str(e)}")
                
                # New query with preserved history
                current_thought_state = {
                    "query_id": query_id,
                    "thoughts": existing_thoughts + [thought],
                    "status": "processing"
                }
                logger.info(f"Started new thought process for query ID: {query_id} with {len(existing_thoughts)} existing thoughts")
            else:
                # Add to existing query
                current_thought_state["thoughts"].append(thought)
                logger.info(f"Added thought to existing process for query ID: {query_id}")
        else:
            # Just add to current state without changing query_id
            current_thought_state["thoughts"].append(thought)
            logger.info(f"Added thought to current process without query ID")
        
        # Write updated state to file
        success = write_thoughts_to_file(current_thought_state)
        
        # Log the result
        if success:
            logger.info(f"Successfully wrote thought to file")
        else:
            logger.error(f"Failed to write thought to file")
        
        return success
    except Exception as e:
        logger.error(f"Error adding thought: {str(e)}")
        return False

# Add a function to set the thought process status
def set_thought_status(status, query_id=None, agent_name=None):
    """Set the status of the thought process."""
    global current_thought_state
    
    try:
        # Load current state from file first
        current_state = read_thoughts_from_file() or current_thought_state
        
        # Update query_id if provided
        if query_id is not None:
            current_state["query_id"] = query_id
        
        # Update status
        current_state["status"] = status
        
        # Update agent if provided and not None
        if agent_name is not None:
            current_state["current_agent"] = agent_name
        elif "current_agent" not in current_state or current_state["current_agent"] is None:
            current_state["current_agent"] = "system"  # Ensure a default value
        
        # Update timestamp
        current_state["last_updated"] = time.time()
        
        # Update the global state
        current_thought_state = current_state
        
        logger.info(f"Updated thought process status to: {status} for query ID: {current_state['query_id']}")
        
        # Write updated state to file
        success = write_thoughts_to_file(current_state)
        
        return success
    except Exception as e:
        logger.error(f"Error setting thought status: {str(e)}")
        return False

# Add an API endpoint to check for thoughts
@app.get("/api/check-thoughts")
async def check_thoughts():
    """Check for thoughts from the file."""
    try:
        if not os.path.exists(THOUGHTS_FILE_PATH):
            return {
                "query_id": None,
                "thoughts": [],
                "status": "idle",
                "response": "",
                "transcript": "",
                "last_updated": time.time(),
                "current_agent": None,
                "error": None
            }
            
        with open(THOUGHTS_FILE_PATH, 'r') as f:
            current_state = json.load(f)
            
        # Ensure all required fields exist
        current_state.setdefault("query_id", None)
        current_state.setdefault("thoughts", [])
        current_state.setdefault("status", "idle")
        current_state.setdefault("response", "")
        current_state.setdefault("transcript", "")
        current_state.setdefault("last_updated", time.time())
        current_state.setdefault("current_agent", None)
        current_state.setdefault("error", None)
        
        logger.info(f"Returning thoughts data - Status: {current_state['status']}, Thought count: {len(current_state['thoughts'])}")
        return current_state
        
    except Exception as e:
        logger.error(f"Error checking thoughts: {str(e)}")
        return {
            "query_id": None,
            "thoughts": [],
            "status": "error",
            "response": "",
            "transcript": "",
            "last_updated": time.time(),
            "current_agent": None,
            "error": str(e)
        }

# Add an API endpoint to clean up thoughts
@app.post("/api/cleanup-thoughts")
async def cleanup_thoughts():
    """Clean up the thoughts file."""
    success = cleanup_thoughts_file()
    return {"status": "success" if success else "error"}

# Add this function at the module level
def process_query_sync(query, request_id=None):
    """Process a query synchronously, for use in threads."""
    try:
        # Generate a request ID if none was provided
        if request_id is None:
            request_id = f"sync_{int(time.time() * 1000)}"
            
        # Setup thought tracking
        add_thought("start", f"Processing query: {query}", request_id)
        add_thought("thinking", "Using CarbonSenseFlow to process your query with event-driven architecture...", request_id)
        set_thought_status("processing", request_id)
        
        # Add intermediate thoughts to mimic the flow's steps (matching process_query_with_flow)
        add_thought("thinking", "Analyzing query intent and extracting entities...", request_id)
        add_thought("thinking", "Researching carbon footprint data from multiple sources...", request_id)
        add_thought("thinking", "Harmonizing data and calculating carbon estimates...", request_id)
        
        # Use the global carbon_flow instance
        global carbon_flow
        result = carbon_flow.process_query_with_flow(query)
        
        # Format result to ensure proper JSON structure
        formatted_result = ensure_proper_json_format(result)
            
        # Add completion thought with result
        add_thought("completion", formatted_result, request_id)
        
        # Update status
        set_thought_status("COMPLETE", request_id)
        
        return formatted_result
    except Exception as e:
        logger.error(f"Error in process_query_sync: {str(e)}")
        
        # Add error thought
        add_thought("error", f"An error occurred while processing your query: {str(e)}", request_id)
        set_thought_status("ERROR", request_id)
        
        return {
            "error": str(e),
            "response": f"An error occurred: {str(e)}"
        }

# Add after the RecordingManager class

class CommunicationStateManager:
    def __init__(self):
        self.state_file = os.path.join(project_root, "logs", "communication", "state.json")
        self.ensure_state_file()
        
    def ensure_state_file(self):
        """Ensure the state file and directory exist."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        if not os.path.exists(self.state_file):
            self.write_state({
                "status": "idle",
                "query_id": None,
                "transcript": "",
                "response": "",
                "thoughts": [],
                "last_updated": time.time()
            })
    
    def read_state(self):
        """Read the current state from file."""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading state file: {str(e)}")
            return None
            
    def write_state(self, state):
        """Write state to file atomically."""
        temp_file = f"{self.state_file}.tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(temp_file, self.state_file)
            return True
        except Exception as e:
            logger.error(f"Error writing state file: {str(e)}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False
            
    def update_state(self, **kwargs):
        """Update specific fields in the state."""
        current_state = self.read_state() or {}
        current_state.update(kwargs)
        current_state["last_updated"] = time.time()
        return self.write_state(current_state)
        
    def reset_state(self):
        """Reset the state to idle."""
        return self.write_state({
            "status": "idle",
            "query_id": None,
            "transcript": "",
            "response": "",
            "thoughts": [],
            "last_updated": time.time()
        })

# Initialize the communication state manager
comm_manager = CommunicationStateManager()

@app.post("/api/cleanup-state")
async def cleanup_state():
    """Reset the communication state."""
    try:
        success = comm_manager.reset_state()
        return {"status": "success" if success else "error"}
    except Exception as e:
        logger.error(f"Error cleaning up state: {str(e)}")
        return {"status": "error", "message": str(e)}

# Function to capture detailed agent output from logs
def capture_agent_output(agent_type, output_text, query_id=None):
    """
    Capture detailed agent output from logs and add it to the thoughts file.
    
    Args:
        agent_type (str): The type/role of the agent (e.g., "Carbon Metric Standardization Specialist")
        output_text (str): The full output text from the agent
        query_id (str, optional): The query ID to associate with this thought
    """
    logger.info(f"Capturing detailed output for agent: {agent_type}")
    
    try:
        # Extract useful sections from the output text
        sections = []
        
        # Add agent header
        sections.append(f"# Agent: {agent_type}")
        
        # Extract task if present
        if "## Task:" in output_text:
            task_parts = output_text.split("## Task:")[1].split("## Final Answer:" if "## Final Answer:" in output_text else "\n\n# Agent:")[0].strip()
            sections.append(f"## Task:{task_parts}")
        
        # Extract thought process if present
        if "Thought:" in output_text:
            thought_parts = output_text.split("Thought:")[1].split("Final Answer:" if "Final Answer:" in output_text else "\n\n")[0].strip()
            sections.append(f"Thought: {thought_parts}")
        
        # Extract final answer if present
        if "## Final Answer:" in output_text:
            answer_parts = output_text.split("## Final Answer:")[1].split("\n\n# Agent:" if "\n\n# Agent:" in output_text else "")[0].strip()
            sections.append(f"## Final Answer:\n{answer_parts}")
        elif "Final Answer:" in output_text:
            answer_parts = output_text.split("Final Answer:")[1].split("\n\n# Agent:" if "\n\n# Agent:" in output_text else "")[0].strip()
            sections.append(f"Final Answer:\n{answer_parts}")
        
        # Combine all sections
        detailed_output = "\n\n".join(sections)
        
        # Only add if we have substantial content
        if len(detailed_output) > 50:  # More than just the agent header
            # Add to thoughts
            add_thought(
                thought_type="agent_detail",
                content=detailed_output,
                query_id=query_id or CURRENT_QUERY_ID
            )
            logger.info(f"Added detailed agent output to thoughts ({len(detailed_output)} chars)")
            return True
        else:
            logger.info("Not enough detailed content to add to thoughts")
            return False
    except Exception as e:
        logger.error(f"Error capturing agent output: {e}", exc_info=True)
        return False

# Run the app using uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 