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
    debug_mode=False,
    use_cache=False,
    store_thoughts=True
)

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

# API endpoint for streaming agent thoughts
@app.get("/api/stream-thoughts")
async def stream_thoughts(request: Request, query: str):
    """Stream agent thoughts as Server-Sent Events."""
    # Generate a unique ID for this streaming connection
    request_id = f"request_{int(time.time() * 1000)}"
    
    logger.info(f"========== THOUGHT STREAM REQUEST ==========")
    logger.info(f"Request ID: {request_id}")
    logger.info(f"Query: {query}")
    logger.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    logger.info(f"============================================")
    
    # Create a dedicated queue for this request
    thought_queues[request_id] = queue.Queue()
    
    # Start thought simulation in a background thread
    thread = threading.Thread(
        target=simulate_agent_thoughts,
        args=(request_id, query),
        daemon=True
    )
    thread.start()
    
    # Define the SSE generator
    async def event_generator():
        try:
            logger.info(f"Starting event generator for request: {request_id}, query: {query}")
            
            # Send retry interval and initial connection message
            yield "retry: 1000\n\n"
            yield f"id: {request_id}\n"
            yield f"data: {json.dumps({'type': 'info', 'content': 'Connection established'})}\n\n"
            logger.info(f"[{request_id}] Sent initial connection message")
            
            # Wait a moment to ensure the client has connected
            await asyncio.sleep(0.2)
            
            # Send a test thought to verify streaming is working
            test_thought = {'type': 'thought', 'content': f'Initializing analysis for: {query}'}
            yield f"id: {int(time.time() * 1000)}\n"
            yield f"data: {json.dumps(test_thought)}\n\n"
            logger.info(f"[{request_id}] Sent initial thought: {test_thought}")
            
            counter = 0
            disconnect_counter = 0
            max_counter = 60  # 30 seconds max
            max_disconnect_attempts = 3
            
            while counter < max_counter:
                counter += 1
                try:
                    # Non-blocking get with timeout
                    thought_data = thought_queues[request_id].get(timeout=0.5)
                    
                    # Reset disconnect counter on successful data retrieval
                    disconnect_counter = 0
                    
                    # Handle "DONE" signal
                    if thought_data == "DONE":
                        logger.info(f"[{request_id}] Received DONE signal, completing stream")
                        yield f"id: {int(time.time() * 1000)}\n"
                        yield f"data: {json.dumps({'type': 'complete', 'content': 'Processing completed'})}\n\n"
                        break
                    
                    # Send the thought data as a server-sent event
                    # Include event ID, event type, and proper formatting with double newlines
                    event_id = int(time.time() * 1000)
                    yield f"id: {event_id}\n"
                    yield f"data: {json.dumps(thought_data)}\n\n"
                    logger.info(f"[{request_id}] Sent event: {thought_data}")
                    
                except queue.Empty:
                    # No messages available, send a keep-alive comment
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(0.5)
                    
                    # If client has disconnected, the keep-alive won't be sent successfully
                    # Check if request is still active
                    try:
                        # If client disconnected, this will raise an exception 
                        await request.is_disconnected()
                        disconnect_counter += 1
                        
                        if disconnect_counter >= max_disconnect_attempts:
                            logger.warning(f"[{request_id}] Client appears to be disconnected after {disconnect_counter} attempts")
                            break
                            
                    except:
                        # Request is still connected if is_disconnected throws an exception
                        pass
                    
                    continue
                except KeyError:
                    # Queue has been removed, likely due to client disconnect
                    logger.warning(f"[{request_id}] Queue no longer exists, client likely disconnected")
                    break
                except Exception as e:
                    logger.error(f"[{request_id}] Error in event stream: {str(e)}")
                    yield f"id: {int(time.time() * 1000)}\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    break
        except Exception as e:
            logger.error(f"Error in event generator: {e}")
        finally:
            # Clean up the queue when done
            if request_id in thought_queues:
                logger.info(f"[{request_id}] Cleaning up queue")
                del thought_queues[request_id]
                logger.info(f"[{request_id}] Stream closed")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": ",".join(ALLOWED_ORIGINS),
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "X-Accel-Buffering": "no"  # Disable buffering for Nginx
        }
    )

def simulate_agent_thoughts(request_id: str, query: str):
    """Simulate agent thoughts based on the query content."""
    try:
        logger.info(f"Starting thought simulation for {request_id}")
        logger.info(f"[{request_id}] Query: {query}")
        
        # Initialize the thought process
        set_thought_status("processing", request_id)
        
        # Initial thought
        time.sleep(0.5)
        add_thought("thought", f"Analyzing query: '{query}'", request_id)
        
        # Determine query type to provide relevant thoughts
        query_lower = query.lower()
        
        # Define thought patterns based on query content with built-in delays
        thoughts = []
        delay = 1.5  # Base delay between thoughts in seconds
        
        # Food-related query
        if any(food_term in query_lower for food_term in ["food", "eat", "beef", "meat", "apple", "fruit"]):
            logger.info(f"[{request_id}] Detected food-related query")
            thoughts = [
                {"type": "thought", "content": "This query relates to food carbon footprint. I need to find data specific to this food item.", "delay": delay},
                {"type": "action", "content": "Searching for food carbon footprint data", "delay": delay},
                {"type": "thought", "content": "Food carbon footprints vary widely based on production methods, transportation, and processing.", "delay": delay},
                {"type": "action", "content": "Retrieving comparative food carbon footprint values", "delay": delay},
                {"type": "thought", "content": "I need to account for serving size and provide context like comparing to other foods.", "delay": delay}
            ]
        # Transport-related query
        elif any(transport_term in query_lower for transport_term in ["drive", "car", "transport"]):
            logger.info(f"[{request_id}] Detected transport-related query")
            thoughts = [
                {"type": "thought", "content": "This is about transportation carbon footprint. I need specific vehicle emission data.", "delay": delay},
                {"type": "action", "content": "Searching for transportation emissions data", "delay": delay},
                {"type": "thought", "content": "Vehicle emissions depend on fuel type, efficiency, distance, and number of passengers.", "delay": delay},
                {"type": "action", "content": "Calculating emissions based on distance and vehicle type", "delay": delay},
                {"type": "thought", "content": "I should provide comparative context with other transportation methods.", "delay": delay}
            ]
        # Energy-related query
        elif any(energy_term in query_lower for energy_term in ["electricity", "energy", "power"]):
            logger.info(f"[{request_id}] Detected energy-related query")
            thoughts = [
                {"type": "thought", "content": "This query is about energy usage. I need to find electricity carbon intensity data.", "delay": delay},
                {"type": "action", "content": "Searching for electricity carbon footprint data", "delay": delay},
                {"type": "thought", "content": "Carbon footprint of electricity varies by region based on energy mix.", "delay": delay},
                {"type": "action", "content": "Calculating emissions based on regional electricity grid data", "delay": delay},
                {"type": "thought", "content": "I should account for renewable energy sources in the calculation.", "delay": delay}
            ]
        # Default pattern for other queries
        else:
            logger.info(f"[{request_id}] Using default thought pattern for query")
            thoughts = [
                {"type": "thought", "content": "I need to understand what information is being requested and search for relevant data.", "delay": delay},
                {"type": "action", "content": "Searching for carbon footprint data related to the query", "delay": delay},
                {"type": "thought", "content": "Evaluating multiple sources to find accurate carbon footprint measurements.", "delay": delay},
                {"type": "action", "content": "Retrieving and comparing data from scientific sources", "delay": delay},
                {"type": "thought", "content": "Need to synthesize this information into a clear, accurate response with confidence levels.", "delay": delay}
            ]
        
        # Send the thoughts with a delay between each
        for i, thought_data in enumerate(thoughts):
            # Extract the delay and remove it from what we send
            current_delay = thought_data.pop("delay", delay)
            time.sleep(current_delay)  # Delay between thoughts
            
            # Add the thought
            add_thought(thought_data["type"], thought_data["content"], request_id)
        
        # Final delay before completing
        time.sleep(1.5)
        
        # Add a concluding thought
        add_thought("thought", "I've gathered all the relevant information and am ready to provide a complete answer.", request_id)
        
        # Signal completion - don't need to do this here as the query processing will handle it
        # set_thought_status("complete", request_id)
        
        logger.info(f"[{request_id}] Thought simulation completed")
        
    except Exception as e:
        logger.error(f"Error in simulate_agent_thoughts: {e}", exc_info=True)
        # Make sure we always set an error status
        set_thought_status("error", request_id)

def extract_structured_response(response_data):
    """
    Extract a properly structured response from potentially nested or malformed responses.
    
    Args:
        response_data: The response data from the crew manager
        
    Returns:
        A dictionary with normalized keys: answer, method, confidence, category, sources
    """
    # Helper function to parse Python-style dictionaries
    def try_parse_python_dict_style(s):
        try:
            import json
            s = s.replace("'", '"')
            s = s.replace("None", "null").replace("True", "true").replace("False", "false")
            return json.loads(s)
        except Exception as e:
            print(f"Failed to parse Python-style JSON: {e}")
            return None
    
    # Initialize default structured response
    structured_response = {
        "answer": "",
        "method": "Based on environmental data analysis.",
        "confidence": 0.7,
        "category": "Miscellaneous",
        "sources": []
    }
    
    # Get the response content - could be in response_data directly or in a 'response' field
    if isinstance(response_data, dict) and "response" in response_data:
        response_content = response_data["response"]
    else:
        response_content = response_data
    
    # Case 1: Handle JSON code blocks in strings
    if isinstance(response_content, str) and "```json" in response_content:
        try:
            import re
            import json
            match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                extracted = json.loads(json_str)
                if isinstance(extracted, dict):
                    response_content = extracted
        except Exception as e:
            print(f"Error extracting JSON from code block: {str(e)}")
    
    # Case 2: Already properly structured content
    if isinstance(response_content, dict) and all(k in response_content for k in ["answer", "method", "confidence", "category"]):
        # Check if answer field itself contains nested JSON or Python-style dict
        if isinstance(response_content["answer"], str):
            nested_json = None
            
            # Try standard JSON format
            if response_content["answer"].startswith("{") and response_content["answer"].endswith("}"):
                try:
                    import json
                    nested_json = json.loads(response_content["answer"])
                except Exception as e:
                    print(f"Failed to parse nested JSON in answer field: {str(e)}")
                    # Try Python style dict as fallback
                    nested_json = try_parse_python_dict_style(response_content["answer"])
            
            # If we successfully parsed nested JSON, check if it's a complete response
            if nested_json and isinstance(nested_json, dict) and all(k in nested_json for k in ["answer", "method", "confidence", "category"]):
                print("Extracted nested JSON or Python dict from answer field")
                return nested_json
        
        structured_response.update(response_content)
    
    # Case 3: Dict with some of the required fields
    elif isinstance(response_content, dict):
        # Update any matching fields
        for key in response_content:
            if key in structured_response:
                structured_response[key] = response_content[key]
        
        # Special case - if there's an "answer" field with JSON or Python-style dict in it
        if "answer" in response_content and isinstance(response_content["answer"], str):
            nested_json = None
            
            # Try standard JSON format
            if response_content["answer"].startswith("{") and response_content["answer"].endswith("}"):
                try:
                    import json
                    nested_json = json.loads(response_content["answer"])
                except Exception as e:
                    print(f"Error extracting JSON from answer field: {str(e)}")
                    # Try Python style dict
                    nested_json = try_parse_python_dict_style(response_content["answer"])
            
            # Process nested JSON if found
            if nested_json and isinstance(nested_json, dict):
                # If the nested JSON contains full structured response, return it directly
                if all(k in nested_json for k in ["answer", "method", "confidence", "category"]):
                    print("Extracted complete structured response from answer field")
                    return nested_json
                
                # Otherwise, only copy the fields we care about
                for key in nested_json:
                    if key in structured_response:
                        structured_response[key] = nested_json[key]
            
            # Also check for JSON inside ```json blocks
            elif "```json" in response_content["answer"]:
                try:
                    import re
                    import json
                    match = re.search(r'```json\s*(.*?)\s*```', response_content["answer"], re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        extracted = json.loads(json_str)
                        if isinstance(extracted, dict) and "answer" in extracted:
                            # Update only the fields we care about
                            for key in extracted:
                                if key in structured_response:
                                    structured_response[key] = extracted[key]
                except Exception as e:
                    print(f"Error extracting JSON from answer field code block: {str(e)}")
        
        # If we still don't have an answer, create one from available data
        if not structured_response["answer"] and response_content:
            if "value" in response_content:
                structured_response["answer"] = f"The carbon footprint is approximately {response_content['value']} {response_content.get('emission_unit', 'CO2e')}."
            else:
                structured_response["answer"] = str(response_content)
    
    # Case 4: String content
    elif isinstance(response_content, str):
        # Check if the entire string is a JSON object or Python-style dict
        json_data = None
        
        if response_content.strip().startswith("{") and response_content.strip().endswith("}"):
            try:
                import json
                json_data = json.loads(response_content)
            except Exception as e:
                print(f"Failed to parse string as JSON: {str(e)}")
                # Try Python style dict
                json_data = try_parse_python_dict_style(response_content)
        
        if json_data and isinstance(json_data, dict):
            if all(k in json_data for k in ["answer", "method", "confidence", "category"]):
                print("Extracted structured response from string JSON/dict")
                return json_data
            
            for key in json_data:
                if key in structured_response:
                    structured_response[key] = json_data[key]
            
            if "answer" not in json_data:
                structured_response["answer"] = str(json_data)
        else:
            structured_response["answer"] = response_content
    
    # Case 5: Any other type
    else:
        structured_response["answer"] = str(response_content)
    
    # Add sources if available in context
    if isinstance(response_data, dict) and "context" in response_data:
        context = response_data["context"]
        if isinstance(context, dict) and "sources" in context:
            structured_response["sources"] = context["sources"]
    
    # Normalize category
    category_mapping = {
        # Standard categories (keep as is)
        "Food & Diet": "Food & Diet",
        "Energy Use": "Energy Use", 
        "Mobility": "Mobility",
        "Purchases": "Purchases",
        "Miscellaneous": "Miscellaneous",
        
        # Variations and older categories
        "Food": "Food & Diet",
        "Diet": "Food & Diet",
        "Transportation": "Mobility",
        "Energy": "Energy Use",
        "Energy Usage": "Energy Use",
        "Shopping": "Purchases", 
        "Consumer Goods": "Purchases",
        "Carbon Footprint": "Miscellaneous",
        "Sustainability Practice": "Miscellaneous",
        "Emission Reduction": "Miscellaneous",
        "Carbon Offset": "Miscellaneous",
        "Data Analysis": "Miscellaneous",
        "Unknown": "Miscellaneous",
        "Error": "Miscellaneous"
    }
    
    current_category = structured_response["category"]
    structured_response["category"] = category_mapping.get(current_category, "Miscellaneous")
    
    # Add debug logging
    print("Final structured_response:")
    print(structured_response)
    
    return structured_response

@app.post("/api/query")
async def query_carbon(request: Request):
    """
    Process a carbon footprint query.
    """
    try:
        data = await request.json()
        query = data.get('query', '')
        request_id = data.get('request_id', None)
        use_crew = data.get('use_crew', False)  # Parameter to choose crew vs flow (default is flow)
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query cannot be empty"}
            )
        
        # Create stream file and tracking for the query
        write_thoughts_to_file([])
        set_thought_status('PROCESSING', request_id)
        
        # Process the query directly (async)
        try:
            result = await process_query_with_flow(query, request_id)
            return JSONResponse(
                content={"result": result, "request_id": request_id}
            )
        except Exception as process_err:
            logger.error(f"Error processing query: {str(process_err)}")
            return JSONResponse(
                status_code=500, 
                content={"error": f"Error processing query: {str(process_err)}"}
            )
    except Exception as e:
        print(f"Error in query_carbon endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error processing query: {str(e)}"}
        )

async def process_query_with_flow(query: str, request_id: str):
    """Process a query using the Flow-based approach with thought streaming."""
    try:
        # Start processing
        add_thought("start", f"Processing query: {query}", request_id)
        add_thought("thinking", "Using CarbonSenseFlow to process your query with event-driven architecture...", request_id)
        
        # Add intermediate thoughts to mimic the flow's steps
        add_thought("thinking", "Analyzing query intent and extracting entities...", request_id)
        add_thought("thinking", "Researching carbon footprint data from multiple sources...", request_id)
        add_thought("thinking", "Harmonizing data and calculating carbon estimates...", request_id)
        
        # Process query with flow (directly use the global instance) using async method
        result = await carbon_flow.process_query_with_flow_async(query)
        
        # Format result for output
        formatted_result = result
        if isinstance(result, dict) and "answer" in result:
            formatted_result = result["answer"]
            
        # Add completion thought with result
        add_thought("completion", formatted_result, request_id)
        
        # Update status
        set_thought_status("COMPLETE", request_id)
        
        return formatted_result
        
    except Exception as e:
        print(f"Error processing query with flow: {str(e)}")
        print(traceback.format_exc())
        
        # Add error thought
        add_thought("error", f"An error occurred while processing your query: {str(e)}", request_id)
        set_thought_status("ERROR", request_id)
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
        if not isinstance(result, dict):
            formatted_result = {
                "response": str(result),
                "transcription": transcript,
                "confidence": 0.8,
                "sources": []
            }
        else:
            formatted_result = result
            formatted_result["transcription"] = transcript
            if "confidence" not in formatted_result:
                formatted_result["confidence"] = 0.8
            if "sources" not in formatted_result:
                formatted_result["sources"] = []
        
        # Check for errors
        if isinstance(formatted_result, dict) and "error" in formatted_result:
            set_thought_status("ERROR", request_id)
            return JSONResponse(
                status_code=400,
                content={"error": formatted_result["error"], "message": formatted_result.get("response", "An error occurred")}
            )
        
        # Set thought processing as complete
        set_thought_status("COMPLETE", request_id)
        
        # Return the response
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
            print(f"---------------------------------> idx: {idx}, Device: {device}")
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
                        transcript_path = os.path.join(project_root, "transcript.txt")
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
            transcript_path = os.path.join(project_root, "transcript.txt")
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
                        # Return the result directly if we have valid content
                        if result.get("transcription") and result.get("response"):
                            logger.info(f"Found complete result in temp file for session {session_id}")
                            return {
                                "status": "complete",
                                "result": result
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
                    return {
                        "status": "complete",
                        "result": {
                            "transcription": _transcript,
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
                            "transcription": _transcript,
                            "response": latest_error.get("content", "An error occurred."),
                            "error": "Processing error",
                            "confidence": 0.5
                        }
                    }
            
            # If processing, show that it's in progress
            elif status == "PROCESSING":
                # Get the most recent thought to show progress
                if thoughts_data.get("thoughts"):
                    thoughts = sorted(thoughts_data["thoughts"], key=lambda x: x.get("timestamp", 0))
                    latest_thought = thoughts[-1]
                    return {
                        "status": "processing",
                        "progress": {
                            "transcription": _transcript,
                            "current_step": latest_thought.get("content", "Processing in progress..."),
                            "type": latest_thought.get("type", "thinking")
                        }
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
        return {
            "status": "transcribed",
            "result": {
                "transcription": _transcript,
                "response": f"Processing query: {_transcript}",
                "confidence": 0.5
            }
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
        transcript_path = os.path.join(project_root, "transcript.txt")
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
    transcript_path = os.path.join(project_root, "transcript.txt")
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
        transcript_path = os.path.join(project_root, "transcript.txt")
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
            debug_mode=False,
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
            debug_mode=False,
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
            with open(THOUGHTS_FILE_PATH, 'r') as f:
                thoughts_data = json.load(f)
                logger.info(f"Successfully read thoughts from file")
                return thoughts_data
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
    """Add a thought to the current state and write to file."""
    global current_thought_state
    
    try:
        # Log detailed debugging info
        logger.info(f"Adding thought - Type: {thought_type}, Content: {content}, Query ID: {query_id}")
        
        # Create thought object
        timestamp = time.time()
        thought = {
            "type": thought_type,
            "content": content,
            "timestamp": timestamp
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
def set_thought_status(status, query_id=None):
    """Set the status of the thought process."""
    global current_thought_state
    
    try:
        # Update query_id if provided
        if query_id is not None:
            current_thought_state["query_id"] = query_id
        
        # Update status
        current_thought_state["status"] = status
        logger.info(f"Updated thought process status to: {status} for query ID: {current_thought_state['query_id']}")
        
        # Write updated state to file
        success = write_thoughts_to_file(current_thought_state)
        
        return success
    except Exception as e:
        logger.error(f"Error setting thought status: {str(e)}")
        return False

# Add an API endpoint to check for thoughts
@app.get("/api/check-thoughts")
async def check_thoughts():
    """Check for thoughts from the file."""
    try:
        # Read thoughts from file
        thoughts_data = read_thoughts_from_file()
        
        if thoughts_data:
            logger.info(f"Returning thoughts data - Status: {thoughts_data.get('status')}, Thought count: {len(thoughts_data.get('thoughts', []))}")
            return thoughts_data
        else:
            logger.warning("No thoughts data available")
            return {
                "query_id": None,
                "thoughts": [],
                "status": "idle"
            }
    except Exception as e:
        logger.error(f"Error checking thoughts: {str(e)}")
        return {
            "query_id": None,
            "thoughts": [],
            "status": "error",
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
        
        # Format result for output (matching process_query_with_flow)
        formatted_result = result
        if isinstance(result, dict) and "answer" in result:
            formatted_result = result["answer"]
            
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

# Run the app using uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 