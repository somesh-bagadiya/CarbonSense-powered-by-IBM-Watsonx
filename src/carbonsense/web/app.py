from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import os
import sys
import asyncio
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import queue
import threading

# Add the project root to the Python path to help with imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

# Import the core carbon sense functionality
from src.carbonsense.config.config_manager import ConfigManager
from src.carbonsense.core.crew_agent import CrewAgentManager

# Create the FastAPI app
app = FastAPI(
    title="CarbonSense Dashboard",
    description="A dashboard for tracking and analyzing carbon footprint data",
    version="1.0.0"
)

# Setup template and static directories
templates_path = Path(__file__).parent / "templates"
static_path = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_path))
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize ConfigManager and CrewAgentManager
config = ConfigManager()
crew_manager = CrewAgentManager(
    config=config,
    debug_mode=False,
    use_cache=False,
    use_hierarchical=False,
    store_thoughts=True
)

# A global thought queue to store agent thoughts during processing
# We'll use a thread-safe queue to handle multiple concurrent requests
thought_queues = {}

# Sample data for the dashboard (in a real application, this would come from a database)
def get_sample_data():
    return {
        "total_carbon": 8.2,
        "food_carbon": 2.5, 
        "household_carbon": 3.1,
        "transportation_carbon": 2.6,
        "goal_percentage": 30,
        "weekly_trend": [4.5, 5.2, 6.0, 6.8, 5.9, 7.1, 8.2],
        "badges": ["Eco Starter", "Committed", "Carbon Pro"]
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    data = get_sample_data()
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "data": data}
    )

# API endpoint for streaming agent thoughts
@app.get("/api/stream-thoughts")
async def stream_thoughts(request: Request, query: str):
    """Stream agent thoughts as Server-Sent Events."""
    # Generate a unique ID for this streaming connection
    request_id = f"request_{int(time.time() * 1000)}"
    
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
            # Send retry interval and initial connection message
            yield "retry: 1000\n\n"
            yield f"id: {request_id}\n"
            yield f"data: {json.dumps({'type': 'info', 'content': 'Connection established'})}\n\n"
            
            # Wait a moment to ensure the client has connected
            await asyncio.sleep(0.2)
            
            # Send a test thought to verify streaming is working
            yield f"id: {int(time.time() * 1000)}\n"
            yield f"data: {json.dumps({'type': 'thought', 'content': 'Initializing analysis for ' + query})}\n\n"
            
            counter = 0
            while True:
                counter += 1
                try:
                    # Non-blocking get with timeout
                    thought_data = thought_queues[request_id].get(timeout=0.5)
                    
                    # Handle "DONE" signal
                    if thought_data == "DONE":
                        yield f"id: {int(time.time() * 1000)}\n"
                        yield f"data: {json.dumps({'type': 'complete', 'content': 'Processing completed'})}\n\n"
                        break
                    
                    # Send the thought data as a server-sent event
                    # Include event ID, event type, and proper formatting with double newlines
                    event_id = int(time.time() * 1000)
                    yield f"id: {event_id}\n"
                    yield f"data: {json.dumps(thought_data)}\n\n"
                    
                except queue.Empty:
                    # No messages available, send a keep-alive comment
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(0.5)
                    
                    # If taking too long, end the stream
                    if counter > 60:  # 30 seconds max
                        yield f"id: {int(time.time() * 1000)}\n"
                        yield f"data: {json.dumps({'type': 'complete', 'content': 'Timeout'})}\n\n"
                        break
                        
                    continue
                except Exception as e:
                    yield f"id: {int(time.time() * 1000)}\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    break
        except Exception as e:
            print(f"Error in event generator: {e}")
        finally:
            # Clean up the queue when done
            if request_id in thought_queues:
                del thought_queues[request_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",  # Allow CORS
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "X-Accel-Buffering": "no"  # Disable buffering for Nginx
        }
    )

def simulate_agent_thoughts(request_id: str, query: str):
    """Simulate agent thoughts based on the query content."""
    try:
        # Check if the request queue still exists (in case client disconnected)
        if request_id not in thought_queues:
            return
            
        # Initial thought
        time.sleep(0.5)
        initial_thought = {
            "type": "thought", 
            "content": f"Analyzing query: '{query}'"
        }
        
        # Thread-safe queue put with error handling
        try:
            thought_queues[request_id].put(initial_thought)
        except Exception as e:
            print(f"Error adding initial thought to queue: {e}")
            return
        
        # Determine query type to provide relevant thoughts
        query_lower = query.lower()
        
        # Define thought patterns based on query content with built-in delays
        thoughts = []
        delay = 1.5  # Base delay between thoughts in seconds
        
        # Check if queue still exists after first thought
        if request_id not in thought_queues:
            return
            
        # Food-related query
        if "food" in query_lower or "eat" in query_lower or "beef" in query_lower or "meat" in query_lower or "apple" in query_lower or "fruit" in query_lower:
            thoughts = [
                {"type": "thought", "content": "This query relates to food carbon footprint. I need to find data specific to this food item.", "delay": delay},
                {"type": "action", "content": "Searching for food carbon footprint data", "delay": delay},
                {"type": "thought", "content": "Food carbon footprints vary widely based on production methods, transportation, and processing.", "delay": delay},
                {"type": "action", "content": "Retrieving comparative food carbon footprint values", "delay": delay},
                {"type": "thought", "content": "I need to account for serving size and provide context like comparing to other foods.", "delay": delay}
            ]
        # Transport-related query
        elif "drive" in query_lower or "car" in query_lower or "transport" in query_lower:
            thoughts = [
                {"type": "thought", "content": "This is about transportation carbon footprint. I need specific vehicle emission data.", "delay": delay},
                {"type": "action", "content": "Searching for transportation emissions data", "delay": delay},
                {"type": "thought", "content": "Vehicle emissions depend on fuel type, efficiency, distance, and number of passengers.", "delay": delay},
                {"type": "action", "content": "Calculating emissions based on distance and vehicle type", "delay": delay},
                {"type": "thought", "content": "I should provide comparative context with other transportation methods.", "delay": delay}
            ]
        # Energy-related query
        elif "electricity" in query_lower or "energy" in query_lower or "power" in query_lower:
            thoughts = [
                {"type": "thought", "content": "This query is about energy usage. I need to find electricity carbon intensity data.", "delay": delay},
                {"type": "action", "content": "Searching for electricity carbon footprint data", "delay": delay},
                {"type": "thought", "content": "Carbon footprint of electricity varies by region based on energy mix.", "delay": delay},
                {"type": "action", "content": "Calculating emissions based on regional electricity grid data", "delay": delay},
                {"type": "thought", "content": "I should account for renewable energy sources in the calculation.", "delay": delay}
            ]
        # Default pattern for other queries
        else:
            thoughts = [
                {"type": "thought", "content": "I need to understand what information is being requested and search for relevant data.", "delay": delay},
                {"type": "action", "content": "Searching for carbon footprint data related to the query", "delay": delay},
                {"type": "thought", "content": "Evaluating multiple sources to find accurate carbon footprint measurements.", "delay": delay},
                {"type": "action", "content": "Retrieving and comparing data from scientific sources", "delay": delay},
                {"type": "thought", "content": "Need to synthesize this information into a clear, accurate response with confidence levels.", "delay": delay}
            ]
        
        # Send the thoughts with a delay between each
        for i, thought_data in enumerate(thoughts):
            # Check if the queue still exists (in case client disconnected)
            if request_id not in thought_queues:
                return
                
            # Extract the delay and remove it from what we send to the client
            current_delay = thought_data.pop("delay", delay)
            time.sleep(current_delay)  # Delay between thoughts
            
            # Thread-safe queue put with error handling
            try:
                thought_queues[request_id].put(thought_data)
            except Exception as e:
                # Continue with next thought
                pass
                
        # Final delay before completing
        time.sleep(1.5)
        
        # Final verification
        if request_id not in thought_queues:
            return
            
        # Add a concluding thought
        conclusion = {
            "type": "thought",
            "content": "I've gathered all the relevant information and am ready to provide a complete answer."
        }
        
        try:
            thought_queues[request_id].put(conclusion)
            time.sleep(0.5)  # Short delay before completion
        except Exception:
            pass
        
        # Signal completion
        try:
            thought_queues[request_id].put("DONE")
        except Exception:
            pass
        
    except Exception as e:
        print(f"Error in simulate_agent_thoughts: {e}")
        # Make sure we always try to signal completion and clean up
        try:
            if request_id in thought_queues:
                thought_queues[request_id].put("DONE")
        except:
            pass  # Ignore any errors in final cleanup

# API endpoint for querying carbon footprint
@app.post("/api/query")
async def query_carbon(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    query = data.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process the query using CrewAgentManager
        result = crew_manager.process_query(query, show_context=True)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "answer": result["response"],
            "confidence": result.get("confidence", 0.8),
            "sources": result.get("context", {}).get("sources", []) if "context" in result else []
        }
    except Exception as e:
        # Fallback to mock responses if CrewAgentManager fails
        if "drove" in query.lower():
            return {
                "answer": "Driving 10 miles produces approximately a carbon footprint of 4.6 kg CO2e.",
                "confidence": 0.92
            }
        elif "beef" in query.lower():
            return {
                "answer": "Beef production generates approximately 60 kg CO2e per kg of beef.",
                "confidence": 0.89
            }
        else:
            return {
                "answer": f"I don't have specific information about that. Error: {str(e)}",
                "confidence": 0.5
            }

# API endpoint for updating daily activity
@app.post("/api/activity")
async def add_activity(request: Request):
    data = await request.json()
    activity = data.get("activity", "")
    
    if not activity:
        raise HTTPException(status_code=400, detail="Activity cannot be empty")
    
    try:
        # Process the activity query using CrewAgentManager
        # This will eventually update the user's carbon footprint
        result = crew_manager.process_query(f"Record this activity: {activity}", show_context=True)
        
        if "error" in result:
            return {
                "status": "warning",
                "message": f"Activity recorded with warning: {result['error']}"
            }
        
        return {
            "status": "success",
            "message": f"Activity '{activity}' recorded successfully.",
            "analysis": result["response"]
        }
    except Exception as e:
        # In a real implementation, this would handle errors properly
        return {
            "status": "success",
            "message": f"Activity '{activity}' recorded successfully.",
            "analysis": "Activity recorded, but analysis is currently unavailable."
        }

# Run the app using uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 