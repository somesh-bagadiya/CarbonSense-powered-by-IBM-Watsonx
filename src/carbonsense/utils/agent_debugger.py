import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import html

class AgentDebugger:
    """Debug handler for AI agent interactions."""
    
    def __init__(self, debug_dir: str = None):
        """
        Initialize the agent debugger.
        
        Args:
            debug_dir: Directory to store debug files (defaults to 'logs' in project root)
        """
        if debug_dir is None:
            # Get project root directory
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", ".."
            ))
            debug_dir = os.path.join(project_root, "logs", "debug")
        
        self.debug_dir = debug_dir
        
        # Create debug directory if it doesn't exist
        os.makedirs(debug_dir, exist_ok=True)
        
        # Dictionary to store sessions
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_interaction(self, session_id: str, agent_name: str, 
                         prompt: str, completion: str,
                         parameters: Dict[str, Any] = None,
                         metadata: Dict[str, Any] = None) -> None:
        """
        Record an agent interaction for debugging.
        
        Args:
            session_id: ID of the session
            agent_name: Name of the agent
            prompt: The prompt sent to the agent
            completion: The completion received from the agent
            parameters: AI parameters used (temperature, tokens, etc.)
            metadata: Any additional metadata about the interaction
        """
        # Ensure session exists in our tracking dictionary
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Create interaction record
        interaction = {
            "id": len(self.sessions[session_id]),  # Use array index as ID
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "prompt": prompt,
            "completion": completion,
            "parameters": parameters or {},
            "metadata": metadata or {}
        }
        
        # Add to session
        self.sessions[session_id].append(interaction)
    
    def export_session_to_html(self, session_id: str) -> str:
        """
        Export a session's debug information to an HTML file.
        
        Args:
            session_id: ID of the session to export
            
        Returns:
            Path to the generated HTML file, or empty string if session not found
        """
        if session_id not in self.sessions:
            return ""
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_session_{session_id}_{timestamp}.html"
        filepath = os.path.join(self.debug_dir, filename)
        
        # Generate HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._generate_debug_html(session_id))
        
        return filepath
    
    def _generate_debug_html(self, session_id: str) -> str:
        """
        Generate an HTML representation of the session's debug information.
        
        Args:
            session_id: ID of the session to generate HTML for
            
        Returns:
            HTML content as a string
        """
        if session_id not in self.sessions:
            return "<html><body><h1>Session not found</h1></body></html>"
        
        # Get all interactions for this session
        interactions = self.sessions[session_id]
        
        # Group interactions by agent
        agents = {}
        for interaction in interactions:
            agent_name = interaction["agent_name"]
            if agent_name not in agents:
                agents[agent_name] = []
            agents[agent_name].append(interaction)
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Debug Session {session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .session-info {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
                .agent-section {{ margin-bottom: 30px; }}
                .agent-header {{ background-color: #007BFF; color: white; padding: 5px 10px; }}
                .interaction {{ border: 1px solid #ddd; margin-bottom: 15px; }}
                .interaction-header {{ background-color: #e9ecef; padding: 5px 10px; display: flex; justify-content: space-between; }}
                .prompt, .completion {{ padding: 10px; white-space: pre-wrap; overflow-x: auto; }}
                .prompt {{ background-color: #f8f9fa; }}
                .parameters {{ background-color: #e9f7ef; padding: 10px; border-top: 1px solid #ddd; }}
                .metadata {{ background-color: #fdf2e9; padding: 10px; border-top: 1px solid #ddd; }}
                .toggle-btn {{ margin-left: 10px; cursor: pointer; color: blue; }}
                pre {{ margin: 0; }}
                .hidden {{ display: none; }}
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{
                    display: none;
                    padding: 6px 12px;
                    border: 1px solid #ccc;
                    border-top: none;
                }}
            </style>
        </head>
        <body>
            <h1>Debug Session {session_id}</h1>
            
            <div class="session-info">
                <p><strong>Total interactions:</strong> {len(interactions)}</p>
                <p><strong>Agents:</strong> {', '.join(agents.keys())}</p>
                <p><strong>Start time:</strong> {interactions[0]['timestamp'] if interactions else 'N/A'}</p>
                <p><strong>End time:</strong> {interactions[-1]['timestamp'] if interactions else 'N/A'}</p>
            </div>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'ByAgent')">By Agent</button>
                <button class="tablinks" onclick="openTab(event, 'Timeline')">Timeline</button>
            </div>
            
            <div id="ByAgent" class="tabcontent" style="display: block;">
        """
        
        # Add agents sections
        for agent_name, agent_interactions in agents.items():
            html_content += f"""
                <div class="agent-section">
                    <h2 class="agent-header">Agent: {agent_name}</h2>
            """
            
            # Add interactions for this agent
            for interaction in agent_interactions:
                html_content += self._generate_interaction_html(interaction)
            
            html_content += """
                </div>
            """
        
        html_content += """
            </div>
            
            <div id="Timeline" class="tabcontent">
        """
        
        # Add timeline view
        for interaction in sorted(interactions, key=lambda x: x['timestamp']):
            html_content += self._generate_interaction_html(interaction, include_agent=True)
        
        html_content += """
            </div>
            
            <script>
                function toggleContent(contentId) {
                    const content = document.getElementById(contentId);
                    const btn = document.querySelector(`[onclick="toggleContent('${contentId}')"]`);
                    if (content.classList.contains('hidden')) {
                        content.classList.remove('hidden');
                        btn.textContent = 'Hide';
                    } else {
                        content.classList.add('hidden');
                        btn.textContent = 'Show';
                    }
                }
                
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_interaction_html(self, interaction: Dict[str, Any], include_agent: bool = False) -> str:
        """
        Generate HTML for a single interaction.
        
        Args:
            interaction: The interaction to generate HTML for
            include_agent: Whether to include the agent name in the header
            
        Returns:
            HTML content as a string
        """
        interaction_id = interaction["id"]
        prompt_id = f"prompt_{interaction_id}"
        completion_id = f"completion_{interaction_id}"
        params_id = f"params_{interaction_id}"
        metadata_id = f"metadata_{interaction_id}"
        
        # Escape HTML characters
        prompt = html.escape(interaction["prompt"])
        completion = html.escape(interaction["completion"])
        params = json.dumps(interaction["parameters"], indent=2)
        metadata = json.dumps(interaction["metadata"], indent=2)
        
        agent_info = f"Agent: {interaction['agent_name']} | " if include_agent else ""
        timestamp = datetime.fromisoformat(interaction["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
            <div class="interaction">
                <div class="interaction-header">
                    <span>{agent_info}Interaction #{interaction_id} | {timestamp}</span>
                    <div>
                        <button class="toggle-btn" onclick="toggleContent('{prompt_id}')">Hide</button>
                        <button class="toggle-btn" onclick="toggleContent('{completion_id}')">Hide</button>
                        <button class="toggle-btn" onclick="toggleContent('{params_id}')">Hide</button>
                        <button class="toggle-btn" onclick="toggleContent('{metadata_id}')">Hide</button>
                    </div>
                </div>
                <div id="{prompt_id}" class="prompt">
                    <h4>Prompt:</h4>
                    <pre>{prompt}</pre>
                </div>
                <div id="{completion_id}" class="completion">
                    <h4>Completion:</h4>
                    <pre>{completion}</pre>
                </div>
                <div id="{params_id}" class="parameters">
                    <h4>Parameters:</h4>
                    <pre>{params}</pre>
                </div>
                <div id="{metadata_id}" class="metadata">
                    <h4>Metadata:</h4>
                    <pre>{metadata}</pre>
                </div>
            </div>
        """
    
    def print_interaction_details(self, session_id: str, agent_name: Optional[str] = None, 
                               interaction_id: Optional[int] = None) -> None:
        """
        Print the details of a specific interaction to the console.
        
        Args:
            session_id: ID of the session
            agent_name: Optional name of the agent to print interactions for
            interaction_id: Optional ID of the specific interaction to print
        """
        if session_id not in self.sessions:
            print(f"Session {session_id} not found.")
            return
            
        interactions = self.sessions[session_id]
        
        # Filter by agent if specified
        if agent_name:
            interactions = [i for i in interactions if i["agent_name"] == agent_name]
            
            if not interactions:
                print(f"No interactions found for agent {agent_name} in session {session_id}.")
                return
        
        # Print specific interaction if ID provided
        if interaction_id is not None:
            filtered_interactions = [i for i in interactions if i["id"] == interaction_id]
            
            if not filtered_interactions:
                print(f"Interaction {interaction_id} not found.")
                return
                
            interaction = filtered_interactions[0]
            self._print_detailed_interaction(interaction)
            return
            
        # Print summary of all interactions
        print(f"\n=== Session {session_id} Summary ===")
        print(f"Total interactions: {len(interactions)}")
        
        # Group by agent
        agents = {}
        for interaction in interactions:
            agent_name = interaction["agent_name"]
            if agent_name not in agents:
                agents[agent_name] = 0
            agents[agent_name] += 1
            
        print("\nInteractions by agent:")
        for agent, count in agents.items():
            print(f"- {agent}: {count}")
            
        print("\nTo view a specific interaction, provide an interaction_id parameter.")
    
    def _print_detailed_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Print the details of a specific interaction in a formatted way.
        
        Args:
            interaction: The interaction to print
        """
        print("\n" + "=" * 80)
        print(f"Interaction #{interaction['id']} | Agent: {interaction['agent_name']}")
        print(f"Timestamp: {interaction['timestamp']}")
        print("=" * 80)
        
        print("\nParameters:")
        print("-" * 80)
        print(json.dumps(interaction["parameters"], indent=2))
        
        if interaction["metadata"]:
            print("\nMetadata:")
            print("-" * 80)
            print(json.dumps(interaction["metadata"], indent=2))
        
        print("\nPrompt:")
        print("-" * 80)
        print(interaction["prompt"])
        
        print("\nCompletion:")
        print("-" * 80)
        print(interaction["completion"])
        print("=" * 80 + "\n")