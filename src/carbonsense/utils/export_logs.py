"""Utility for exporting agent interaction logs to human-readable formats."""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, List, Any, Optional

def load_log_file(log_path: str) -> Dict[str, Any]:
    """Load a JSON log file containing agent interactions.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Dictionary containing the log data
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {log_path} is not a valid JSON file")
        return {}
    except FileNotFoundError:
        print(f"Error: {log_path} not found")
        return {}

def load_params_file(params_path: str) -> Dict[str, Any]:
    """Load a YAML file containing AI parameters.
    
    Args:
        params_path: Path to the parameters file
        
    Returns:
        Dictionary containing the parameter data
    """
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError:
        print(f"Error: {params_path} is not a valid YAML file")
        return {}
    except FileNotFoundError:
        print(f"Error: {params_path} not found")
        return {}

def format_json(json_data: Any) -> str:
    """Format JSON data for display in HTML.
    
    Args:
        json_data: JSON data to format
        
    Returns:
        Formatted JSON string
    """
    try:
        if isinstance(json_data, (dict, list)):
            return json.dumps(json_data, indent=2)
        elif isinstance(json_data, str):
            try:
                parsed = json.loads(json_data)
                return json.dumps(parsed, indent=2)
            except:
                return json_data
        else:
            return str(json_data)
    except:
        return str(json_data)

def generate_html(log_data: Dict[str, Any], params_data: Dict[str, Any], output_path: str) -> None:
    """Generate an HTML report from log data and parameter data.
    
    Args:
        log_data: Dictionary containing agent interaction logs
        params_data: Dictionary containing AI parameters
        output_path: Path to save the HTML report
    """
    # Extract key information
    session_id = log_data.get("session_id", "Unknown")
    timestamp = log_data.get("timestamp", "Unknown")
    interactions = log_data.get("interactions", {})
    
    # Start building HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CarbonSense Agent Interactions - {session_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        .session-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .agent-section {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .interaction {{
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .interaction-header {{
            background-color: #34495e;
            color: white;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
        }}
        .interaction-header button {{
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }}
        .interaction-content {{
            padding: 0;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .input, .output {{
            padding: 15px;
        }}
        .input {{
            background-color: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }}
        .output {{
            background-color: #ffffff;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .params-section {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .tab {{
            overflow: hidden;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
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
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #34495e;
            color: white;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            background-color: white;
            border-radius: 0 0 5px 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CarbonSense Agent Interactions</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="session-info">
            <h2>Session Information</h2>
            <p><strong>Session ID:</strong> {session_id}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
        </div>

        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'InteractionsTab')">Agent Interactions</button>
            <button class="tablinks" onclick="openTab(event, 'ParamsTab')">AI Parameters</button>
        </div>

        <div id="InteractionsTab" class="tabcontent" style="display: block;">
"""

    # Process each agent's interactions
    for agent_name, agent_interactions in interactions.items():
        html += f"""
            <div class="agent-section">
                <h2>{agent_name.title()} Agent</h2>
                <p><strong>Total Interactions:</strong> {len(agent_interactions)}</p>
"""
        
        # Add each interaction
        for i, interaction in enumerate(agent_interactions):
            interaction_id = interaction.get("interaction_id", f"interaction-{i}")
            timestamp = interaction.get("timestamp", "Unknown time")
            
            # Format input and output
            input_text = format_json(interaction.get("input", "No input provided"))
            output_text = format_json(interaction.get("output", "No output provided"))
            
            # Get metadata
            metadata = interaction.get("metadata", {})
            source_type = interaction.get("source_type", "Not specified")
            tools_used = ", ".join(interaction.get("tools_used", ["None"]))
            
            html += f"""
                <div class="interaction">
                    <div class="interaction-header">
                        <span>Interaction {interaction_id} - {timestamp}</span>
                        <button onclick="toggleInteraction(this)">▼</button>
                    </div>
                    <div class="interaction-content">
                        <div class="input">
                            <h3>Input</h3>
                            <pre>{input_text}</pre>
                        </div>
                        <div class="output">
                            <h3>Output</h3>
                            <pre>{output_text}</pre>
                        </div>
                        <div class="metadata">
                            <h3>Metadata</h3>
                            <p><strong>Source Type:</strong> {source_type}</p>
                            <p><strong>Tools Used:</strong> {tools_used}</p>
                            <pre>{format_json(metadata)}</pre>
                        </div>
                    </div>
                </div>
"""
        html += """
            </div>
"""

    # Add Parameter Information Tab
    html += """
        </div>
        <div id="ParamsTab" class="tabcontent">
            <div class="params-section">
                <h2>Default Parameters</h2>
"""
    
    # Add default parameters
    default_params = params_data.get("default", {})
    html += f"""
                <pre>{format_json(default_params)}</pre>
                
                <h2>Agent-Specific Parameters</h2>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Role</th>
                        <th>Temperature</th>
                        <th>Model</th>
                        <th>Sources Tracked</th>
                    </tr>
"""
    
    # Add each agent's parameters
    agents_params = params_data.get("agents", {})
    for agent_name, agent_params in agents_params.items():
        role = agent_params.get("role", "Not specified")
        temperature = agent_params.get("temperature", "Not specified")
        model = agent_params.get("model", "Not specified")
        sources_tracked = "Yes" if agent_params.get("sources_tracked", False) else "No"
        
        html += f"""
                    <tr>
                        <td>{agent_name}</td>
                        <td>{role}</td>
                        <td>{temperature}</td>
                        <td>{model}</td>
                        <td>{sources_tracked}</td>
                    </tr>
"""
    
    html += """
                </table>
                
                <h2>Output Format</h2>
"""
    
    # Add output format
    output_format = params_data.get("output_format", {})
    html += f"""
                <pre>{format_json(output_format)}</pre>
                
                <h2>Data Sources</h2>
"""
    
    # Add data sources
    data_sources = params_data.get("data_sources", {})
    html += f"""
                <pre>{format_json(data_sources)}</pre>
                
                <h2>Tools</h2>
"""
    
    # Add tools
    tools = params_data.get("tools", {})
    html += f"""
                <pre>{format_json(tools)}</pre>
            </div>
        </div>
    </div>

    <script>
        // Toggle interaction visibility
        function toggleInteraction(button) {{
            const content = button.closest(".interaction").querySelector(".interaction-content");
            
            if (content.style.maxHeight) {{
                button.textContent = "▼";
                content.style.maxHeight = null;
            }} else {{
                button.textContent = "▲";
                content.style.maxHeight = content.scrollHeight + "px";
            }}
        }}
        
        // Tab functionality
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Automatically expand first interaction
        document.addEventListener("DOMContentLoaded", function() {{
            if (document.querySelector(".interaction")) {{
                const firstButton = document.querySelector(".interaction-header button");
                toggleInteraction(firstButton);
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"HTML report generated at: {output_path}")

def main():
    """Main function for the export utility."""
    parser = argparse.ArgumentParser(description="Export agent interaction logs to human-readable formats")
    parser.add_argument("--log", required=True, help="Path to the agent interaction log file")
    parser.add_argument("--params", default=None, help="Path to the AI parameters file")
    parser.add_argument("--output", default=None, help="Output path for the HTML report")
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.params:
        # Try to find the params file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        carbonsense_dir = os.path.dirname(script_dir)
        default_params = os.path.join(carbonsense_dir, "config", "ai_parameters.yaml")
        args.params = default_params if os.path.exists(default_params) else None
        
        if not args.params:
            print("Warning: AI parameters file not found, generating report without parameters")
    
    if not args.output:
        log_path = Path(args.log)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{log_path.stem}_report_{timestamp}.html"
    
    # Load the data
    log_data = load_log_file(args.log)
    params_data = load_params_file(args.params) if args.params else {}
    
    # Generate the HTML report
    if log_data:
        generate_html(log_data, params_data, args.output)

if __name__ == "__main__":
    main()