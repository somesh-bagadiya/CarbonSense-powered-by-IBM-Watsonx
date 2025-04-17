import os
import yaml
from pathlib import Path

# Find the directory where this file is located
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Load the YAML schema file
with open(current_dir.parent / 'core' / 'config' / 'common_schema.yaml', 'r') as f:
    schema_data = yaml.safe_load(f)

# Export the carbon metric schema
carbon_metric = schema_data['carbon_metric'] 