"""
CarbonSense: A carbon footprint estimation system powered by IBM watsonx
"""

# Make sure core components are available
from .core.crew_agent import CrewAgentManager, CarbonSenseCrew
from .core.carbon_flow import CarbonSenseFlow, CarbonSenseState

# Version info
__version__ = "0.1.0" 