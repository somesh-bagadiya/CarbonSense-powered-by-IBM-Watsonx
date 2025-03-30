import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# Load environment variables
load_dotenv()

# Get API key and project ID from .env
API_KEY = os.getenv("COS_API_KEY")
PROJECT_ID = os.getenv("WATSON_STUDIO_PROJECT_ID")

# Initialize Watsonx Credentials
creds = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=API_KEY
)

# Initialize the LLM
llm = ModelInference(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    project_id=PROJECT_ID,
    credentials=creds
)

# Ask a question
question = "How can individuals reduce their carbon footprint at home?"

# Prompt format
prompt = f"Answer the following question in a clear and helpful way:\n\nQuestion: {question}\n\nAnswer:"

# Generate the answer
response = llm.generate(prompt=prompt)
answer = response['results'][0]['generated_text']

# Print result
print("🧠 Watsonx Answer:\n", answer)
