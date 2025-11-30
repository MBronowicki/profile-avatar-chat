import os
from dotenv import load_dotenv

from langsmith import Client, traceable

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
class Config:
    def __init__(self):
        load_dotenv(override=True)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")

        # Initialize LangSmith
        self.langsmith_client = Client(api_key=self.langsmith_api_key)
