import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')

# Determine which service to use
USE_AZURE = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

# Model options
AZURE_MODEL_OPTIONS = ['gpt-4o']  # Azure deployment names
OPENAI_MODEL_OPTIONS = ['gpt-4o-2024-08-06']  # Standard OpenAI model names

MODEL_OPTIONS = AZURE_MODEL_OPTIONS if USE_AZURE else OPENAI_MODEL_OPTIONS

# Application settings
LANGUAGE_OPTIONS = ['English', 'Spanish', 'French', 'German', 'Japanese', 'Chinese']
CONSISTENCY_OPTIONS = ['Standardized Terminology', 'Consistent Formatting', 'Consistent Style']

# File upload settings
ALLOWED_FILE_TYPES = ["pdf", "docx", "txt"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Streamlit page configuration
PAGE_TITLE = "KB Dojo"
PAGE_LAYOUT = "wide"

# Title Styling
TITLE_STYLE = """
    <style>
        .kb-dojo-title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #4F8BF9;
            animation: glow 2s infinite alternate;
        }
        @keyframes glow {
            0% { filter: drop-shadow(0 0 5px #FFA07A); }
            50% { filter: drop-shadow(0 0 20px #4F8BF9); }
            100% { filter: drop-shadow(0 0 5px #FFA07A); }
        }
    </style>
"""

# Default Prompt Role
DEFAULT_PROMPT_ROLE = "You are a helpful assistant who writes knowledge base articles in KCS 6 format. Always respond in markdown."

# Error messages
API_ERROR_MESSAGE = "OpenAI API Error: {}"
FILE_LOAD_ERROR_MESSAGE = "Failed to load file content: {}"
UNSUPPORTED_FILE_TYPE_MESSAGE = "Unsupported file type: {}"
