# KBDojoToo

KBDojoToo is an advanced AI-powered knowledge base management and technical support system designed to streamline IT support processes, generate comprehensive KB articles, and provide intelligent routing for support tickets.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

### 1. Knowledge Base Article Generation

- **Single File Workflow**: Generate KB articles from individual files or user input.
- **Multiple Files Workflow**: Process multiple files simultaneously to create consistent KB articles.
- **Template Support**: Use custom templates for formatting KB articles.
- **Multilingual Support**: Translate KB articles into multiple languages.
- **Perspective Sections**: Create KB articles with different perspective sections (e.g., User, Support Analyst, Administrator).

### 2. ServiceNow Ticket Assignment Group Predictor

- **Incident Lookup**: Retrieve incident data using incident numbers.
- **AI-Powered Routing**: Predict the most appropriate assignment group for tickets using Azure ML.
- **Feedback System**: Collect user feedback on prediction accuracy for continuous improvement.

### 3. AI-Assisted Troubleshooting

- **Interactive Chat Interface**: Engage in a conversation with an AI assistant for technical support.
- **Context-Aware Responses**: Utilize incident details and chat history for more accurate assistance.
- **Dynamic KB Article Generation**: Convert chat conversations into formatted KB articles.
- **Resolution Steps Extraction**: Generate step-by-step resolution guides from troubleshooting sessions.

### 4. Advanced Research Capabilities

- **Perplexity Integration**: Leverage Perplexity AI for up-to-date research on technical topics.
- **Iterative Research**: Conduct multi-step research processes to gather comprehensive information.

### 5. Customization and Flexibility

- **Temperature Control**: Adjust the creativity and randomness of AI-generated content.
- **Token Limit Configuration**: Set maximum token limits for generated content.
- **Model Selection**: Choose between different AI models for various tasks.

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/KBDojoToo.git
   cd KBDojoToo

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

3. Install the required packages:
   pip install -r requirements.txt

4. Set up environment variables:
   - Copy `example.env` to `.env`
   - Fill in the necessary API keys and configuration values

## Usage

To run the application:

streamlit run app.py

Navigate to the provided URL (usually `http://localhost:8501`) in your web browser to access the KBDojoToo interface.

## Components

- `app.py`: Main Streamlit application entry point
- `core_functions.py`: Core functionality for KB article generation and processing
- `config.py`: Configuration settings and constants
- `utils/`: Utility functions for file handling, API interactions, and more
- `layout/`: Streamlit page layouts and UI components

## Configuration

Adjust the following files to customize the application:

- `.env`: Environment variables for API keys and global settings
- `config.py`: Application-wide configuration options
- `requirements.txt`: Python package dependencies

## Contributing

Contributions to KBDojoToo are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Create a new Pull Request

## License

[Insert your chosen license here]

---

For more information or support, please contact [Your Contact Information].