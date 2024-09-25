import os
from langchain.llms import OpenAI  # Import OpenAI from langchain
import streamlit as st
from typing import List, Dict, Optional
from config import DEFAULT_PROMPT_ROLE, OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Initialize Langchain's OpenAI instance
chat = ChatOpenAI(
    model_name="gpt-4o-2024-08-06",  # or your default model
    temperature=0.7,
    max_tokens=2000,
    openai_api_key=OPENAI_API_KEY
)

def validate_headers_and_numbers(content: str) -> str:
    """
    Validate headers and document numbers in the content.
    Ensure they're correct or leave them blank.

    Args:
        content: The article content.

    Returns:
        The validated article content.
    """
    if "Header" not in content:
        content = "Header: \n" + content
    if "Document Number" not in content:
        content = "Document Number: \n" + content
    return content


def create_perspective_sections(content: str, perspectives: list) -> str:
    """
    Create sections for different perspectives within the content.

    Args:
        content: The article content.
        perspectives: A list of perspectives to create sections for.

    Returns:
        The article content with perspective sections.
    """
    sections = []
    for perspective in perspectives:
        sections.append(f"### {perspective} Section\n\n{content}\n")
    return "\n".join(sections)


def create_base_prompt(title: str, content: str, template_content: str = None, options: dict = None) -> str:
    """
    Create the base prompt for the OpenAI API.

    Args:
        title: The title of the article.
        content: The content of the article.
        template_content: The content of the template, if provided.
        options: The selected options for the article.

    Returns:
        The base prompt as a string.
    """
    if template_content:
        return f"\n- Construct a KB article from the following user information using the provided template: Title\n\n{title}\n\nContent\n\n{content}\n\nTemplate\n\n{template_content}\n\n"
    else:
        return f"\n- Construct a KB article from the following user information: Title\n\n{title}\n\nContent\n\n{content}\n\n"


def modify_prompt_with_options(prompt: str, options: dict) -> str:
    """
    Modify the prompt based on the selected options.

    Args:
        prompt: The base prompt.
        options: The selected options for the article.

    Returns:
        The modified prompt as a string.
    """
    consistency_prompt = "Consistency requirements:\n"
    for option, value in options.items():
        if option.startswith("Consistent") and value:
            consistency_prompt += f"- {option}\n"
    prompt += f"\n{consistency_prompt}"

    if options.get('Action') == 'Reauthor Content':
        prompt = "Reauthor the following content, maintaining its core information but improving its clarity, structure, and readability. Follow KCS 6 standards and write for Service Now KB publication:\n\n" + prompt
    elif options.get('Action') == 'Format to Template':
        prompt = "Format the following content to match the provided template structure, maintaining the original information as much as possible:\n\n" + prompt

    return prompt


async def translate_content(content: str, language: str, selected_model: str, options: Dict) -> str:
    prompt = f"""Translate the following markdown-formatted text to {language}. 
    Maintain all markdown formatting, including headers, bold text, italics, lists, and code blocks.
    Ensure that the structure and formatting of the original text is preserved in the translation.
    Here's the text to translate:

    {content}
    """

    try:
        messages = [
            SystemMessage(content=DEFAULT_PROMPT_ROLE),
            HumanMessage(content=prompt)
        ]
        response = await chat.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return content  # Return original content if translation fails


def create_incident_context(ticket_data, top_prediction):
    return f"""
    You are a helpful AI assistant for IT support. Here are the raw incident details:

    {json.dumps(ticket_data, indent=2)}

    Predicted Assignment Group: {top_prediction}

    The description field contains all the information that the user has provided so far about the issue.
    Please use these details to provide assistance. Do not summarize or omit any information.
    Await the user's first message for specific questions or requests for help.
    """