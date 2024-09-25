# layout/main_content.py

import streamlit as st
from utils import file_handlers
from config import ALLOWED_FILE_TYPES
from core_functions import handle_chat_message, convert_chat_to_kb_article, convert_chat_to_resolution_steps
import asyncio
from layout.troubleshooting_page import display_troubleshooting_page
from typing import List, Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

def setup_main_content() -> Tuple[List[str], List[str], str, Optional[st.runtime.uploaded_file_manager.UploadedFile]]:
    """
    Set up the main content area of the application.

    Returns:
        Tuple containing:
        - List of KB article titles
        - List of KB article contents
        - Selected workflow option
        - Uploaded template file (if any)
    """
    st.subheader('Enter KB Article Details')
    workflow_option = st.radio("Select Workflow", ("Single File", "Multiple Files", "Troubleshooting"))

    kb_article_titles: List[str] = []
    kb_article_contents: List[str] = []

    if workflow_option == "Single File":
        kb_article_titles, kb_article_contents = handle_single_file_workflow()
    elif workflow_option == "Multiple Files":
        kb_article_titles, kb_article_contents = handle_multiple_files_workflow()
    elif workflow_option == "Troubleshooting":
        handle_troubleshooting_workflow()
    else:
        st.error("Invalid workflow option selected.")
        return [], [], workflow_option, None

    template_file = st.file_uploader(
        "Upload a template document (optional)",
        type=ALLOWED_FILE_TYPES
    )

    st.info("""
    You can enable Perplexity research in the sidebar to enhance your KB articles with up-to-date information. 
    When enabled, the system will use Perplexity AI to gather additional relevant data before generating the final article.
    """)

    return kb_article_titles, kb_article_contents, workflow_option, template_file

def handle_single_file_workflow() -> Tuple[List[str], List[str]]:
    """
    Handle the single file workflow for KB article generation.

    Returns:
        Tuple containing a list with a single KB article title and a list with a single KB article content.
    """
    kb_article_title = st.text_input('Input a Title for the KB Article')
    kb_article_content = st.text_area('Input the Article Content', height=200)

    uploaded_file = st.file_uploader(
        "Upload a file to import the text into the content field below",
        type=ALLOWED_FILE_TYPES
    )

    if uploaded_file is not None:
        file_content = file_handlers.load_file_content(uploaded_file)
        if file_content:
            kb_article_title = uploaded_file.name
            kb_article_content = file_content
            st.success(f"File '{uploaded_file.name}' loaded successfully!")
        else:
            st.error("Failed to load the file content. Please try again.")

    return [kb_article_title], [kb_article_content]

def handle_multiple_files_workflow() -> Tuple[List[str], List[str]]:
    """
    Handle the multiple files workflow for KB article generation.

    Returns:
        Tuple containing a list of KB article titles and a list of KB article contents.
    """
    uploaded_files = st.file_uploader(
        "Upload files to process",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True
    )

    kb_article_titles: List[str] = []
    kb_article_contents: List[str] = []

    if uploaded_files:
        for file in uploaded_files:
            content = file_handlers.load_file_content(file)
            if content:
                kb_article_titles.append(file.name)
                kb_article_contents.append(content)

        st.success(f"Number of files loaded successfully: {len(kb_article_titles)}")
    else:
        st.info("Please upload one or more files to process.")

    return kb_article_titles, kb_article_contents

def handle_troubleshooting_workflow() -> None:
    """
    Handle the troubleshooting workflow.
    """
    st.subheader("Troubleshooting Chat")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's your question?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in asyncio.run(handle_chat_message(prompt, st.session_state.selected_model, st.session_state.options, st.session_state.chat_history)):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.button("Convert to KB Article"):
        with st.spinner("Converting chat to KB article..."):
            kb_article = asyncio.run(convert_chat_to_kb_article(st.session_state.chat_history, st.session_state.selected_model, st.session_state.options))
            st.session_state.kb_article = kb_article
            st.success("KB Article generated successfully!")
            with st.expander("View Generated KB Article", expanded=True):
                st.markdown(kb_article)

    if st.button("Convert to Resolution Steps"):
        with st.spinner("Converting chat to resolution steps..."):
            resolution_steps = asyncio.run(convert_chat_to_resolution_steps(st.session_state.chat_history, st.session_state.selected_model, st.session_state.options))
            st.session_state.resolution_steps = resolution_steps
            st.success("Resolution Steps generated successfully!")
            with st.expander("View Generated Resolution Steps", expanded=True):
                st.markdown(resolution_steps)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    setup_main_content()
