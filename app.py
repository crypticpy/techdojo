# app.py

import asyncio
import streamlit as st
from config import TITLE_STYLE
from utils import file_handlers, pandoc_utils
from utils.stats_display import display_perplexity_stats
from layout import main_content, sidebar, incident_lookup, troubleshooting_page
from core_functions import (
    process_single_article,
    process_multiple_articles,
    display_results,
    convert_chat_to_kb_article,
    convert_chat_to_resolution_steps
)
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize all necessary session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = []

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Incident Lookup"

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4o"

    if 'options' not in st.session_state:
        st.session_state.options = {
            "Temperature": 0.1,
            "Max Tokens": 2000,
            "Perplexity Model": "llama-3.1-sonar-small-128k-online",
            "Perplexity Max Tokens": 1024,
            "Perplexity Temperature": 0.2,
            "Use Perplexity Research": True,
            "Max Research Iterations": 5,
            "Actions": ["Generate Content"],
            "Translation Languages": ["English"],
            "Perspective": [],
            "Consistency": []
        }

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'kb_article' not in st.session_state:
        st.session_state.kb_article = None

    if 'resolution_steps' not in st.session_state:
        st.session_state.resolution_steps = None

    if 'agent' not in st.session_state:
        st.session_state.agent = None

    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "contact_type": "",
            "requested_for_title": "",
            "requested_for_department": "",
            "requested_for_location": "",
            "category": "",
            "sub_category": "",
            "priority": "3 - Moderate",
            "description": "",
            "extract_product": "",
            "summary": ""
        }

    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def main():
    st.set_page_config(page_title="Tech Dojo", layout="wide")

    st.markdown('<div class="kb-dojo-title">Tech Dojo</div>', unsafe_allow_html=True)
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)

    initialize_session_state()

    # Navigation
    pages = ["Incident Lookup", "Troubleshooting", "KB Article Generation"]
    st.session_state.current_page = st.sidebar.radio("Navigation", pages,
                                                     index=pages.index(st.session_state.current_page))

    if st.session_state.current_page == "Incident Lookup":
        incident_lookup.display_incident_lookup_page()
    elif st.session_state.current_page == "Troubleshooting":
        if 'form_data' in st.session_state:
            troubleshooting_page.display_troubleshooting_page(
                st.session_state.form_data,
                st.session_state.get('prediction_result', None)
            )
        else:
            st.warning("Please enter incident data before accessing the troubleshooting page.")
            st.session_state.current_page = "Incident Lookup"
            st.rerun()
    else:  # KB Article Generation
        asyncio.run(handle_kb_article_generation())

async def handle_kb_article_generation():
    """Handle the KB Article Generation page."""
    col1, col2 = st.columns([2, 1])

    with col1:
        kb_article_titles, kb_article_contents, workflow_option, template_file = main_content.setup_main_content()

    with col2:
        options = sidebar.setup_sidebar(workflow_option)

    # Update options in session state
    st.session_state.options.update(options)

    if template_file:
        template_content = file_handlers.load_file_content(template_file, is_template=True)
    else:
        template_content = None

    if st.button('Generate KB Article'):
        logger.debug(f"Workflow option: {workflow_option}")
        logger.debug(f"Selected model: {st.session_state.selected_model}")
        logger.debug(f"Selected options: {st.session_state.options}")

        if workflow_option == "Single File":
            logger.debug(f"Single file content: {kb_article_contents[0][:100]}...")
            st.session_state.results = await process_single_article(kb_article_titles[0], kb_article_contents[0], st.session_state.options,
                                       st.session_state.selected_model,
                                       template_content)
        elif workflow_option == "Multiple Files":
            logger.debug(f"Multiple files, number of files: {len(kb_article_titles)}")
            st.session_state.results = await process_multiple_articles(kb_article_titles, kb_article_contents, st.session_state.options,
                                          st.session_state.selected_model,
                                          template_content)

        display_results(st.session_state.results)

    # Add a section to display research results if the option is enabled
    if st.session_state.options.get('Use Perplexity Research', False):
        st.subheader("Research Results")
        for result in st.session_state.results:
            title, content = result[1], result[2]
            with st.expander(f"Research for {title}"):
                st.markdown(content)

    # Handle KB article and resolution steps conversion
    if st.button("Convert Chat to KB Article"):
        with st.spinner("Converting chat to KB article..."):
            kb_article = await convert_chat_to_kb_article(
                st.session_state.chat_history,
                st.session_state.selected_model,
                st.session_state.options
            )
            st.session_state.kb_article = kb_article
            st.success("KB Article generated successfully!")
            with st.expander("View Generated KB Article", expanded=True):
                st.markdown(kb_article)

    if st.button("Convert Chat to Resolution Steps"):
        with st.spinner("Converting chat to resolution steps..."):
            resolution_steps = await convert_chat_to_resolution_steps(
                st.session_state.chat_history,
                st.session_state.selected_model,
                st.session_state.options
            )
            st.session_state.resolution_steps = resolution_steps
            st.success("Resolution Steps generated successfully!")
            with st.expander("View Generated Resolution Steps", expanded=True):
                st.markdown(resolution_steps)

    # Download buttons for KB article and resolution steps
    if st.session_state.kb_article:
        docx_bytes = pandoc_utils.save_as_word(st.session_state.kb_article)
        st.download_button(
            label="Download KB Article as Word",
            data=docx_bytes.getvalue(),
            file_name="KB_Article.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    if st.session_state.resolution_steps:
        st.download_button(
            label="Download Resolution Steps as Text",
            data=st.session_state.resolution_steps,
            file_name="Resolution_Steps.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    pandoc_utils.ensure_pandoc()
    main()
