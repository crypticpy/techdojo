# layout/sidebar.py

import streamlit as st
from config import LANGUAGE_OPTIONS, CONSISTENCY_OPTIONS
from typing import Dict, Any

def setup_sidebar(workflow_option: str) -> Dict[str, Any]:
    """
    Set up the sidebar with various options for KB article generation and processing.

    Args:
        workflow_option (str): The current workflow option selected by the user.

    Returns:
        Dict[str, Any]: A dictionary containing all the selected options.
    """
    options: Dict[str, Any] = {}

    with st.sidebar:
        st.subheader('KB Article Options')
        options['Actions'] = st.multiselect(
            'Select Actions',
            ['Generate Content', 'Reauthor Content', 'Format to Template', 'Translate'],
            default=['Generate Content']
        )

        options['Translation Languages'] = st.multiselect(
            'Translation Languages',
            LANGUAGE_OPTIONS,
            default=['English']
        )

        options['Perspective'] = st.multiselect(
            'Perspective Sections',
            ['User', 'Support Analyst', 'Administrator']
        )

        if workflow_option == "Multiple Files":
            st.subheader('Consistency Options')
            options['Consistency'] = st.multiselect(
                'Ensure Consistency Across Articles',
                CONSISTENCY_OPTIONS
            )

        st.subheader('Research Options')
        options['Use Perplexity Research'] = st.checkbox('Use Perplexity for Up-to-date Research', value=False)
        if options['Use Perplexity Research']:
            options['Max Research Iterations'] = st.slider(
                'Max Research Iterations',
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of iterations for Perplexity research"
            )

        st.subheader('Advanced Options')
        options['Temperature'] = st.slider(
            'Temperature',
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in generation. Lower values make output more focused and deterministic."
        )

        options['Max Tokens'] = st.number_input(
            'Max Tokens',
            min_value=100,
            max_value=8192,
            value=3000,
            step=100,
            help="Maximum number of tokens to generate. One token is roughly 4 characters for normal English text."
        )

        if options['Use Perplexity Research']:
            st.subheader('Perplexity API Options')
            options['Perplexity Model'] = st.selectbox(
                'Perplexity Model',
                ['llama-3.1-sonar-small-128k-online', 'llama-3.1-sonar-medium-128k-online'],
                index=0,
                help="Select the Perplexity model to use for research"
            )
            options['Perplexity Max Tokens'] = st.number_input(
                'Perplexity Max Tokens',
                min_value=100,
                max_value=4096,
                value=1024,
                step=100,
                help="Maximum number of tokens for Perplexity API responses"
            )
            options['Perplexity Temperature'] = st.slider(
                'Perplexity Temperature',
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Controls randomness in Perplexity API responses"
            )

    return options
