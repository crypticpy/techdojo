# utils/display_utils.py

import streamlit as st

def display_research_progress(iteration: int, query: str, summary: str):
    """
    Display the progress of the Perplexity research process.

    Args:
        iteration (int): The current iteration number.
        query (str): The query sent to Perplexity.
        summary (str): The summary of the research results.
    """
    with st.expander(f"Research Iteration {iteration}", expanded=False):
        st.write(f"Query: {query}")
        st.write("Summary:")
        st.write(summary)