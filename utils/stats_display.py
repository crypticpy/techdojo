# utils/stats_display.py

import streamlit as st

def display_perplexity_stats(query_count: int, total_tokens: int = None):
    """
    Display Perplexity API usage statistics in the sidebar.

    Args:
        query_count (int): The number of queries made to Perplexity API.
        total_tokens (int, optional): The total number of tokens used in Perplexity API calls.
    """
    with st.sidebar:
        st.subheader("Perplexity API Usage")
        st.write(f"Queries made: {query_count}")
        if total_tokens is not None:
            st.write(f"Total tokens used: {total_tokens}")
        else:
            st.write("Token usage information not available")
