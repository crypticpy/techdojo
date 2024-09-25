import os
import tempfile
from io import BytesIO
import pypandoc
import streamlit as st

def clean_markdown(text: str) -> str:
    """
    Clean and sanitize the Markdown text to avoid issues during conversion.

    Args:
        text: The markdown text to clean.

    Returns:
        The cleaned markdown text.
    """
    # Remove any potential YAML frontmatter
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            text = parts[2]
    return text.strip()

def ensure_pandoc():
    """
    Ensure Pandoc is installed for document conversion.
    If not installed, attempt to download it.
    """
    try:
        pypandoc.get_pandoc_path()
    except OSError:
        st.warning("Pandoc not found. Attempting to download...")
        try:
            pypandoc.download_pandoc()
            st.success("Pandoc has been successfully downloaded.")
        except Exception as e:
            st.error(f"Failed to download Pandoc: {str(e)}")
            st.error("Please install Pandoc manually to enable document conversion.")

@st.cache_data
def save_as_word(markdown_content: str) -> BytesIO:
    """
    Convert the given Markdown content to a Word document and return a BytesIO object.

    Args:
        markdown_content: The Markdown content to convert.

    Returns:
        A BytesIO object containing the Word document.
    """
    cleaned_content = clean_markdown(markdown_content)

    try:
        # Create a temporary file to store the output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            output_path = tmp_file.name

        # Check if reference.docx exists
        if not os.path.exists('reference.docx'):
            st.warning("reference.docx not found. Using Pandoc's default Word styling.")
            extra_args = []
        else:
            extra_args = ['--reference-doc=reference.docx']

        # Convert markdown to docx using pandoc with reference.docx if available
        pypandoc.convert_text(
            cleaned_content,
            'docx',
            format='markdown',
            outputfile=output_path,
            extra_args=extra_args
        )

        # Read the contents of the output file
        with open(output_path, 'rb') as docx_file:
            docx_bytes = BytesIO(docx_file.read())

        # Remove the temporary file
        os.unlink(output_path)

        return docx_bytes
    except (OSError, RuntimeError) as e:
        st.error(f"Pandoc conversion error: {str(e)}")
        return None

def markdown_to_html(text: str) -> str:
    """
    Convert the given Markdown text to HTML.

    Args:
        text: The markdown text to convert.

    Returns:
        The HTML representation of the markdown text.
    """
    try:
        return pypandoc.convert_text(text, 'html', format='markdown')
    except RuntimeError as e:
        st.error(f"Markdown to HTML conversion error: {str(e)}")
        return text  # Return original text if conversion fails

def get_pandoc_version() -> str:
    """
    Get the version of Pandoc being used.

    Returns:
        The Pandoc version as a string.
    """
    return pypandoc.get_pandoc_version()
