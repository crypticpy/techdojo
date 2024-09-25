import io
import zipfile
from datetime import datetime
import streamlit as st
import docx2txt
from PyPDF2 import PdfReader
from config import FILE_LOAD_ERROR_MESSAGE, UNSUPPORTED_FILE_TYPE_MESSAGE
from typing import Union

@st.cache_data
def load_file_content(uploaded_file: Union[st.runtime.uploaded_file_manager.UploadedFile, None], is_template: bool = False) -> str:
    """
    Load the content of the uploaded file based on its type.

    Args:
        uploaded_file: The uploaded file object.
        is_template: Flag indicating if the file is a template. Default is False.

    Returns:
        The content of the file as a string.
    """
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "application/pdf":
            return extract_pdf_content(uploaded_file, is_template)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_docx_content(uploaded_file, is_template)
        elif uploaded_file.type in ["text/plain", "application/octet-stream"]:
            return uploaded_file.read().decode("utf-8")
        else:
            st.error(UNSUPPORTED_FILE_TYPE_MESSAGE.format(uploaded_file.type))
            return ""
    except Exception as e:
        st.error(FILE_LOAD_ERROR_MESSAGE.format(str(e)))
        return ""

def extract_pdf_content(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, is_template: bool) -> str:

    """
    Extract content from a PDF file.

    Args:
        uploaded_file: The uploaded PDF file object.
        is_template: Flag indicating if the file is a template.

    Returns:
        The extracted text content as a string.
    """
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    content = []
    for page in pdf_reader.pages:
        content.append(page.extract_text())
    return '\n'.join(content)

def extract_docx_content(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, is_template: bool) -> str:
    """
    Extract content from a DOCX file.

    Args:
        uploaded_file: The uploaded DOCX file object.
        is_template: Flag indicating if the file is a template.

    Returns:
        The extracted text content as a string.
    """
    return docx2txt.process(io.BytesIO(uploaded_file.read()))

def create_download_button(docx_bytes: io.BytesIO, title: str, workflow_option: str) -> None:
    """
    Create a download button for the generated KB article(s).

    Args:
        docx_bytes: The Word document as bytes.
        title: The title of the KB article.
        workflow_option: The selected workflow option (single file or multiple files).
    """
    if workflow_option == "Single File":
        st.download_button(
            label="Download KB Article as Word",
            data=docx_bytes,
            file_name=f"{title}_{datetime.now().strftime('%Y%m%d')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(f"{title}.docx", docx_bytes.getvalue())
        zip_buffer.seek(0)
        st.download_button(
            label="Download KB Article as ZIP",
            data=zip_buffer,
            file_name=f"kb_article_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
        )
        