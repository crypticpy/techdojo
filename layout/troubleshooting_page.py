# layout/troubleshooting_page.py

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, List
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler

from utils.insight_tool import InSightKB
from utils.perplexity_tool import PerplexitySearchTool
from utils import pandoc_utils, file_handlers
from core_functions import convert_chat_to_kb_article, convert_chat_to_resolution_steps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_agent(model: str, temperature: float, incident_context: str):
    """
    Set up the Langchain agent with the necessary tools and memory.
    """
    try:
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=2000
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        system_message = SystemMessage(content=f"""
        You are an AI assistant specializing in technical support for IT issues. Your role is to provide detailed, 
        step-by-step instructions for troubleshooting hardware and software problems or providing guidance how to handle trouble tickets.

        When responding:
        1. Always provide specific, actionable steps that a technician can follow while instructing a user.
        2. Include potential outcomes for each step and what to do next based on those outcomes.
        3. Use technical language appropriate for IT support, but ensure it's clear enough for end-users to understand if needed.
        4. If a step involves potential risks (e.g., data loss, hardware damage), clearly state these risks and any precautions to take.
        5. When relevant, suggest diagnostic tools or commands that can provide more information about the issue.
        6. If a step doesn't resolve the issue, always provide the next troubleshooting step or escalation path.

        We are working on the following case: {incident_context}

        You have access to the following tools:
        - Perplexity Search: Use this to look up current information or specific technical details about hardware, software, or troubleshooting procedures.
        - InSightKB: Use this tool to search our internal knowledge base for specific resolutions, policies, processes, SOPS, or other information related to the incident that would be specific to our clients who all work at the city of austin.

        Maintain context throughout the conversation, build upon previous responses, and avoid unnecessary repetition.
        """)

        memory.chat_memory.add_message(system_message)

        perplexity_tool = PerplexitySearchTool()
        insight_kb_tool = InSightKB(base_url="http://localhost:8004")
        tools = [perplexity_tool, insight_kb_tool]

        logger.info(f"Tools initialized: {[tool.name for tool in tools]}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful AI assistant helping support staff resolve incidents for the City of Austin."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(
                content="Provide a detailed, step-by-step response to the following technical support question or issue: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={"prompt": prompt}
        )

        logger.info(f"Agent initialized with tools: {[tool.name for tool in agent.tools]}")

        return agent
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        raise

async def handle_user_input(agent, user_input: str):
    try:
        with st.spinner("AI is thinking..."):
            streamlit_handler = StreamlitCallbackHandler(st.container())
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.run,
                    user_input,
                    callbacks=[streamlit_handler]
                ),
                timeout=60
            )

        logger.info(f"AI response: {response}")
        return response
    except asyncio.TimeoutError:
        logger.error("AI response timed out")
        return "I apologize, but it's taking longer than expected to process your request. Please try again or rephrase your question."
    except Exception as e:
        logger.error(f"Error in AI response: {str(e)}", exc_info=True)
        return f"I apologize, but I encountered an error while processing your request. Error details: {str(e)}. Please try rephrasing your question or contact support if the issue persists."


def display_troubleshooting_page(ticket_data: Dict[str, Any], prediction_result: Dict[str, Any]):
    st.write("## Troubleshooting Chat")

    with st.expander("Incident Information", expanded=False):
        st.json(ticket_data)

    st.write("### Predicted Assignment Group")
    top_prediction = "No prediction available"
    if prediction_result and 'output' in prediction_result:
        predictions = prediction_result['output']
        top_prediction = max(predictions, key=predictions.get)
    st.write(f"Top predicted group: {top_prediction}")

    incident_context = f"""
    Incident details: {ticket_data}
    Predicted Assignment Group: {top_prediction}
    """

    # Always initialize the agent if it doesn't exist or if it's None
    if 'agent' not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = setup_agent(
            model=st.session_state.get('selected_model', 'gpt-4o'),
            temperature=st.session_state.get('options', {}).get('Temperature', 0.1),
            incident_context=incident_context
        )

    # Display chat history
    if hasattr(st.session_state.agent, 'memory') and hasattr(st.session_state.agent.memory, 'chat_memory'):
        for message in st.session_state.agent.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    else:
        st.warning("Chat history is not available.")

    if prompt := st.chat_input("What would you like to know?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = asyncio.run(handle_user_input(st.session_state.agent, prompt))
            st.markdown(response)

    # Create two columns for KB Article and Resolution Steps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KB Article")
        if st.button("Generate KB Article"):
            with st.spinner("Generating KB Article..."):
                if hasattr(st.session_state.agent, 'memory') and hasattr(st.session_state.agent.memory, 'chat_memory'):
                    kb_article = asyncio.run(convert_chat_to_kb_article(st.session_state.agent.memory.chat_memory.messages, st.session_state.get('selected_model', 'gpt-4o'), {}))
                    st.session_state.kb_article = kb_article
                    st.success("KB Article generated successfully!")
                else:
                    st.error("Unable to generate KB Article: Chat history is not available.")

        if 'kb_article' in st.session_state and st.session_state.kb_article:
            with st.expander("View Generated KB Article", expanded=True):
                st.markdown(st.session_state.kb_article)
            if st.button("Download KB Article as Word"):
                docx_bytes = pandoc_utils.save_as_word(st.session_state.kb_article)
                file_handlers.create_download_button(docx_bytes, "KB_Article", "Single File")

    with col2:
        st.subheader("Resolution Steps")
        generate_steps_button = st.button("Generate Resolution Steps")

        if generate_steps_button:
            if hasattr(st.session_state.agent, 'memory') and hasattr(st.session_state.agent.memory,
                                                                     'chat_memory') and st.session_state.agent.memory.chat_memory.messages:
                with st.spinner("Generating Resolution Steps..."):
                    # Convert the chat memory to a list of dictionaries
                    chat_history = [
                        {"role": "system",
                         "content": "You are an AI assistant that extracts resolution steps from support conversations."}
                    ]
                    for message in st.session_state.agent.memory.chat_memory.messages:
                        if isinstance(message, HumanMessage):
                            chat_history.append({"role": "user", "content": message.content})
                        elif isinstance(message, AIMessage):
                            chat_history.append({"role": "assistant", "content": message.content})

                    # Add the incident details to the chat history
                    chat_history.insert(1, {
                        "role": "system",
                        "content": f"Incident details: {ticket_data}\nPredicted Assignment Group: {top_prediction}"
                    })

                    resolution_steps = asyncio.run(convert_chat_to_resolution_steps(
                        chat_history,
                        st.session_state.get('selected_model', 'gpt-4o'),
                        {}
                    ))
                    st.session_state.resolution_steps = resolution_steps
                    st.success("Resolution Steps generated successfully!")
            else:
                st.warning(
                    "Unable to generate Resolution Steps: Chat history is not available. Please start a conversation first.")

        if 'resolution_steps' in st.session_state and st.session_state.resolution_steps:
            with st.expander("View Generated Resolution Steps", expanded=True):
                st.markdown(st.session_state.resolution_steps)

            # Add a download button for the resolution steps
            st.download_button(
                label="Download Resolution Steps",
                data=st.session_state.resolution_steps,
                file_name="Resolution_Steps.txt",
                mime="text/plain"
            )

    if st.button("Reset Conversation"):
        if hasattr(st.session_state.agent, 'memory'):
            st.session_state.agent.memory.clear()
        if 'resolution_steps' in st.session_state:
            del st.session_state.resolution_steps
        st.session_state.kb_article = None
        st.rerun()

    # Auto-scroll script
    st.components.v1.html(
        """
        <script>
            var element = window.parent.document.querySelector('section.main');
            element.scrollTop = element.scrollHeight;
        </script>
        """,
        height=0
    )
