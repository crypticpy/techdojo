# core_functions.py

import asyncio
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import OPENAI_API_KEY, DEFAULT_PROMPT_ROLE
from utils import file_handlers, prompt_utils, pandoc_utils, perplexity_utils
from utils.display_utils import display_research_progress
from utils.stats_display import display_perplexity_stats
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from utils.perplexity_tool import PerplexitySearchTool
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import logging
from typing import List, Tuple, Dict, Generator, Any, Optional
import io
import zipfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def send_to_openai_api_async(prompt: str, selected_model: str, options: dict) -> str:
    """Sends a prompt to the OpenAI API using Langchain."""
    chat = ChatOpenAI(
        model_name=selected_model,
        temperature=options.get("Temperature", 0.1),
        max_tokens=options.get("Max Tokens", 2000),
        openai_api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=DEFAULT_PROMPT_ROLE),
        HumanMessage(content=prompt)
    ]

    response = await chat.ainvoke(messages)
    return response.content.strip()


async def advanced_iterative_research(title: str, content: str, selected_model: str, options: dict,
                                      max_iterations: int = 5) -> str:
    research_summary = f"Title: {title}\nInitial Content: {content}\n\nResearch Findings:\n"
    iteration = 0
    query_count = 0

    chat = ChatOpenAI(
        model_name=selected_model,
        temperature=options.get("Temperature", 0.1),
        max_tokens=options.get("Max Tokens", 2000),
        openai_api_key=OPENAI_API_KEY
    )

    perplexity_tool = PerplexitySearchTool()

    system_message = """You are an advanced AI research assistant. Your task is to gather comprehensive information on a given topic through iterative research. Follow these steps for each iteration:

1. Analyze the current research summary and identify gaps or areas that need more information.
2. Formulate a specific, concise query (max 15 words) to address one of these gaps or explore a new aspect of the topic.
3. After receiving search results, analyze them in the context of existing information.
4. Provide a brief summary of new, relevant findings (max 100 words).
5. Decide whether to continue research or conclude if sufficient information has been gathered.

Ensure each query is distinct from previous ones and targets unexplored aspects of the topic."""

    while iteration < max_iterations:
        with st.expander(f"Research Iteration {iteration + 1}", expanded=False):
            # Formulate query
            query_prompt = f"{system_message}\n\nCurrent research summary:\n{research_summary}\n\nFormulate the next query or conclude the research."
            query_response = await chat.ainvoke([HumanMessage(content=query_prompt)])
            query = query_response.content.strip()

            if "SUFFICIENT" in query.upper():
                st.write("Research completed.")
                break

            st.write(f"Query: {query}")

            # Execute Perplexity search
            search_results = await perplexity_tool._arun(query)

            # Analyze results
            analysis_prompt = f"""Analyze the following search results in the context of our existing research:
            {search_results}

            Provide a concise summary of new, relevant findings (max 100 words). Focus on information not already present in our research summary.
            Current Research Summary: {research_summary}"""

            analysis_response = await chat.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis = analysis_response.content.strip()

            st.write("Summary of new findings:")
            st.write(analysis)

            research_summary += f"\nIteration {iteration + 1}:\nQuery: {query}\nFindings: {analysis}\n"

        iteration += 1
        query_count += 1

    display_perplexity_stats(query_count)

    return research_summary

# Update the research_and_process_article function to use the new advanced_iterative_research
async def research_and_process_article(title: str, content: str, options: dict, selected_model: str,
                                       template_content: str = None) -> List[Dict]:
    logger.debug(f"Researching and processing article: {title}")

    research_summary = await advanced_iterative_research(title, content, selected_model, options,
                                                         options.get('Max Research Iterations', 5))

    final_prompt = f"""Using the original content, template (if provided), and the researched information, 
    create a comprehensive and up-to-date KB article:

    Title: {title}
    Original Content: {content}
    Template: {template_content}
    Researched Information: {research_summary}

    Provide the final KB article in markdown format, incorporating the most relevant and up-to-date information 
    from both the original content and the research summary. Ensure the article is well-structured, 
    comprehensive, and follows best practices for knowledge base articles."""

    final_content = await send_to_openai_api_async(final_prompt, selected_model, options)

    return await process_article_async(title, final_content, options, selected_model, template_content)



async def process_article_async(title: str, content: str, options: dict, selected_model: str,
                                template_content: str = None) -> List[Dict]:
    logger.debug(f"Processing article: {title}")
    logger.debug(f"Initial content: {content[:100]}...")

    results = []
    original_content = content
    processed_content = original_content

    actions = options['Actions']
    perspectives = options['Perspective']
    languages = options['Translation Languages']

    if 'Generate Content' in actions:
        processed_content = await generate_content(title, content, perspectives, selected_model, options)
        logger.debug(f"Content after generation: {processed_content[:100]}...")
    elif 'Reauthor Content' in actions or 'Format to Template' in actions:
        processed_content = await reauthor_content(processed_content, template_content, perspectives, selected_model,
                                                   options)
        logger.debug(f"Content after reauthoring and/or formatting: {processed_content[:100]}...")

    if perspectives and 'Reauthor Content' not in actions:
        processed_content = prompt_utils.create_perspective_sections(processed_content, perspectives)
        logger.debug(f"Content after applying perspectives: {processed_content[:100]}...")

    if 'Translate' in actions:
        for lang in languages:
            translated_content = await translate_content(processed_content, lang, selected_model, options)
            logger.debug(f"Translated content ({lang}): {translated_content[:100]}...")

            docx_bytes = pandoc_utils.save_as_word(translated_content)

            results.append((docx_bytes, f"{title}_{lang}", translated_content, lang))
    else:
        docx_bytes = pandoc_utils.save_as_word(processed_content)
        results.append((docx_bytes, title, processed_content, 'Original'))

    return results

async def generate_content(title: str, content: str, perspectives: List[str], selected_model: str,
                           options: dict) -> str:
    logger.debug("Generating content")
    prompt = f"Generate a detailed knowledge base article based on the following title and content. If the content is a specific request (e.g., 'write a recipe for chocolate cake'), create an article that fulfills that request. Title: {title}\n\nContent or Request: {content}\n\n"

    if perspectives:
        prompt += f"Include separate sections for the following perspectives: {', '.join(perspectives)}.\n\n"

    prompt += "Provide the generated content in markdown format, without any additional comments or questions."

    generated_content = await send_to_openai_api_async(prompt, selected_model, options)
    if generated_content is None or generated_content.strip() == "":
        logger.error("Content generation failed or returned empty content")
        return content
    return generated_content

async def reauthor_content(content: str, template_content: str, perspectives: List[str], selected_model: str,
                           options: dict) -> str:
    logger.debug("Reauthoring content")
    prompt = f"Reauthor the following content"

    if not content.strip():
        prompt = "Generate a knowledge base article based on the following template:"

    if template_content:
        prompt += f", using the provided template as a guide for formatting:\n\nTemplate:\n{template_content}\n\nContent to reauthor:"
    else:
        prompt += ", maintaining its core information but improving its clarity, structure, and readability:"

    prompt += f"\n\n{content}\n\n"

    if perspectives:
        prompt += f"Include separate sections for the following perspectives: {', '.join(perspectives)}.\n\n"

    prompt += "Provide the reauthored content directly, without any additional comments or questions."

    reauthored_content = await send_to_openai_api_async(prompt, selected_model, options)
    if reauthored_content is None or reauthored_content.strip() == "":
        logger.error("Reauthoring failed or returned empty content")
        return content
    return reauthored_content

async def format_to_template(content: str, template: str, selected_model: str, options: dict) -> str:
    logger.debug("Formatting content to template")
    prompt = f"Format the following content to match the provided template structure, maintaining the original information:\n\nContent:\n{content}\n\nTemplate:\n{template}"
    return await send_to_openai_api_async(prompt, selected_model, options)

async def translate_content(content: str, language: str, selected_model: str, options: dict) -> str:
    logger.debug(f"Translating content to {language}")
    prompt = f"Translate the following content to {language}, maintaining its original structure and formatting. Provide the translated content directly, without any additional comments or questions:\n\n{content}"
    return await send_to_openai_api_async(prompt, selected_model, options)

async def process_multiple_articles_async(titles: List[str], contents: List[str], options: dict, selected_model: str,
                                          template_content: str) -> List[Dict]:
    all_results = []
    total_operations = len(titles) * (len(options['Actions']) + (
        len(options['Translation Languages']) if 'Translate' in options['Actions'] else 0))
    progress_bar = st.progress(0)
    completed_operations = 0

    for title, content in zip(titles, contents):
        logger.debug(f"Processing article: {title}")
        if options.get('Use Perplexity Research', False):
            results = await research_and_process_article(title, content, options, selected_model, template_content)
        else:
            results = await process_article_async(title, content, options, selected_model, template_content)
        all_results.extend(results)
        completed_operations += len(options['Actions']) + (
            len(options['Translation Languages']) if 'Translate' in options['Actions'] else 0)
        progress_bar.progress(completed_operations / total_operations)

    return all_results

async def process_single_article(title: str, content: str, options: dict, selected_model: str, template_content: str) -> \
List[Dict]:
    with st.spinner('Generating KB article...'):
        if options.get('Use Perplexity Research', False):
            return await research_and_process_article(title, content, options, selected_model, template_content)
        else:
            return await process_article_async(title, content, options, selected_model, template_content)

async def process_multiple_articles(titles: List[str], contents: List[str], options: dict, selected_model: str,
                                    template_content: str) -> List[Dict]:
    with st.spinner('Generating KB articles...'):
        return await process_multiple_articles_async(titles, contents, options, selected_model, template_content)

def display_results(results: List[Dict]):
    st.subheader("Generated Articles")
    for i, (docx_bytes, title, api_response, lang) in enumerate(results):
        with st.expander(f"{title} - {lang}"):
            st.markdown(f'<div class="generated-content">{api_response}</div>', unsafe_allow_html=True)
            if docx_bytes is not None:
                st.download_button(
                    label=f"Download {title} ({lang}) as Word",
                    data=docx_bytes.getvalue(),
                    file_name=f"{title}_{lang}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.error(f"Failed to generate Word document for {title} ({lang})")

    if len(results) > 1:
        create_zip_download(results)

def create_zip_download(results: List[Dict]):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for docx_bytes, title, _, lang in results:
            zip_file.writestr(f"{title}_{lang}.docx", docx_bytes.getvalue())

    st.download_button(
        label="Download All Articles as ZIP",
        data=zip_buffer.getvalue(),
        file_name="kb_articles.zip",
        mime="application/zip",
    )

async def handle_chat_message(user_message: str, selected_model: str, options: Dict, chat_history: List[Dict]) -> str:
    try:
        chat = ChatOpenAI(
            model_name=selected_model,
            temperature=options.get("Temperature", 0.1),
            max_tokens=options.get("Max Tokens", 2000),
            openai_api_key=OPENAI_API_KEY
        )

        perplexity_tool = PerplexitySearchTool()
        tools = [perplexity_tool]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Convert chat history to Langchain message format
        for message in chat_history:
            if message["role"] == "user":
                memory.chat_memory.add_message(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                memory.chat_memory.add_message(AIMessage(content=message["content"]))

        agent = initialize_agent(
            tools,
            chat,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

        try:
            response = await asyncio.wait_for(agent.arun(user_message), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("AI response timed out")
            return "I apologize, but it's taking longer than expected to process your request. Please try again or rephrase your question."

        return response
    except Exception as e:
        logger.error(f"Error in handle_chat_message: {str(e)}", exc_info=True)
        return f"I apologize, but I encountered an error while processing your request. Error details: {str(e)}. Please try rephrasing your question or contact support if the issue persists."

async def convert_chat_to_kb_article(chat_history: List[Dict], selected_model: str, options: Dict) -> str:
    chat = ChatOpenAI(
        model_name=selected_model,
        temperature=options.get("Temperature", 0.1),
        max_tokens=options.get("Max Tokens", 2000),
        openai_api_key=OPENAI_API_KEY
    )

    formatted_messages = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_messages.append(HumanMessage(content=message.content))
        elif isinstance(message, AIMessage):
            formatted_messages.append(AIMessage(content=message.content))

    system_message = SystemMessage(
        content="You are an AI assistant that creates knowledge base articles from conversations.")

    prompt = HumanMessage(content="""
    Convert the following conversation into a knowledge base article, focusing on the troubleshooting steps and resolution.
    Provide the KB article in markdown format, following best practices for knowledge base articles.
    Include a title, summary, problem description, troubleshooting steps, and resolution.
    """)

    messages = [system_message] + formatted_messages + [prompt]

    response = await chat.ainvoke(messages)
    return response.content.strip()


async def convert_chat_to_resolution_steps(chat_history: List[Dict[str, str]], selected_model: str,
                                           options: Dict[str, Any]) -> str:
    logger.info(f"Converting chat to resolution steps. Chat history length: {len(chat_history)}")

    chat = ChatOpenAI(
        model_name=selected_model,
        temperature=options.get("Temperature", 0.1),
        max_tokens=options.get("Max Tokens", 2000),
        openai_api_key=OPENAI_API_KEY
    )

    system_message = SystemMessage(
        content="You are an AI assistant that extracts resolution steps from ServiceNow support conversations. Provide a formatted response that matches Service Now best practices for incident resolution notes. The incident metadata is for your knowledge and does not need to be repeated in the notes "
    )

    messages = [system_message] + [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else
        AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
        SystemMessage(content=msg["content"])
        for msg in chat_history
    ]

    messages.append(HumanMessage(
        content="Extract the key resolution steps from this conversation and format them as Service Now resolution notes."))

    logger.info(f"Sending request to OpenAI. Number of messages: {len(messages)}")

    response = await chat.ainvoke(messages)

    logger.info(f"Received response from OpenAI. Response length: {len(response.content)}")

    return response.content.strip()


async def advanced_iterative_research(title: str, content: str, selected_model: str, options: dict,
                                      max_iterations: int = 5) -> str:
    research_context = f"Title: {title}\nInitial Content: {content}\n\nResearch Findings:\n"
    iteration = 0
    query_count = 0

    chat = ChatOpenAI(
        model_name=selected_model,
        temperature=options.get("Temperature", 0.1),
        max_tokens=options.get("Max Tokens", 2000),
        openai_api_key=OPENAI_API_KEY
    )

    perplexity_tool = PerplexitySearchTool()
    tools = [perplexity_tool]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an advanced AI research assistant. Your task is to gather comprehensive information on a given topic by formulating queries, analyzing search results, and building context. Follow these steps:

1. Analyze the current research context.
2. Formulate a specific, concise query (max 15 words) for the Perplexity search engine to gather new, relevant information.
3. Analyze the search results in the context of your existing knowledge.
4. Summarize key findings, focusing on new or updated information.
5. Decide whether to continue research with a new query or conclude if sufficient information has been gathered.

Provide your response in the following format:
QUERY: [Your formulated query]
ANALYSIS: [Your analysis of the search results]
DECISION: [Either "CONTINUE" to formulate a new query, or "SUFFICIENT" if research is complete]"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(
            content="Current research context:\n{research_context}\n\nFormulate the next query or conclude the research.")
    ])

    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={"prompt": prompt}
    )

    while iteration < max_iterations:
        with st.expander(f"Research Iteration {iteration + 1}", expanded=False):
            response = await agent.arun(research_context)

            # Parse the response
            query = ""
            analysis = ""
            decision = ""
            current_section = ""
            for line in response.split('\n'):
                if line.startswith("QUERY:"):
                    current_section = "query"
                    query = line.replace("QUERY:", "").strip()
                elif line.startswith("ANALYSIS:"):
                    current_section = "analysis"
                elif line.startswith("DECISION:"):
                    decision = line.replace("DECISION:", "").strip()
                elif current_section == "analysis":
                    analysis += line + "\n"

            if decision.upper() == "SUFFICIENT":
                st.write("Research completed.")
                break

            st.write(f"Query: {query}")

            # Execute the Perplexity search
            search_results = await agent.arun(f"Use the Perplexity Search tool to find information about: {query}")

            st.write("Analyzing search results...")
            st.write("Summary of findings:")
            st.write(analysis)

            research_context += f"\nIteration {iteration + 1}:\nQuery: {query}\nFindings: {analysis}\n"

        iteration += 1
        query_count += 1

    display_perplexity_stats(query_count)

    return research_context
