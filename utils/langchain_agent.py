from langchain.agents import Tool, AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from typing import List, Dict, Any, Tuple
import requests
from pydantic import BaseModel, Field

API_URL = "http://localhost:8004"  # Your InSight API URL

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]

class RelatedTopic(BaseModel):
    topic: str
    source: str
    doc_id: str

# Function to search the vectorstore
def search_vectorstore(query: str) -> List[SearchResult]:
    response = requests.post(f"{API_URL}/search", params={"query": query})
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    return [SearchResult(**doc) for doc in data['documents']]

# Function to get related topics
def get_related_topics(query: str) -> List[RelatedTopic]:
    response = requests.post(f"{API_URL}/related_topics", params={"query": query})
    response.raise_for_status()
    data = response.json()
    return [RelatedTopic(**topic) for topic in data['topics']]

# Define tools used by the agent
tools = [
    Tool(
        name="Search",
        func=search_vectorstore,
        description="Search the vectorstore for relevant documents"
    ),
    Tool(
        name="RelatedTopics",
        func=get_related_topics,
        description="Find topics related to the search query"
    )
]

# Custom prompt for the agent
template = """You are an AI assistant helping support agents with It and Service support cases for City of Austin staff. 
Use the following tools to help answer the user's question:
{tools}

User's question: {input}

To use a tool, please use the following format:
```tool_code
{{
"tool": tool_name,
"input": tool_input
}}
Use the tools to gather information for the user, then provide a final answer based on the information you've collected.
Your response:
{agent_scratchpad}
"""

# Custom prompt class
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "agent_scratchpad"]
)

llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0)
output_parser = OpenAIFunctionsAgentOutputParser()

# Function to create the agent
def create_insight_kb_agent(model_name: str = "gpt-4o-2024-08-06", temperature: float = 0) -> AgentExecutor:
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=output_parser
    )

    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Async function to perform a search and get related topics
async def perform_insight_kb_search(query: str, selected_model: str, options: Dict[str, Any]) -> Tuple[
    List[SearchResult], List[RelatedTopic]]:
    """
    Perform a search against the InSight KB using the Langchain agent.

    Args:
        query (str): The search query.
        selected_model (str): The model to use.
        options (dict): Additional options.

    Returns:
        Tuple[List[SearchResult], List[RelatedTopic]]: Search results and related topics.
    """
    agent_executor = create_insight_kb_agent(selected_model, options.get("Temperature", 0.1))
    result = await agent_executor.arun(query)

    # Extract search results and related topics from the agent's output
    search_results = []
    related_topics = []
    for action, observation in result['intermediate_steps']:
        if action.tool == 'Search':
            search_results.extend(observation)
        elif action.tool == 'RelatedTopics':
            related_topics.extend(observation)

    return search_results, related_topics
