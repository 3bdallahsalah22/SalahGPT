
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

# ==============================
# Project Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "Data"

DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ENV_FILE)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ==============================
# LangChain Imports
# ==============================

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    START,
    MessagesState
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.sqlite import SqliteSaver


# ==============================
# Project Imports
# ==============================

from Tools.tools import tools


# ==============================
# Configuration
# ==============================
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

ALLOWED_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
}

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named SalahGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""


# ==============================
# Model Validation
# ==============================

def normalize_model_name(model_name: str | None) -> str:

    """
    Validate selected model from frontend.
    If model is missing or not allowed,
    fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# ==============================
# Build Agent
# ==============================

def build_agent(model_name: str):

    """
    Build one LangGraph agent for a selected Gemini model.
    """

    selected_model = normalize_model_name(model_name)

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
           timeout=60,
    max_retries=2,
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Chatbot Node
    def chatbot_node(state: MessagesState):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # Tool Node
    tool_node = ToolNode(tools)

    # Create Workflow
    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)

    workflow.add_node("tools", tool_node)

    # Define Workflow Edges
    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge("tools", "chatbot")

    # 
    # SQLite Checkpointer
    # 

    checkpoint_db = DATA_DIR / "langgraph_checkpoints.sqlite"

    conn = sqlite3.connect(
        str(checkpoint_db),
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    # Compile Agent
    return workflow.compile(
        checkpointer=checkpointer
    )


# 
# Agent Cache
# 

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):

    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = build_agent(
            selected_model
        )

    return _AGENT_CACHE[selected_model]
