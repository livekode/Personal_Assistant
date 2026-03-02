import os
from typing import Annotated, Sequence, TypedDict, Optional
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
from email_sender import send_gmail

# Load environment variables
load_dotenv()

# Get API key from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class ToolAgent:
    """A reusable tool agent that can process queries with web search and email capabilities."""
    
    def __init__(self, model_name: str = "gpt-4.1-nano", verbose: bool = False):
        """
        Initialize the tool agent.
        
        Args:
            model_name: OpenAI model to use (default: "gpt-4")
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self.app = self._build_agent(model_name)
    
    def _build_agent(self, model_name: str):
        """Build the LangGraph agent."""
        
        @tool
        def websearch(query: str) -> str:
            """Execute this tool to make a search on the internet."""
            if not TAVILY_API_KEY:
                return "Error: TAVILY_API_KEY not found in environment variables."
            
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily_client.search(query)
            
            if response and response.get('results'):
                best_result = response['results'][0]
                return best_result.get('content', 'No content found in search results.')
            return "No search results found."

        @tool
        def send_email(name: str, email: str, subject: str, body_text: str) -> str:
            """Send a confirmation email via Gmail API."""
            try:
                message_id = send_gmail(email, subject, body_text)
                return f"✅ Email sent to {name} ({email}). Gmail message ID: {message_id}"
            except Exception as e:
                return f"❌ Failed to send email to {name} ({email}): {e}"

        self.tools = [websearch, send_email]
        
        # Bind tools to model
        self.model = ChatOpenAI(model=model_name).bind_tools(self.tools)
        
        # Build the graph
        graph = StateGraph(AgentState)
        
        def call_model(state: AgentState) -> AgentState:
            """Call the model with the current state."""
            system_prompt = SystemMessage(content="You are my AI assistant. Use the available tools when needed to answer queries accurately.")
            response = self.model.invoke([system_prompt] + state["messages"])
            return {"messages": [response]}
        
        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(tools=self.tools))
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Add conditional edges directly
        graph.add_conditional_edges(
            "agent",
            lambda state: "tools" if state["messages"][-1].tool_calls else END
        )
        
        # Add edge from tools back to agent
        graph.add_edge("tools", "agent")
        
        return graph.compile()
    
    def query(self, user_input: str) -> str:
        """
        Process a user query and return the final response.
        
        Args:
            user_input: The user's question or command
            
        Returns:
            The agent's final response as a string
        """
        if self.verbose:
            print(f"📝 Processing query: '{user_input}'")
        
        # Prepare input
        inputs = {"messages": [("user", user_input)]}
        
        # Stream through the graph
        final_response = None
        for s in self.app.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            
            # Only capture if it's an AI message with content and no tool calls
            if hasattr(message, 'type') and message.type == "ai" and message.content and not message.tool_calls:
                final_response = message.content
                if self.verbose:
                    print(f"✅ Final response: '{final_response[:100]}...'")
        
        return final_response or "No response generated."
    
    async def aquery(self, user_input: str) -> str:
        """
        Async version of query for use in async applications.
        
        Args:
            user_input: The user's question or command
            
        Returns:
            The agent's final response as a string
        """
        # For now, this just calls the sync version
        # You could implement proper async streaming if needed
        return self.query(user_input)

# Create a default instance for easy importing
_default_agent = None

def get_agent(verbose: bool = False) -> ToolAgent:
    """Get or create a default tool agent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = ToolAgent(verbose=verbose)
    return _default_agent

def process_query(user_input: str, verbose: bool = False) -> str:
    """
    Simple function to process a query using the default agent.
    
    Args:
        user_input: The user's question or command
        verbose: Whether to print debug information
        
    Returns:
        The agent's response as a string
    """
    agent = get_agent(verbose=verbose)
    return agent.query(user_input)

async def aprocess_query(user_input: str, verbose: bool = False) -> str:
    """
    Async version of process_query.
    
    Args:
        user_input: The user's question or command
        verbose: Whether to print debug information
        
    Returns:
        The agent's response as a string
    """
    agent = get_agent(verbose=verbose)
    return await agent.aquery(user_input)

# If run directly, demonstrate usage
if __name__ == "__main__":
    # Test the agent
    result = process_query("What is the current temperature in Monaco?", verbose=True)
    print(f"\n🎯 Result: {result}")