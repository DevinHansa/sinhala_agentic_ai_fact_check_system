"""LangGraph workflow for fact checking."""
import asyncio
import os
import json
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END

# noinspection PyUnresolvedReference
from google import genai

from .models import FactCheckState
from .vector_store import QdrantVectorStore
from .gemini_router import GeminiRouter
from .mcp.server import MCPServer
from .mcp.client import MCPClient
from .agents import (
    ClassifierAgent,
    ResearcherAgent,
    AnalystAgent,
    ReviewerAgent,
    WriterAgent
)


class FactCheckingWorkflow:
    """LangGraph-based fact checking workflow with 4-Agent Architecture."""
    
    def __init__(self, vector_store: QdrantVectorStore, client=None):
        """Initialize workflow with dependencies."""
        self.vector_store = vector_store
        # Initialize MCP Architecture
        self.mcp_server = MCPServer()
        self.mcp_client = MCPClient(self.mcp_server)
        
        self.client = client or genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.router = GeminiRouter(client=self.client)
        
        # Initialize Agents
        self.classifier = ClassifierAgent(self.router)
        self.researcher = ResearcherAgent(self.router, self.vector_store, self.mcp_client)
        self.analyst = AnalystAgent(self.router, self.mcp_client)
        self.reviewer = ReviewerAgent(self.router)
        self.writer = WriterAgent(self.router)
        
        self.workflow = self._create_workflow()
    
    def _check_review(self, state: FactCheckState) -> Literal["writer_agent", "researcher_agent"]:
        """Conditional edge to determine next step based on review."""
        # Limit loops to 2 revisions to prevent infinite cycles
        if state.get("critique") and state.get("revision_count", 0) < 2:
            state["revision_count"] = state.get("revision_count", 0) + 1
            return "researcher_agent"
        return "writer_agent"

    def _create_workflow(self):
        """Build the Multi-Agent LangGraph workflow."""
        workflow = StateGraph(FactCheckState)
        
        # Add Nodes (Agent Wrappers)
        # We need to wrap the classes' run methods to be compatible with StateGraph nodes
        workflow.add_node("classify_agent", self.classifier.run)
        workflow.add_node("researcher_agent", self.researcher.run)
        workflow.add_node("analysis_agent", self.analyst.run)
        workflow.add_node("reviewer_agent", self.reviewer.run)
        workflow.add_node("writer_agent", self.writer.run)
        
        # Define Flow
        workflow.set_entry_point("classify_agent")
        workflow.add_edge("classify_agent", "researcher_agent")
        workflow.add_edge("researcher_agent", "analysis_agent")
        workflow.add_edge("analysis_agent", "reviewer_agent")
        
        # Conditional Edge
        workflow.add_conditional_edges(
            "reviewer_agent",
            self._check_review,
            {
                "researcher_agent": "researcher_agent",
                "writer_agent": "writer_agent"
            }
        )
        
        workflow.add_edge("writer_agent", END)
        
        return workflow.compile()
    
    def verify(self, statement: str) -> dict:
        """Verify a statement synchronously."""
        initial_state = {
            "statement": statement,
            "domain": "",
            "retrieved_docs": [],
            "search_results": [],
            "analysis": "",
            "verdict": "",
            "method_used": "multi-agent-graph-v2",
            "sufficiency": None,
            "search_source": None,
            "cached": False,
            "revision_count": 0,
            "critique": None,
            "editor_feedback": None,
            "final_report": None,
            "step": "start"
        }
        
        result = self.workflow.invoke(initial_state)
        return result
    
    async def verify_async(self, statement: str) -> dict:
        """Verify a statement asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.verify, statement)
