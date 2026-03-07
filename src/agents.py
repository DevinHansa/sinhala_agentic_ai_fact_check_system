"""Agent classes for the Sinhala Fact-Checking System."""
import json
from typing import Dict, Any
from .models import FactCheckState
from .gemini_router import GeminiRouter
from .mcp.client import MCPClient
from .vector_store import QdrantVectorStore

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, router: GeminiRouter):
        self.router = router

    def run(self, state: FactCheckState) -> FactCheckState:
        """Run the agent logic."""
        raise NotImplementedError


class ClassifierAgent(BaseAgent):
    """Classifies the domain of the statement."""
    
    def run(self, state: FactCheckState) -> FactCheckState:
        prompt = f"""You are an expert classification agent.
        Classify this Sinhala statement into ONE domain: politics, economics, or health.
        
        Statement: {state['statement']}
        
        Respond with ONLY the domain name in English (politics/economics/health).
        If uncertain, default to politics."""
        
        try:
            response = self.router.route("classify", prompt)
            domain = (response.text or "").strip().lower()
        except Exception:
            domain = "politics"
        
            
        state["domain"] = domain
        state["step"] = "classification"
        state.setdefault("trace", []).append("classification")
        return state


class ResearcherAgent(BaseAgent):
    """Retrieves information from Vector Store and Web (MCP)."""
    
    def __init__(self, router: GeminiRouter, vector_store: QdrantVectorStore, mcp_client: MCPClient):
        super().__init__(router)
        self.vector_store = vector_store
        self.mcp_client = mcp_client
        
    def run(self, state: FactCheckState) -> FactCheckState:
        # 1. Search Vector Store
        docs = self.vector_store.search(
            state["statement"],
            state["domain"],
            limit=3
        )
        
        # 2. Web Search via MCP
        # If there's critique, search for specific missing info
        if state.get("critique"):
            query = f"{state['statement']} {state['critique']}"
        else:
            query = state["statement"]
            
        try:
            search_res = self.mcp_client.call_tool("search_web", {"query": query})
            results = search_res.get("results", [])
            source = search_res.get("source", "unknown")
        except Exception as e:
            results = []
            source = "error"
            print(f"Search failed: {e}")

        # Update state
        # In a loop, we might want to append, but for now let's refresh or merge?
        # Let's merge if it's a loop
        if state.get("search_results"):
             state["search_results"].extend(results)
        else:
             state["search_results"] = results
             
        state["retrieved_docs"] = docs # Keep vector docs fresh
        state["search_source"] = source
        state["step"] = "research"
        state.setdefault("trace", []).append("research")
        return state


class AnalystAgent(BaseAgent):
    """Analyzes evidence and determines facts."""
    
    def __init__(self, router: GeminiRouter, mcp_client: MCPClient = None):
        super().__init__(router)
        self.mcp_client = mcp_client

    def run(self, state: FactCheckState) -> FactCheckState:
        evidence = state.get("search_results", []) + state.get("retrieved_docs", [])
        evidence_text = "\n\n".join([
            f"Source {i+1}: {e.get('text') or e.get('content') or ''} (URL: {e.get('url', 'N/A')})"
            for i, e in enumerate(evidence[:10]) 
        ])
        
        context = ""
        if state.get("critique"):
            context = f"\nPrevious Analysis Attempt Critiqued: {state.get('critique')}\nAddress this critique explicitly."

        # simple version: prompt engineering to encourage quantitative check
        prompt = f"""You are an expert Fact Analysis Agent. Analyze the following Sinhala statement against the provided evidence.
        
        Statement: {state['statement']}
        
        Evidence:
        {evidence_text[:6000]}
        
        {context}
        
        Task:
        1. Compare the statement with the evidence.
        2. Identify corroborating facts and contradictions.
        3. Assess the credibility of the evidence.
        4. Provide a detailed analysis in Sinhala.
        
        If the statement involves numbers/stats, verify them carefully.
        
        Output ONLY the analysis in Sinhala."""
        
        try:
            # Future expansion: If we want to use the python tool, we would handle it here.
            # For now, just having the client available is the first step.
            response = self.router.route("analyze", prompt)
            state["analysis"] = response.text or "Error generating analysis."
        except Exception as e:
            state["analysis"] = f"Analysis failed: {str(e)}"
            
        state["step"] = "analysis"
        state.setdefault("trace", []).append("analysis")
        return state


class ReviewerAgent(BaseAgent):
    """Reviews the analysis for completeness and accuracy."""
    
    def run(self, state: FactCheckState) -> FactCheckState:
        analysis = state.get("analysis", "")
        statement = state.get("statement", "")
        
        prompt = f"""You are a strict Reviewer Agent. Review the following analysis of a statement.
        
        Statement: {statement}
        Analysis: {analysis}
        
        Task:
        1. Is the analysis supported by evidence?
        2. Is the conclusion clear?
        3. Is the Sinhala language natural and professional?
        
        Return a JSON object:
        {{
            "approved": boolean,
            "critique": "If not approved, explain what is missing or wrong in English. If approved, return null.",
            "instructions": "Instructions for the Researcher/Analyst if changes are needed."
        }}
        """
        
        try:
            response = self.router.route("decide", prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            
            if result.get("approved"):
                 state["critique"] = None
            else:
                 state["critique"] = result.get("critique")
                 state["editor_feedback"] = result.get("instructions")
                 
        except Exception:
            # If parsing fails, assume approved to avoid infinite loops in error state
            state["critique"] = None
            
        state.setdefault("trace", []).append("review")
        return state


class WriterAgent(BaseAgent):
    """Generates the final verdict and report."""
    
    def run(self, state: FactCheckState) -> FactCheckState:
        analysis = state.get("analysis", "")
        
        prompt = f"""You are the Final Writer Agent. Produce a polished, production-grade final report in Sinhala based on this analysis.
        
        Analysis: {analysis}
        
        Task:
        1. Determine the final verdict (True/False/Insufficient).
        2. Write a clear justification in Sinhala.
        3. Format nicely with Markdown.
        
        Respond with a JSON object:
        {{
            "verdict": "true" | "false" | "insufficient",
            "report": "The full Markdown report in Sinhala."
        }}
        """
        
        try:
            response = self.router.route("flash", prompt) # Use fast model for formatting
            text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            
            state["verdict"] = data.get("verdict", "insufficient").lower()
            state["final_report"] = data.get("report", analysis)
            # Legacy field support
            state["analysis"] = state["final_report"] 
            
        except Exception:
            state["verdict"] = "insufficient"
            state["final_report"] = analysis
            
        state["step"] = "finished"
        state.setdefault("trace", []).append("write")
        return state
