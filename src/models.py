"""Data models for fact checking system."""
from typing import TypedDict, List, Optional


class FactCheckState(TypedDict):
    """State for the fact-checking workflow."""
    statement: str
    domain: str
    retrieved_docs: List[dict]
    search_results: List[dict]
    analysis: str
    verdict: str
    method_used: str
    sufficiency: Optional[str]
    search_source: Optional[str]
    cached: bool
    # New fields for Multi-Agent Workflow
    revision_count: int
    critique: Optional[str]
    editor_feedback: Optional[str]
    final_report: Optional[str]
    step: str # current step name
    trace: List[str] # log of steps taken
