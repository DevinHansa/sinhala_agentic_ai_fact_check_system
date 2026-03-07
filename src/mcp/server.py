"""Model Context Protocol (MCP) Server Simulation.

This module simulates an MCP Server that exposes tools to the agent.
See: https://modelcontextprotocol.io/
"""
from typing import List, Dict, Any, Optional
from ..search import MultiSourceSearch
import sys
import io
import contextlib
import traceback


class Tool:
    """Definition of an MCP Tool."""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

class MCPServer:
    """Simulated MCP Server for Search Tools."""
    
    def __init__(self):
        self.search_engine = MultiSourceSearch()
        self.tools = [
            Tool(
                name="search_web",
                description="Search the web for Sinhala fact checking using multiple sources (Tavily, Brave, DDG). Use this to find evidence for a claim.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query in Sinhala or English."
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
        self.tools.append(
            Tool(
                name="execute_python",
                description="Execute Python code to analyze data or perform calculations. Use this for quantitative analysis. Standard library + pandas/numpy only.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute."
                        }
                    },
                    "required": ["code"]
                }
            )
        )

    def list_tools(self) -> List[Tool]:
        """List available tools."""
        return self.tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call."""
        if tool_name == "search_web":
            query = arguments.get("query")
            if not query:
                raise ValueError("Query argument is required for search_web")
            return self.search_engine.search(query)
        
        elif tool_name == "execute_python":
            code = arguments.get("code")
            if not code:
                raise ValueError("Code argument is required for execute_python")
            return self._execute_python(code)

        
        raise ValueError(f"Tool {tool_name} not found")

    def get_quota_status(self):
         return self.search_engine.get_quota_status()

    def _execute_python(self, code: str) -> str:
        """Execute Python code in a restricted environment."""
        # Simple safeguard: forbid dangerous imports in the source text
        dangerous = ["os.system", "subprocess", "shutil", "requests", "urllib"]
        if any(d in code for d in dangerous):
            return "Error: Security violation. System commands and network requests are not allowed."

        buffer = io.StringIO()
        
        # Restricted globals
        allowed_globals = {
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
            "print": print,
            "range": range,
            "len": len,
            "list": list,
            "dict": dict,
            "set": set,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "tuple": tuple,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
        }
        
        # Try importing pandas/numpy if available
        try:
            import pandas as pd
            import numpy as np
            allowed_globals["pd"] = pd
            allowed_globals["np"] = np
        except ImportError:
            pass

        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, allowed_globals)
            output = buffer.getvalue()
            return output if output else "Code executed successfully (no output)."
        except Exception:
            return f"Execution Error:\n{traceback.format_exc()}"
