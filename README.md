<h1 align="center">සිංහල සත්‍ය සෙවුම්කරු<br>Sinhala Agentic AI Fact-Check System</h1>

<p align="center">
  <strong>A multi-agent AI system that verifies Sinhala-language claims using a 5-agent LangGraph workflow, Gemini AI, vector search, and real-time web retrieval.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Gemini-AI-orange" alt="Gemini">
  <img src="https://img.shields.io/badge/LangGraph-Workflow-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-red" alt="Qdrant">
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b" alt="Streamlit">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │      Streamlit Frontend      │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │    LangGraph State Machine   │
                        └──────────────┬──────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
  ┌───────▼───────┐          ┌────────▼────────┐         ┌────────▼────────┐
  │  1. Classifier │          │  2. Researcher  │         │  3. Analyst     │
  │  Agent         │──────▶   │  Agent          │──────▶  │  Agent          │
  │ (Domain ID)    │          │ (Evidence)      │         │ (Reasoning)     │
  └────────────────┘          └────────┬────────┘         └────────┬────────┘
                                       │                           │
                               ┌───────┴───────┐         ┌────────▼────────┐
                               │  MCP Server   │         │  4. Reviewer    │
                               │ (Search Hub)  │         │  Agent          │
                               └───────┬───────┘         │ (QA / Critique) │
                                       │                 └────────┬────────┘
                          ┌────────────┼────────────┐             │
                          │            │            │    ┌────────▼────────┐
                     ┌────▼───┐  ┌─────▼──┐  ┌─────▼─┐  │  5. Writer      │
                     │ Tavily │  │ Brave  │  │  DDG  │  │  Agent          │
                     └────────┘  └────────┘  └───────┘  │ (Final Report)  │
                                                        └─────────────────┘
                               ┌───────────────┐
                               │  Qdrant       │
                               │  Vector Store │
                               └───────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| AI Engine | **Google Gemini** (Flash / Pro / Thinking) | Powering all five agents with model routing |
| Orchestration | **LangGraph** | Stateful multi-agent workflow with conditional edges |
| Vector Store | **Qdrant** | Domain-specific collections (politics, economics, health) |
| Search | **Tavily + Brave + DuckDuckGo** | Multi-source web search via MCP |
| Tool Protocol | **MCP (Model Context Protocol)** | Unified tool interface for search providers |
| Frontend | **Streamlit** | Interactive Sinhala-native web UI |
| Model Routing | **GeminiRouter** | Intelligent quota-aware model selection |

## Features

- **5-Agent Workflow** — Classify, Research, Analyze, Review, and Write agents collaborate in a directed graph
- **Sinhala-Native Processing** — Prompts and UI optimized for Sinhala language claims
- **Multi-Source Evidence** — Combines vector-store retrieval with live web search from three providers
- **Smart Model Routing** — Automatically selects Gemini Flash, Pro, or Thinking based on task complexity and quota
- **Reviewer Loop** — Quality-assurance agent can send results back for revision
- **Caching** — TTL-based result caching to avoid duplicate API calls
- **MCP Integration** — Search tools exposed through Model Context Protocol server/client
- **Docker Ready** — One-command containerized deployment

## Project Structure

```
sinhala_agentic_ai_fact_check_system/
├── app.py                    # Streamlit application entry point
├── main.py                   # CLI entry point
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image definition
├── docker-compose.yml        # Multi-service orchestration
├── .env.example              # Environment variable template
├── test_system.py            # Integration tests (requires API keys)
├── test_system_mock.py       # Mock tests (no API keys needed)
├── src/
│   ├── __init__.py           # Package metadata (version)
│   ├── agents.py             # 5 agent classes (Classifier → Writer)
│   ├── workflow.py           # LangGraph state machine definition
│   ├── models.py             # Pydantic state models
│   ├── vector_store.py       # Qdrant vector store wrapper
│   ├── search.py             # Multi-source search aggregator
│   ├── gemini_router.py      # Quota-aware Gemini model router
│   ├── cache.py              # TTL-based result cache
│   ├── async_processor.py    # Async task utilities
│   └── mcp/
│       ├── server.py         # MCP tool server (search tools)
│       └── client.py         # MCP client for agent tool calls
├── data/                     # Seed data for vector collections
└── .github/
    └── workflows/
        └── ci.yml            # CI pipeline (lint + test)
```

## Prerequisites

- **Python 3.10+**
- **Google Gemini API key** (required)
- At least one search API key (Tavily or Brave) recommended
- **Visual C++ Redistributable** on Windows (for Qdrant native libraries) — [download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevinHansa/sinhala_agentic_ai_fact_check_system.git
   cd sinhala_agentic_ai_fact_check_system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The UI will open at `http://localhost:8501`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key ([get one](https://aistudio.google.com/apikey)) |
| `TAVILY_API_KEY` | Recommended | Tavily search API key — 1,000 free searches/month ([get one](https://tavily.com)) |
| `BRAVE_API_KEY` | Optional | Brave search API key — 2,000 free searches/month ([get one](https://brave.com/search/api/)) |
| `QDRANT_PATH` | Optional | Local Qdrant storage path (default: `./qdrant_data`) |
| `QDRANT_URL` | Optional | Remote Qdrant instance URL |
| `QDRANT_API_KEY` | Optional | API key for remote Qdrant instance |

> DuckDuckGo search is always available as a free fallback — no API key needed.

## Docker Deployment

**Build and run with Docker:**
```bash
docker build -t sinhala-fact-check .
docker run -p 8501:8501 --env-file .env sinhala-fact-check
```

**Or use Docker Compose:**
```bash
docker-compose up --build
```

## How It Works

1. **User submits a Sinhala claim** through the Streamlit UI
2. **Classifier Agent** identifies the domain (politics / economics / health)
3. **Researcher Agent** retrieves evidence from Qdrant vector store and live web search (Tavily → Brave → DuckDuckGo fallback) via MCP
4. **Analyst Agent** reasons over the evidence using Gemini Pro/Thinking to assess claim accuracy
5. **Reviewer Agent** critiques the analysis for logical gaps; may loop back for revision
6. **Writer Agent** produces the final verdict (True / False / Insufficient) with a Sinhala-language explanation

The entire flow is orchestrated as a **LangGraph state machine** with conditional edges, ensuring deterministic agent transitions and state persistence across steps.

## Testing

**Mock tests** (no API keys required):
```bash
python test_system_mock.py
```

**Integration tests** (requires `GOOGLE_API_KEY`):
```bash
python test_system.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.
