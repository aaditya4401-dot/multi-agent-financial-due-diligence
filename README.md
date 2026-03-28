# Multi-Agent Due Diligence System

A multi-agent AI system built with LangGraph that performs automated due diligence on companies by deploying 4 specialized agents in parallel to analyze financial health, market position, risk factors, and public sentiment — then resolves conflicts between agents and synthesizes findings into a structured investment memo with confidence scoring.

![Demo](screenshots/demo.png)

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd multi-agent-financial-due-diligence
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...        # Required — powers all 4 agents + resolver + synthesizer
TAVILY_API_KEY=tvly-...      # Required — web search, news, Reddit, SEC filings
```

### 3. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Type a company name (e.g. `Stripe`, `Tesla`, `Coinbase`) and click **Run Analysis**.

### 4. Or run from CLI

```bash
python -m src.graph Stripe
```

Prints the full JSON report to stdout.

### 5. Run tests

```bash
python -m pytest tests/test_tools.py -v
```

---

## Demo: Analyzing Stripe

When you analyze **Stripe**, the system produces:

| Metric | Value |
|--------|-------|
| **Overall Score** | 85/100 |
| **Verdict** | Favorable |
| **Risk Level** | Moderate |
| **Confidence** | 85% |

**What the agents found:**

- **Financial Agent** — Stripe's payment volume reached $1.9T (34% YoY growth), Revenue suite ARR hitting $1B, valuation at $159B after latest tender offer. yfinance returns nothing (private company), so the agent automatically fell back to Tavily web search.
- **Market Agent** — 17% global payment processing share, $1T processing volume is <2% of the total digital payments TAM, significant room for growth.
- **Risk Agent** — Compliance challenges with KYC/identity verification, restrictions on high-risk industries (gambling, crypto). No major litigation flagged.
- **Sentiment Agent** — Neutral news trajectory, positive developer sentiment on Reddit, mixed employee reviews on Glassdoor (3.4/5 compensation rating).

**3 conflicts detected and resolved:**

1. **Factual Contradiction** — Financial agent reported $1.9T payment volume vs Market agent's $1.4T. Resolved by preferring the higher-confidence source.
2. **Stale Data** — $159B valuation (recent) vs $50B Series I valuation (older). Resolved by accepting the more recent figure.
3. **Complementary Tension** — Risk agent flagged compliance challenges while Sentiment agent found neutral media coverage. Both true — compliance issues haven't yet impacted public perception.

The full analysis runs in ~60 seconds with all 4 agents executing concurrently.

---

## Architecture Overview

```
User Input: "Analyze Stripe"
        │
        ▼
┌─────────────────────┐
│   Orchestrator      │  ← LangGraph StateGraph
│   (routes, manages) │
└────┬───┬───┬───┬────┘
     │   │   │   │        ← Fan-out (parallel execution)
     ▼   ▼   ▼   ▼
   ┌───┐┌───┐┌───┐┌───┐
   │ F ││ M ││ R ││ S │  ← 4 independent agents
   │ i ││ a ││ i ││ e │     each with own LLM + tools
   │ n ││ r ││ s ││ n │
   └─┬─┘└─┬─┘└─┬─┘└─┬─┘
     │   │   │   │        ← Fan-in (merge results)
     ▼   ▼   ▼   ▼
┌─────────────────────┐
│  Conflict Resolver  │  ← Detects contradictions
│  (classify + score) │     between agent findings
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Synthesizer       │  ← Produces structured
│   (final report)    │     investment memo
└──────────┬──────────┘
           ▼
   Due Diligence Report
   (with confidence scores)
```

### How the graph is wired

```python
# Fan-out: orchestrator → all 4 agents in parallel
workflow.add_edge("orchestrator", "financial")
workflow.add_edge("orchestrator", "market")
workflow.add_edge("orchestrator", "risk")
workflow.add_edge("orchestrator", "sentiment")

# Fan-in: all 4 → conflict resolver
workflow.add_edge(["financial", "market", "risk", "sentiment"], "conflict_resolver")

# Linear: resolver → synthesizer → END
workflow.add_edge("conflict_resolver", "synthesizer")
workflow.add_edge("synthesizer", END)
```

LangGraph runs all four agent nodes concurrently via asyncio. Each node writes to its own state field (`financial_findings`, `market_findings`, etc.) — no write conflicts. Shared fields (`conflicts`, `messages`) use `Annotated[list, add]` reducers that concatenate results.

---

## Key Concepts

### Multi-Agent vs Single Agent

Single agent = **one brain with many tools**.
This project = **4 agents with their own brains** — each has its own LLM call, system prompt, personality, and tool set.

Key implications:
- Each agent can **fail independently** (financial API down, but others succeed)
- Agents can **disagree** (different data sources → different conclusions)
- You need an **orchestrator** to coordinate them and a **resolver** to handle disagreements

### Conflict Resolution

When agents disagree, the resolver classifies conflicts into 3 types:

| Type | Example | Resolution |
|------|---------|------------|
| **Factual contradiction** | Agent A: revenue $20B, Agent B: revenue $14B | Pick the one with better source quality |
| **Complementary tension** | Financial: growing fast, Risk: burning cash | Both are true — surface the nuance |
| **Stale data** | Agent A: 2024 data, Agent B: 2026 data | Prefer the more recent source |

The resolver is a single structured-output LLM call (not a react agent) that classifies conflicts and assigns confidence scores.

### Graceful Degradation

Every agent and tool handles failures without crashing the pipeline:

```python
try:
    data = await call_financial_api(company)
except Exception:
    return AgentFindings(
        findings=[],
        summary="Financial data unavailable — API error",
        confidence=0.0
    )
```

If yfinance fails (e.g. private company), the financial agent falls back to web search. If Tavily is down, tools return empty lists. The report still generates — it just flags which sections have lower reliability.

---

## Project Structure

```
multi-agent-due-diligence/
├── README.md
├── CLAUDE.md                     # AI coding assistant instructions
├── requirements.txt
├── .env                          # API keys (OPENAI_API_KEY, TAVILY_API_KEY)
├── app.py                        # Streamlit frontend (dark fintech theme)
├── src/
│   ├── __init__.py
│   ├── state.py                  # DueDiligenceState, Finding, AgentFindings, Conflict
│   ├── graph.py                  # LangGraph StateGraph wiring + CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── utils.py              # Shared: parse_react_output, error_findings
│   │   ├── financial.py          # Financial analysis agent (yfinance + Tavily)
│   │   ├── market.py             # Market research agent (Tavily)
│   │   ├── risk.py               # Risk assessment agent (SEC + Tavily)
│   │   ├── sentiment.py          # Sentiment analysis agent (news + Reddit + Tavily)
│   │   ├── conflict_resolver.py  # Conflict detection & classification (structured output)
│   │   └── synthesizer.py        # Final report generation (structured output)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py       # Tavily web search wrapper
│   │   ├── financial_tools.py    # yfinance + Tavily fallback for financials
│   │   ├── news_tools.py         # News search + Reddit sentiment via Tavily
│   │   └── sec_tools.py          # SEC EDGAR filing search via Tavily
│   └── models/
│       ├── __init__.py
│       └── schemas.py            # Pydantic v2 models: DueDiligenceReport, ReportSection, etc.
├── screenshots/
│   └── demo.png                  # Streamlit app screenshot
└── tests/
    ├── __init__.py
    └── test_tools.py             # Smoke tests for all tool modules
```

---

## Tools & APIs

| Tool | Used By | What It Does |
|------|---------|-------------|
| **yfinance** | Financial Agent | Stock prices, revenue, margins, cash flow for public companies |
| **Tavily Search** | All Agents | Web search (general, news, Reddit, SEC). Single API key powers everything |
| **Tavily (news topic)** | Sentiment Agent | `search_depth="advanced"` + `topic="news"` for recent articles |
| **Tavily (reddit scope)** | Sentiment Agent | `include_domains=["reddit.com"]` for developer/user discussions |
| **Tavily (sec.gov scope)** | Risk Agent | `include_domains=["sec.gov"]` for SEC EDGAR filings |

All tools return empty results on failure instead of raising exceptions, so the pipeline always completes.

---

## Output Structure

The synthesizer produces a structured investment memo via `with_structured_output`:

```python
class DueDiligenceReport(BaseModel):
    company_name: str
    report_date: str
    overall_score: int                    # 0-100
    overall_verdict: Literal["Favorable", "Cautious", "Unfavorable"]
    risk_level: Literal["Low", "Moderate", "High"]
    overall_confidence: float             # 0.0-1.0

    financial_section: ReportSection
    market_section: ReportSection
    risk_section: ReportSection
    sentiment_section: SentimentSection   # + news_trajectory, developer/employee sentiment

    conflicts_detected: list[Conflict]
    executive_summary: str
```

---

## Requirements

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
pydantic>=2.0
yfinance>=0.2.0
tavily-python>=0.3.0
streamlit>=1.38.0
python-dotenv>=1.0.0
```
