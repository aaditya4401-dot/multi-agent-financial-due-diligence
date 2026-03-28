import asyncio
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import run  # noqa: E402 (after load_dotenv)

logger = logging.getLogger(__name__)


def run_sync(company: str, timeout: int = 120) -> dict:
    """Run the async pipeline from sync Streamlit context with a timeout."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    async def _run_with_timeout():
        return await asyncio.wait_for(run(company), timeout=timeout)

    if loop and loop.is_running():
        # Streamlit is already running an event loop — use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_with_timeout())
            return future.result(timeout=timeout + 10)
    else:
        return asyncio.run(_run_with_timeout())

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Due Diligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — fintech dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---- Hide Streamlit chrome ---- */
    #MainMenu, header, footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0F1117;
        color: #E2E4EB;
    }

    /* ---- Inputs ---- */
    .stTextInput > div > div > input {
        background: #1A1D29;
        border: 1px solid #2A2D39;
        border-radius: 6px;
        color: #E2E4EB;
        font-family: 'Inter', sans-serif;
        padding: 0.6rem 0.8rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4F8EF7;
        box-shadow: 0 0 0 1px #4F8EF7;
    }
    .stTextInput > div > div > input::placeholder {
        color: #555A6E;
    }

    /* ---- Primary button ---- */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #4F8EF7;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.6rem 1.2rem;
        transition: background 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #3D7AE0;
        color: #FFFFFF;
    }

    /* ---- Dividers ---- */
    hr {
        border-color: #2A2D39;
        opacity: 0.6;
    }

    /* ---- Spinner ---- */
    .stSpinner > div > div {
        border-top-color: #4F8EF7 !important;
    }
    .stSpinner > div > span {
        color: #8B8FA3 !important;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: #1A1D29;
        border: 1px solid #2A2D39;
        border-radius: 8px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0.25rem 0;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 12px;
        color: #8B8FA3;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    /* ---- Badges (shared) ---- */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }

    /* Verdict badges */
    .badge-favorable   { background: #1B3A2D; color: #4ADE80; }
    .badge-cautious    { background: #3A2F1B; color: #FBBF24; }
    .badge-unfavorable { background: #3A1B1B; color: #F87171; }

    /* Severity badges */
    .badge-strong { background: #1B3A2D; color: #4ADE80; }
    .badge-watch  { background: #3A2F1B; color: #FBBF24; }
    .badge-flag   { background: #3A1B1B; color: #F87171; }
    .badge-low    { background: #1E2030; color: #8B8FA3; }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #E2E4EB;
        margin: 0.8rem 0 0.5rem 0;
    }
    h1 {
        font-family: 'Inter', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    h3, .stMarkdown h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #E2E4EB !important;
    }

    /* ---- Expanders ---- */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #E2E4EB;
        background: #1A1D29;
        border: 1px solid #2A2D39;
        border-radius: 6px;
    }
    [data-testid="stExpander"] {
        background: #1A1D29;
        border: 1px solid #2A2D39;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    [data-testid="stExpander"] details {
        border: none !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #E2E4EB;
    }
    [data-testid="stExpander"] summary:hover {
        color: #4F8EF7;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: 1px solid #2A2D39;
    }

    /* ---- Finding rows ---- */
    .finding-row {
        padding: 0.65rem 0;
        border-bottom: 1px solid #2A2D39;
        color: #C8CAD4;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    .finding-row small {
        color: #6B6F82 !important;
    }

    /* ---- Conflict cards ---- */
    .conflict-card {
        background: #1A1D29;
        border-left: 3px solid #FBBF24;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-radius: 0 6px 6px 0;
        color: #C8CAD4;
        font-size: 0.875rem;
        line-height: 1.55;
    }
    .conflict-card strong {
        color: #E2E4EB;
    }
    .conflict-card em {
        color: #8B8FA3;
    }
    .conflict-card small {
        color: #6B6F82;
    }

    /* ---- Streamlit metric override (sentiment section) ---- */
    [data-testid="stMetric"] {
        background: #1A1D29;
        border: 1px solid #2A2D39;
        border-radius: 6px;
        padding: 0.8rem;
    }
    [data-testid="stMetric"] label {
        color: #8B8FA3 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* ---- Caption ---- */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #6B6F82 !important;
    }

    /* ---- Section confidence bar ---- */
    .conf-bar {
        display: inline-block;
        font-size: 12px;
        color: #8B8FA3;
        font-weight: 500;
        letter-spacing: 0.03em;
        padding: 0.2rem 0;
        margin-bottom: 0.3rem;
    }

    /* ---- Info/Success/Warning/Error boxes ---- */
    [data-testid="stAlert"] {
        background: #1A1D29;
        border: 1px solid #2A2D39;
        color: #C8CAD4;
        border-radius: 6px;
    }

    /* ---- Title tagline ---- */
    .tagline {
        font-size: 13px;
        color: #555A6E;
        letter-spacing: 0.04em;
        margin-top: -0.6rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VERDICT_CLASSES = {
    "Favorable": "badge-favorable",
    "Cautious": "badge-cautious",
    "Unfavorable": "badge-unfavorable",
}

SEVERITY_CLASSES = {
    "Strong": "badge-strong",
    "Watch": "badge-watch",
    "Flag": "badge-flag",
    "Low": "badge-low",
}

CONFLICT_TYPE_LABELS = {
    "factual_contradiction": "Factual Contradiction",
    "complementary_tension": "Complementary Tension",
    "stale_data": "Stale Data",
}


def render_metric(label: str, value: str, col):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_verdict_badge(verdict: str):
    cls = VERDICT_CLASSES.get(verdict, "badge-cautious")
    st.markdown(f'<span class="badge {cls}">{verdict}</span>', unsafe_allow_html=True)


def render_severity_badge(severity: str) -> str:
    cls = SEVERITY_CLASSES.get(severity, "badge-low")
    return f'<span class="badge {cls}">{severity}</span>'


def render_findings(section: dict):
    findings = section.get("findings", [])
    if not findings:
        st.markdown('<span style="color:#555A6E;font-size:0.85rem;">No findings available.</span>',
                    unsafe_allow_html=True)
        return

    conf = section.get("section_confidence", 0)
    st.markdown(f'<span class="conf-bar">SECTION CONFIDENCE: {conf:.0%}</span>', unsafe_allow_html=True)

    for f in findings:
        sev_badge = render_severity_badge(f.get("severity", "Low"))
        conf_pct = f"{f.get('confidence', 0):.0%}"
        st.markdown(
            f'<div class="finding-row">'
            f'{sev_badge} &nbsp; {f.get("claim", "")}<br/>'
            f'<small>Confidence: {conf_pct} &middot; Source: {f.get("source", "N/A")}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_conflicts(conflicts: list[dict]):
    if not conflicts:
        st.markdown(
            '<span style="color:#4ADE80;font-size:0.85rem;">No conflicts detected between agents.</span>',
            unsafe_allow_html=True,
        )
        return

    for c in conflicts:
        type_label = CONFLICT_TYPE_LABELS.get(c.get("type", ""), c.get("type", ""))
        st.markdown(
            f'<div class="conflict-card">'
            f'<strong>{type_label}</strong> &mdash; '
            f'{c.get("agent_a", "?")} vs {c.get("agent_b", "?")}<br/>'
            f'<em>"{c.get("claim_a", "")[:200]}"</em> vs '
            f'<em>"{c.get("claim_b", "")[:200]}"</em><br/><br/>'
            f'<strong>Resolution:</strong> {c.get("resolution", "")}<br/>'
            f'<small>Resolved confidence: {c.get("resolved_confidence", 0):.0%}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Due Diligence Terminal")
st.markdown('<div class="tagline">MULTI-AGENT ANALYSIS &nbsp;|&nbsp; LANGGRAPH ORCHESTRATION &nbsp;|&nbsp; CONFLICT RESOLUTION</div>',
            unsafe_allow_html=True)

col_input, col_btn = st.columns([4, 1])
company = col_input.text_input("Company name", placeholder="Enter company name (e.g. Stripe, Tesla, Coinbase)",
                               label_visibility="collapsed")
run_clicked = col_btn.button("Run Analysis", type="primary", use_container_width=True)

if run_clicked and company.strip():
    with st.spinner(f"Agents analyzing {company}..."):
        state = run_sync(company.strip(), timeout=120)

    report: dict = state.get("final_report", {})

    if not report or (report.get("overall_score", 0) == 0 and "failed" in report.get("executive_summary", "").lower()):
        st.error(f"Pipeline failed. {report.get('executive_summary', 'Unknown error')}")
        st.stop()

    # ---- Metric cards ----
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    render_metric("Overall Score", f"{report.get('overall_score', 0)}/100", c1)
    render_metric("Risk Level", report.get("risk_level", "N/A"), c2)
    render_metric("Confidence", f"{report.get('overall_confidence', 0):.0%}", c3)
    render_metric("Report Date", report.get("report_date", "N/A"), c4)

    # ---- Verdict ----
    st.markdown("")
    st.markdown("### Verdict")
    render_verdict_badge(report.get("overall_verdict", "Cautious"))

    # ---- Executive summary ----
    st.markdown("")
    st.markdown("### Executive Summary")
    st.markdown(f'<div style="color:#C8CAD4;font-size:0.9rem;line-height:1.7;">'
                f'{report.get("executive_summary", "No summary available.")}</div>',
                unsafe_allow_html=True)

    # ---- Section findings ----
    st.markdown("")
    st.markdown("### Detailed Findings")

    with st.expander("Financial Analysis", expanded=True):
        render_findings(report.get("financial_section", {}))

    with st.expander("Market Analysis"):
        render_findings(report.get("market_section", {}))

    with st.expander("Risk Assessment"):
        render_findings(report.get("risk_section", {}))

    with st.expander("Sentiment Analysis"):
        section = report.get("sentiment_section", {})
        render_findings(section)
        if section:
            st.markdown("")
            s1, s2, s3 = st.columns(3)
            s1.metric("News Trajectory", section.get("news_trajectory", "N/A"))
            s2.metric("Developer Sentiment", section.get("developer_sentiment", "N/A"))
            s3.metric("Employee Sentiment", section.get("employee_sentiment", "N/A"))

    # ---- Conflicts ----
    st.markdown("")
    st.markdown("### Conflicts Detected")
    render_conflicts(report.get("conflicts_detected", []))

elif run_clicked:
    st.warning("Please enter a company name.")
