import asyncio
import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import pending_review, resume, run  # noqa: E402 (after load_dotenv)

logger = logging.getLogger(__name__)

#: Human review pauses the graph mid-run, and Streamlit reruns its whole script
#: on every interaction — so the paused state has to outlive the process's
#: memory. An interrupt is a checkpoint, and this is where checkpoints go.
CHECKPOINT_PATH = os.getenv("DD_CHECKPOINT_PATH", ".dd_checkpoints.sqlite")


def _await(make_coro, timeout: int = 180):
    """Run one async call from Streamlit's synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    async def _go():
        return await asyncio.wait_for(make_coro(), timeout=timeout)

    if loop and loop.is_running():
        # Streamlit is already running an event loop — use a new thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _go()).result(timeout=timeout + 10)
    return asyncio.run(_go())


def run_sync(company: str, human_review: bool = False, timeout: int = 180) -> dict:
    """Start a run. With review on, this returns a *paused* state."""
    return _await(
        lambda: run(
            company,
            human_review=human_review,
            checkpoint_path=CHECKPOINT_PATH if human_review else None,
        ),
        timeout,
    )


def resume_sync(decision: dict, thread_id: str, timeout: int = 180) -> dict:
    """Continue a paused run with the reviewer's decision."""
    return _await(
        lambda: resume(decision, thread_id, checkpoint_path=CHECKPOINT_PATH),
        timeout,
    )


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
    .badge-favorable    { background: #1B3A2D; color: #4ADE80; }
    .badge-cautious     { background: #3A2F1B; color: #FBBF24; }
    .badge-unfavorable  { background: #3A1B1B; color: #F87171; }
    .badge-inconclusive { background: #1E2030; color: #8B8FA3; }

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
    "Inconclusive": "badge-inconclusive",
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
    # An unavailable section means the agent failed — that is unknown, not
    # a negative finding, and must not be shown as "no issues found".
    if not section.get("available", True):
        st.markdown(
            '<span style="color:#FBBF24;font-size:0.85rem;">'
            'Unavailable &mdash; this agent failed to run. Treat as unknown, not as a clean result.'
            '</span>',
            unsafe_allow_html=True,
        )
        return

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

review_first = st.checkbox(
    "Review the research plan before running",
    help="Pause after planning so you can approve, trim, or cancel the work "
         "before any agent spends money.",
)

# Streamlit reruns this script top to bottom on every click, so a run that is
# paused mid-graph has to be remembered explicitly.
st.session_state.setdefault("dd_state", None)
st.session_state.setdefault("dd_thread", None)
st.session_state.setdefault("dd_pending", None)


def _store(state: dict) -> None:
    st.session_state.dd_state = state
    st.session_state.dd_thread = state.get("__thread_id__")
    st.session_state.dd_pending = pending_review(state)


if run_clicked and company.strip():
    st.session_state.dd_state = None
    st.session_state.dd_pending = None
    label = "Planning..." if review_first else f"Agents analyzing {company}..."
    with st.spinner(label):
        _store(run_sync(company.strip(), human_review=review_first))
elif run_clicked:
    st.warning("Please enter a company name.")


# ---------------------------------------------------------------------------
# Approval gate — shown only while the graph is paused
# ---------------------------------------------------------------------------

pending = st.session_state.dd_pending
if pending:
    st.markdown("### Research plan awaiting approval")
    st.info(pending.get("question", "Approve this plan?"))

    kind = pending.get("company_type", "unknown")
    ticker = pending.get("ticker")
    st.markdown(f"**Classified:** {kind}{f' ({ticker})' if ticker else ''}")
    st.caption(pending.get("rationale", ""))

    tasks = pending.get("tasks", [])
    for task in tasks:
        st.markdown(f"**{task['agent']}** — {', '.join(task['tools']) or 'all tools'}")
        if task.get("caveat"):
            st.caption(task["caveat"])

    drop = st.multiselect(
        "Skip these agents",
        [t["agent"] for t in tasks],
        help="A skipped agent is reported as not researched, not as a failure.",
    )
    note = st.text_input("Note (kept in the audit trail)", "")

    approve_col, cancel_col = st.columns([1, 1])
    if approve_col.button(
        "Run this plan" if not drop else f"Run without {', '.join(drop)}",
        type="primary", use_container_width=True,
    ):
        decision = {
            "action": "revise" if drop else "approve",
            "drop_agents": drop,
            "note": note,
        }
        with st.spinner("Agents working..."):
            _store(resume_sync(decision, st.session_state.dd_thread))
        st.rerun()

    if cancel_col.button("Cancel run", use_container_width=True):
        with st.spinner("Cancelling..."):
            _store(resume_sync(
                {"action": "cancel", "note": note}, st.session_state.dd_thread))
        st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

state = st.session_state.dd_state
if state is not None:
    report: dict = state.get("final_report", {})

    if not report:
        st.error("Pipeline returned no report.")
        st.stop()

    if report.get("overall_verdict") == "Inconclusive":
        st.error(report.get("executive_summary", "Analysis could not be completed."))
        st.stop()

    unavailable = report.get("unavailable_sections", [])
    if unavailable:
        st.warning(
            f"Partial analysis — {', '.join(unavailable)} "
            f"{'agent' if len(unavailable) == 1 else 'agents'} failed. "
            "Those sections are unknown, not clean."
        )

    # A skipped section is not a failure and must not be shown as one: the
    # planner chose not to research it, so nothing we wanted is missing.
    skipped = report.get("skipped_sections", [])
    if skipped:
        st.info(
            f"Not researched — {', '.join(skipped)}. "
            "Left out of the research plan for this company, not a failure."
        )

    plan = state.get("plan")
    if plan is not None:
        kind = getattr(plan.company_type, "value", str(plan.company_type))
        label = f"Research plan — {kind}"
        if plan.ticker:
            label += f" ({plan.ticker})"
        with st.expander(label):
            st.markdown(plan.rationale)
            for task in plan.tasks:
                tools = ", ".join(task.tools) or "all available"
                st.markdown(f"**{task.agent}** — {tools}")
                if task.caveat:
                    st.caption(task.caveat)

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

    # ---- Investment thesis ----
    thesis = report.get("thesis")
    if thesis:
        st.markdown("")
        st.markdown("### Investment Thesis")

        rec = thesis.get("recommendation", "Insufficient evidence")
        st.markdown(f"**{rec}** — {thesis.get('summary', '')}")
        if thesis.get("recommendation_rationale"):
            st.caption(thesis["recommendation_rationale"])

        drivers = thesis.get("drivers", [])
        if drivers:
            st.markdown("**Drivers**")
            for d in drivers:
                arrow = "▲" if d.get("direction") == "supports" else "▼"
                st.markdown(
                    f"{arrow} {d['statement']}  \n"
                    f'<span style="color:#555A6E;font-size:0.82rem;">'
                    f"Falsified if: {d.get('what_would_falsify_this', 'n/a')} "
                    f"&nbsp;·&nbsp; {len(d.get('evidence_claim_ids', []))} sourced claim(s) "
                    f"&nbsp;·&nbsp; confidence {d.get('confidence', 0):.0%}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "No thesis driver could be traced to a sourced claim, so no "
                "recommendation is supportable."
            )

        cases = [("Bull", "bull_case"), ("Base", "base_case"), ("Bear", "bear_case")]
        cols = st.columns(3)
        for (label, key), col in zip(cases, cols):
            case = thesis.get(key) or {}
            col.markdown(f"**{label} case**")
            col.caption(case.get("narrative", "Not established."))

        # Deal breakers are graded separately from ordinary risks on purpose.
        breakers = [f for f in thesis.get("red_flags", [])
                    if f.get("severity") == "deal_breaker"]
        others = [f for f in thesis.get("red_flags", [])
                  if f.get("severity") != "deal_breaker"]
        if breakers:
            st.error("**Deal breakers** — " + "; ".join(f["issue"] for f in breakers))
        if others:
            st.markdown("**Other flags**")
            for flag in others:
                st.markdown(f"- _{flag.get('severity', 'monitor')}_ — {flag['issue']}")

        # Generated by the gap analyzer, not invented by the model.
        questions = thesis.get("open_questions", [])
        if questions:
            with st.expander(f"Open diligence items ({len(questions)})"):
                for q in questions:
                    st.markdown(f"- {q}")

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
