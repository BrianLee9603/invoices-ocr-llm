"""
Invoice OCR + LLM Extraction — Premium Streamlit Dashboard

This dashboard implements Part E (Web Demo) and Part D (Evaluation Analytics)
from the R&D Technical Assessment. It connects directly to the existing
PostgreSQL database to display pipeline results, evaluation metrics, and
real-time job monitoring.

Launch:
    streamlit run demo/app.py
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import psycopg2.extras
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
#  Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "localpassword"),
    "dbname": os.getenv("POSTGRES_DB", "invoice_extraction"),
}

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
#  Database Helpers (synchronous psycopg2 — Streamlit is sync)
# ---------------------------------------------------------------------------

def get_db_connection():
    """Create a new psycopg2 connection using env-based config."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_all_jobs():
    """Fetch all jobs ordered by creation time descending."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, tenant_id, status, input_file_path,
                       confidence_score, ocr_data, extraction_data,
                       evaluation_data, ground_truth, error_message,
                       retry_count, created_at, updated_at
                FROM jobs
                ORDER BY created_at DESC
            """)
            return cur.fetchall()
    finally:
        conn.close()


def fetch_job_by_id(job_id: str):
    """Fetch a single job by UUID."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM jobs WHERE id = %s",
                (job_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def fetch_done_jobs_with_evaluation():
    """Fetch completed jobs that have evaluation data."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, status, confidence_score, extraction_data,
                       evaluation_data, ground_truth, ocr_data,
                       created_at, updated_at
                FROM jobs
                WHERE status = 'done' AND evaluation_data IS NOT NULL
                ORDER BY created_at DESC
            """)
            return cur.fetchall()
    finally:
        conn.close()


def count_jobs_by_status():
    """Count jobs grouped by status."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status, COUNT(*) as cnt
                FROM jobs
                GROUP BY status
                ORDER BY cnt DESC
            """)
            return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#A0AEC0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
        colorway=[
            "#6366F1", "#8B5CF6", "#A78BFA", "#14B8A6",
            "#2DD4BF", "#F59E0B", "#EF4444", "#EC4899",
        ],
    )
)


# ---------------------------------------------------------------------------
#  Custom CSS Injection — Premium Obsidian Theme
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    /* ── Root Variables ───────────────────────────────── */
    :root {
        --bg-primary: #0D0E12;
        --bg-secondary: #13141A;
        --surface: rgba(30, 32, 42, 0.65);
        --surface-border: rgba(255, 255, 255, 0.08);
        --surface-hover: rgba(40, 42, 55, 0.8);
        --text-primary: #E2E8F0;
        --text-secondary: #A0AEC0;
        --text-muted: #636B7D;
        --accent-indigo: #6366F1;
        --accent-violet: #8B5CF6;
        --accent-teal: #14B8A6;
        --accent-amber: #F59E0B;
        --accent-rose: #EF4444;
        --gradient-primary: linear-gradient(135deg, #6366F1, #8B5CF6);
        --gradient-teal: linear-gradient(135deg, #14B8A6, #2DD4BF);
        --gradient-amber: linear-gradient(135deg, #F59E0B, #FBBF24);
        --gradient-rose: linear-gradient(135deg, #EF4444, #F87171);
        --radius: 12px;
        --radius-lg: 16px;
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
        --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global Overrides ─────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(13, 14, 18, 0.85) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--surface-border);
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--surface-border);
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    /* ── Typography ────────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    h1 { font-weight: 700 !important; font-size: 1.85rem !important; }
    h2 { font-weight: 600 !important; font-size: 1.4rem !important; }
    h3 { font-weight: 600 !important; font-size: 1.15rem !important; }

    p, span, li, label, div {
        color: var(--text-secondary);
    }

    /* ── Glass Cards ──────────────────────────────────── */
    .glass-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: var(--radius-lg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: var(--transition);
    }
    .glass-card:hover {
        background: var(--surface-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* ── Metric Cards ─────────────────────────────────── */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: var(--radius-lg);
        backdrop-filter: blur(16px);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-card);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .metric-card.indigo::before { background: var(--gradient-primary); }
    .metric-card.teal::before { background: var(--gradient-teal); }
    .metric-card.amber::before { background: var(--gradient-amber); }
    .metric-card.rose::before { background: var(--gradient-rose); }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* ── Status Badges ────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-done { background: rgba(20, 184, 166, 0.15); color: #2DD4BF; }
    .badge-processing { background: rgba(99, 102, 241, 0.15); color: #818CF8; }
    .badge-failed { background: rgba(239, 68, 68, 0.15); color: #F87171; }
    .badge-queued { background: rgba(245, 158, 11, 0.15); color: #FBBF24; }
    .badge-pass { background: rgba(20, 184, 166, 0.15); color: #2DD4BF; }
    .badge-fail { background: rgba(239, 68, 68, 0.15); color: #F87171; }

    /* ── Data Tables ──────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.6rem 1.5rem !important;
        transition: var(--transition) !important;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* ── File Uploader ────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--surface-border) !important;
        border-radius: var(--radius-lg) !important;
        background: var(--surface) !important;
        transition: var(--transition);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-indigo) !important;
    }

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: var(--bg-secondary) !important;
        border-radius: var(--radius) !important;
        padding: 6px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius) !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        padding: 8px 16px !important;
        background: transparent !important;
        transition: var(--transition) !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid var(--surface-border) !important;
    }

    /* ── JSON Display ─────────────────────────────────── */
    [data-testid="stJson"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Expander ──────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Select box ───────────────────────────────────── */
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label {
        color: var(--text-secondary) !important;
    }

    /* ── Divider ──────────────────────────────────────── */
    hr {
        border-color: var(--surface-border) !important;
        opacity: 0.5;
    }

    /* ── Hide default Streamlit branding ──────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ── Alert styling ────────────────────────────────── */
    .alert-banner {
        padding: 0.85rem 1.2rem;
        border-radius: var(--radius);
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .alert-warn {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #FBBF24;
    }
    .alert-ok {
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.25);
        color: #2DD4BF;
    }
    .alert-err {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #F87171;
    }

    /* ── Scrollbar ─────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--surface-border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  UI Component Helpers
# ---------------------------------------------------------------------------

def render_metric_card(label: str, value: str, sub: str = "", accent: str = "indigo"):
    """Render a premium metric card with accent stripe."""
    st.markdown(f"""
    <div class="metric-card {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Return HTML for a color-coded status badge."""
    css_map = {
        "done": "badge-done",
        "queued": "badge-queued",
        "failed": "badge-failed",
        "ocr_processing": "badge-processing",
        "ocr_done": "badge-processing",
        "extracting": "badge-processing",
        "extracted": "badge-processing",
        "validating": "badge-processing",
    }
    css = css_map.get(status, "badge-queued")
    return f'<span class="badge {css}">{status}</span>'


def render_pass_fail_badge(passed: bool) -> str:
    """Return HTML for a pass/fail badge."""
    if passed:
        return '<span class="badge badge-pass">PASS</span>'
    return '<span class="badge badge-fail">FAIL</span>'


# ---------------------------------------------------------------------------
#  PAGE 1: Interactive Demo
# ---------------------------------------------------------------------------

def page_interactive_demo():
    st.markdown("## Interactive Demo")
    st.markdown(
        '<p style="color: var(--text-muted); margin-top: -0.5rem;">'
        'Upload an invoice image or PDF to trigger the full OCR + LLM extraction pipeline.'
        '</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your invoice here",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"],
        key="demo_uploader",
    )

    if uploaded is not None:
        # Fetch job details if a job is currently active
        job = None
        job_id = st.session_state.get("demo_job_id")
        if job_id:
            job = fetch_job_by_id(job_id)

        col_preview, col_result = st.columns([1, 1], gap="large")

        with col_preview:
            st.markdown("### Document Preview")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if uploaded.type and uploaded.type.startswith("image"):
                st.image(uploaded, use_container_width=True)
            else:
                st.info(f"Uploaded: **{uploaded.name}** ({uploaded.size:,} bytes)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display OCR Data under Document Preview as soon as it becomes available
            if job and job.get("ocr_data"):
                st.markdown("")
                st.markdown("### OCR Raw Text")
                ocr = job["ocr_data"]
                if isinstance(ocr, dict):
                    conf = ocr.get("average_confidence")
                    if conf is not None:
                        st.markdown(
                            f'<div class="alert-banner alert-ok" style="margin-bottom: 0.5rem; padding: 0.5rem 1rem;">'
                            f'OCR Confidence: {conf:.1%}</div>',
                            unsafe_allow_html=True,
                        )
                    raw_text = ocr.get("raw_text", "")
                    st.code(raw_text if raw_text else json.dumps(ocr, indent=2), language="text")
                else:
                    st.json(ocr)

        with col_result:
            st.markdown("### Pipeline Results")

            # Submit to API
            if "demo_job_id" not in st.session_state:
                st.session_state.demo_job_id = None
                st.session_state.demo_status = None

            if st.button("Run Pipeline", key="btn_run_pipeline"):
                with st.spinner("Submitting to pipeline..."):
                    try:
                        uploaded.seek(0)
                        resp = requests.post(
                            f"{API_BASE}/ingest",
                            files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                            timeout=30,
                        )
                        if resp.status_code in (200, 202):
                            data = resp.json()
                            st.session_state.demo_job_id = data.get("job_id")
                            st.session_state.demo_status = "queued"
                            st.rerun()
                        else:
                            st.error(f"API returned {resp.status_code}: {resp.text}")
                    except requests.ConnectionError:
                        st.error(
                            "Cannot connect to the API server. "
                            "Make sure the FastAPI server is running on " + API_BASE
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

            # Poll & display results
            if st.session_state.demo_job_id:
                st.markdown(
                    f'<p style="font-size: 0.8rem; color: var(--text-muted);">'
                    f'Job ID: <code>{job_id}</code></p>',
                    unsafe_allow_html=True,
                )

                if job:
                    status = job["status"]
                    st.markdown(
                        f"Status: {render_status_badge(status)}",
                        unsafe_allow_html=True,
                    )

                    if status in ("ocr_processing", "extracting", "validating", "queued", "ocr_done", "extracted"):
                        st.info("Pipeline is processing. Click **Refresh** to check progress.")
                        if st.button("Refresh", key="btn_refresh"):
                            st.rerun()

                    elif status == "done":
                        tabs = st.tabs(["Structured Output", "OCR Data", "Evaluation"])

                        with tabs[0]:
                            if job.get("extraction_data"):
                                st.json(job["extraction_data"])
                            else:
                                st.warning("No extraction data available.")

                        with tabs[1]:
                            if job.get("ocr_data"):
                                ocr = job["ocr_data"]
                                if isinstance(ocr, dict):
                                    conf = ocr.get("average_confidence")
                                    if conf is not None:
                                        st.markdown(
                                            f'<div class="alert-banner alert-ok">'
                                            f'OCR Confidence: {conf:.1%}</div>',
                                            unsafe_allow_html=True,
                                        )
                                    raw_text = ocr.get("raw_text", "")
                                    st.code(raw_text if raw_text else json.dumps(ocr, indent=2), language="text")
                                else:
                                    st.json(ocr)
                            else:
                                st.warning("No OCR data available.")

                        with tabs[2]:
                            if job.get("evaluation_data"):
                                ev = job["evaluation_data"]
                                passed = ev.get("passed", False)
                                accs = ev.get("field_accuracies", {})
                                st.markdown(
                                    f"Result: {render_pass_fail_badge(passed)}",
                                    unsafe_allow_html=True,
                                )
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    v = accs.get("invoice_no", 0)
                                    render_metric_card("Invoice No", f"{v:.0%}", accent="indigo" if v == 1 else "rose")
                                with c2:
                                    v = accs.get("invoice_date", 0)
                                    render_metric_card("Invoice Date", f"{v:.0%}", accent="indigo" if v == 1 else "rose")
                                with c3:
                                    v = accs.get("total_net_worth", 0)
                                    render_metric_card("Total Net Worth", f"{v:.0%}", accent="indigo" if v == 1 else "rose")
                            else:
                                st.markdown(
                                    '<div class="alert-banner alert-warn">'
                                    'No ground truth available for evaluation.</div>',
                                    unsafe_allow_html=True,
                                )

                    elif status == "failed":
                        st.markdown(
                            f'<div class="alert-banner alert-err">'
                            f'Pipeline failed: {job.get("error_message", "Unknown error")}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("Job not found in database.")

    # ── Also let users check an existing Job ID ─────────────
    st.markdown("---")
    st.markdown("### Lookup Existing Job")
    lookup_id = st.text_input("Enter a Job ID", key="lookup_job_id", placeholder="e.g. 1c63a2a3-2395-4dc6-9259-...")
    if lookup_id:
        try:
            uuid.UUID(lookup_id)
            job = fetch_job_by_id(lookup_id)
            if job:
                st.markdown(f"Status: {render_status_badge(job['status'])}", unsafe_allow_html=True)
                tabs = st.tabs(["Extraction", "OCR", "Evaluation", "Ground Truth", "Raw Record"])
                with tabs[0]:
                    st.json(job.get("extraction_data") or {"message": "No extraction data"})
                with tabs[1]:
                    if job.get("ocr_data") and isinstance(job["ocr_data"], dict):
                        st.code(job["ocr_data"].get("raw_text", json.dumps(job["ocr_data"], indent=2)), language="text")
                    else:
                        st.json(job.get("ocr_data") or {"message": "No OCR data"})
                with tabs[2]:
                    st.json(job.get("evaluation_data") or {"message": "No evaluation data"})
                with tabs[3]:
                    st.json(job.get("ground_truth") or {"message": "No ground truth"})
                with tabs[4]:
                    # Build a clean dict for display (handle non-serializable fields)
                    raw = {}
                    for k, v in job.items():
                        if isinstance(v, (datetime,)):
                            raw[k] = v.isoformat()
                        elif isinstance(v, uuid.UUID):
                            raw[k] = str(v)
                        else:
                            raw[k] = v
                    st.json(raw)
            else:
                st.warning("No job found with that ID.")
        except ValueError:
            st.error("Invalid UUID format.")


# ---------------------------------------------------------------------------
#  PAGE 2: Evaluation Analytics
# ---------------------------------------------------------------------------

def page_evaluation_analytics():
    st.markdown("## Evaluation Analytics")
    st.markdown(
        '<p style="color: var(--text-muted); margin-top: -0.5rem;">'
        'Performance metrics and accuracy analysis across all evaluated invoices.'
        '</p>',
        unsafe_allow_html=True,
    )

    jobs = fetch_done_jobs_with_evaluation()

    if not jobs:
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:3rem;">'
            '<p style="color: var(--text-muted); font-size: 1rem;">No evaluated jobs found.</p>'
            '<p style="color: var(--text-muted); font-size: 0.85rem;">'
            'Run the pipeline with ground truth data to see evaluation metrics here.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Parse evaluation data
    total = len(jobs)
    passed_count = 0
    inv_no_acc = []
    inv_date_acc = []
    net_worth_acc = []
    confidence_scores = []
    processing_times = []

    for job in jobs:
        ev = job.get("evaluation_data") or {}
        if ev.get("passed"):
            passed_count += 1
        accs = ev.get("field_accuracies", {})
        inv_no_acc.append(accs.get("invoice_no", 0.0))
        inv_date_acc.append(accs.get("invoice_date", 0.0))
        net_worth_acc.append(accs.get("total_net_worth", 0.0))

        if job.get("confidence_score") is not None:
            confidence_scores.append(job["confidence_score"])

        if job.get("created_at") and job.get("updated_at"):
            delta = (job["updated_at"] - job["created_at"]).total_seconds()
            if delta >= 0:
                processing_times.append(delta)

    pass_rate = passed_count / total if total > 0 else 0
    avg_no = sum(inv_no_acc) / len(inv_no_acc) if inv_no_acc else 0
    avg_date = sum(inv_date_acc) / len(inv_date_acc) if inv_date_acc else 0
    avg_net = sum(net_worth_acc) / len(net_worth_acc) if net_worth_acc else 0

    # ── KPI Cards ──────────────────────────────────────
    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Total Evaluated", str(total),
            f"{passed_count} passed, {total - passed_count} failed",
            accent="indigo",
        )
    with c2:
        render_metric_card(
            "Pass Rate", f"{pass_rate:.1%}",
            "All 3 mandatory fields correct",
            accent="teal" if pass_rate >= 0.7 else "amber" if pass_rate >= 0.4 else "rose",
        )
    with c3:
        avg_overall = (avg_no + avg_date + avg_net) / 3
        render_metric_card(
            "Average Accuracy", f"{avg_overall:.1%}",
            "Mean across 3 fields",
            accent="teal" if avg_overall >= 0.7 else "amber",
        )
    with c4:
        avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        render_metric_card(
            "OCR Confidence", f"{avg_conf:.1%}" if confidence_scores else "N/A",
            f"{len(confidence_scores)} samples",
            accent="indigo",
        )

    st.markdown("")

    # ── Per-Field Accuracy ─────────────────────────────
    st.markdown("### Field-Level Accuracy")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Invoice Number", f"{avg_no:.1%}",
            f"{sum(1 for x in inv_no_acc if x == 1.0)}/{total} correct",
            accent="indigo" if avg_no >= 0.7 else "rose",
        )
    with c2:
        render_metric_card(
            "Invoice Date", f"{avg_date:.1%}",
            f"{sum(1 for x in inv_date_acc if x == 1.0)}/{total} correct",
            accent="indigo" if avg_date >= 0.7 else "rose",
        )
    with c3:
        render_metric_card(
            "Total Net Worth", f"{avg_net:.1%}",
            f"{sum(1 for x in net_worth_acc if x == 1.0)}/{total} correct",
            accent="indigo" if avg_net >= 0.7 else "rose",
        )

    st.markdown("")

    # ── Plotly Charts ──────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("### Accuracy Distribution")
        field_names = ["Invoice No"] * total + ["Invoice Date"] * total + ["Total Net Worth"] * total
        field_values = inv_no_acc + inv_date_acc + net_worth_acc

        fig_acc = go.Figure()
        for field, values, color in [
            ("Invoice No", inv_no_acc, "#6366F1"),
            ("Invoice Date", inv_date_acc, "#8B5CF6"),
            ("Total Net Worth", net_worth_acc, "#14B8A6"),
        ]:
            correct = sum(1 for v in values if v == 1.0)
            incorrect = len(values) - correct
            fig_acc.add_trace(go.Bar(
                name=field,
                x=[field],
                y=[correct],
                marker_color=color,
                text=[f"{correct}/{len(values)}"],
                textposition="auto",
                textfont=dict(color="white", size=13),
            ))

        fig_acc.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis_title="Correct Extractions",
            bargap=0.35,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_right:
        st.markdown("### Processing Latency")
        if processing_times:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Histogram(
                x=processing_times,
                nbinsx=20,
                marker_color="#14B8A6",
                opacity=0.85,
            ))
            fig_lat.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Seconds",
                yaxis_title="Number of Jobs",
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.markdown(
                '<div class="glass-card" style="text-align:center; padding:2rem;">'
                '<p style="color: var(--text-muted);">No timing data available.</p></div>',
                unsafe_allow_html=True,
            )

    # ── OCR Confidence Distribution ────────────────────
    if confidence_scores:
        st.markdown("### OCR Confidence Distribution")
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Histogram(
            x=confidence_scores,
            nbinsx=25,
            marker=dict(
                color="rgba(99, 102, 241, 0.7)",
                line=dict(color="#6366F1", width=1),
            ),
        ))
        fig_conf.update_layout(
            template=PLOTLY_TEMPLATE,
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Detailed Results Table ─────────────────────────
    st.markdown("### Detailed Results")
    table_data = []
    for job in jobs:
        ev = job.get("evaluation_data") or {}
        accs = ev.get("field_accuracies", {})
        table_data.append({
            "Job ID": str(job["id"]),
            "Passed": "PASS" if ev.get("passed") else "FAIL",
            "Invoice No": "correct" if accs.get("invoice_no", 0) == 1 else "wrong",
            "Invoice Date": "correct" if accs.get("invoice_date", 0) == 1 else "wrong",
            "Total Net Worth": "correct" if accs.get("total_net_worth", 0) == 1 else "wrong",
            "Confidence": f"{job.get('confidence_score', 0) or 0:.2%}",
            "Created": job["created_at"].strftime("%Y-%m-%d %H:%M") if job.get("created_at") else "-",
        })

    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
#  PAGE 3: Pipeline Monitor
# ---------------------------------------------------------------------------

def page_pipeline_monitor():
    st.markdown("## Pipeline Monitor")
    st.markdown(
        '<p style="color: var(--text-muted); margin-top: -0.5rem;">'
        'Real-time view of pipeline health, queue status, and job history.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Status Counts ──────────────────────────────────
    st.markdown("### Queue Health")
    status_counts = count_jobs_by_status()
    all_jobs = fetch_all_jobs()

    if status_counts:
        cols = st.columns(min(len(status_counts), 4))
        accent_map = {
            "done": "teal", "queued": "amber", "failed": "rose",
            "ocr_processing": "indigo", "extracting": "indigo",
            "validating": "indigo", "ocr_done": "indigo",
            "extracted": "indigo",
        }
        for i, row in enumerate(status_counts):
            with cols[i % len(cols)]:
                render_metric_card(
                    row["status"].upper().replace("_", " "),
                    str(row["cnt"]),
                    accent=accent_map.get(row["status"], "indigo"),
                )
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:2rem;">'
            '<p style="color: var(--text-muted);">No jobs in the pipeline.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Warning Deck: stuck jobs ───────────────────────
    now = datetime.now(timezone.utc)
    stuck_threshold = timedelta(minutes=10)
    stuck_jobs = []
    processing_statuses = {"queued", "ocr_processing", "extracting", "validating", "ocr_done", "extracted"}

    for job in all_jobs:
        if job["status"] in processing_statuses:
            updated = job.get("updated_at")
            if updated:
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                if (now - updated) > stuck_threshold:
                    stuck_jobs.append(job)

    if stuck_jobs:
        st.markdown("### Warnings")
        
        ITEMS_PER_PAGE = 5
        total_items = len(stuck_jobs)
        total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1
        
        if "warning_page" not in st.session_state:
            st.session_state.warning_page = 1
            
        if st.session_state.warning_page > total_pages:
            st.session_state.warning_page = total_pages
        if st.session_state.warning_page < 1:
            st.session_state.warning_page = 1
            
        start_idx = (st.session_state.warning_page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        
        for sj in stuck_jobs[start_idx:end_idx]:
            age = (now - sj["updated_at"].replace(tzinfo=timezone.utc)).total_seconds() / 60
            st.markdown(
                f'<div class="alert-banner alert-warn">'
                f'Job <code>{str(sj["id"])}</code> has been in '
                f'<strong>{sj["status"]}</strong> for {age:.0f} min</div>',
                unsafe_allow_html=True,
            )
            
        if total_pages > 1:
            col_prev, col_page, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("← Prev Warnings", disabled=(st.session_state.warning_page == 1)):
                    st.session_state.warning_page -= 1
                    st.rerun()
            with col_page:
                st.markdown(
                    f'<p style="text-align: center; margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted);">'
                    f'Page {st.session_state.warning_page} of {total_pages} ({total_items} total)</p>',
                    unsafe_allow_html=True
                )
            with col_next:
                if st.button("Next Warnings →", disabled=(st.session_state.warning_page == total_pages)):
                    st.session_state.warning_page += 1
                    st.rerun()

    # ── Job History Table ──────────────────────────────
    st.markdown("### Job History")

    if all_jobs:
        import pandas as pd

        table_rows = []
        for job in all_jobs:
            created = job["created_at"]
            updated = job["updated_at"]
            duration = ""
            if created and updated:
                delta = (updated - created).total_seconds()
                if delta >= 0:
                    duration = f"{delta:.1f}s"

            table_rows.append({
                "Job ID": str(job["id"]),
                "Status": job["status"],
                "File": job.get("input_file_path", "")[:40],
                "Confidence": f"{job['confidence_score']:.2%}" if job.get("confidence_score") else "-",
                "Duration": duration,
                "Error": (job.get("error_message") or "-")[:60],
                "Created": created.strftime("%Y-%m-%d %H:%M:%S") if created else "-",
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

        # ── Details Inspector ──────────────────────────
        st.markdown("### Details Inspector")
        job_ids = [str(j["id"]) for j in all_jobs]
        selected = st.selectbox(
            "Select a Job ID to inspect",
            options=job_ids,
            format_func=lambda x: x,
            key="inspector_select",
        )

        if selected:
            job = fetch_job_by_id(selected)
            if job:
                detail_tabs = st.tabs(["Extraction", "OCR Data", "Evaluation", "Ground Truth", "Full Record"])
                with detail_tabs[0]:
                    st.json(job.get("extraction_data") or {"message": "No extraction data"})
                with detail_tabs[1]:
                    ocr = job.get("ocr_data")
                    if ocr and isinstance(ocr, dict):
                        conf = ocr.get("average_confidence")
                        if conf is not None:
                            st.markdown(
                                f'<div class="alert-banner alert-ok">'
                                f'Average Confidence: {conf:.1%}</div>',
                                unsafe_allow_html=True,
                            )
                        st.code(ocr.get("raw_text", json.dumps(ocr, indent=2)), language="text")
                    else:
                        st.json(ocr or {"message": "No OCR data"})
                with detail_tabs[2]:
                    st.json(job.get("evaluation_data") or {"message": "No evaluation data"})
                with detail_tabs[3]:
                    st.json(job.get("ground_truth") or {"message": "No ground truth"})
                with detail_tabs[4]:
                    raw = {}
                    for k, v in job.items():
                        if isinstance(v, datetime):
                            raw[k] = v.isoformat()
                        elif isinstance(v, uuid.UUID):
                            raw[k] = str(v)
                        else:
                            raw[k] = v
                    st.json(raw)
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center; padding:2rem;">'
            '<p style="color: var(--text-muted);">No jobs recorded yet.</p></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
#  MAIN APPLICATION
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Invoice Extraction System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # ── Sidebar Navigation ─────────────────────────────
    with st.sidebar:
        st.markdown(
            '<h1 style="font-size: 1.4rem; margin-bottom: 0.2rem;">Invoice Extraction</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1.5rem;">'
            'OCR + LLM Pipeline Dashboard</p>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            options=["Interactive Demo", "Evaluation Analytics", "Pipeline Monitor"],
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size: 0.72rem; color: var(--text-muted);">'
            'R&D Technical Assessment<br>Brian Lee &copy; 2026</p>',
            unsafe_allow_html=True,
        )

    # ── Page Routing ───────────────────────────────────
    if page == "Interactive Demo":
        page_interactive_demo()
    elif page == "Evaluation Analytics":
        page_evaluation_analytics()
    elif page == "Pipeline Monitor":
        page_pipeline_monitor()


if __name__ == "__main__":
    main()
