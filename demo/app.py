import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import io
import time
import uuid
from datetime import datetime, timedelta, timezone
from PIL import Image
from sqlalchemy import create_engine, text
import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings

# ── Headless Page Configurations ─────────────────────────
st.set_page_config(
    page_title="Invoice Extraction Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.2rem;
        color: #f8fafc;
        font-weight: 700;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .status-badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-done { background-color: #10b981; color: white; }
    .status-processing { background-color: #3b82f6; color: white; }
    .status-failed { background-color: #ef4444; color: white; }
    .status-queued { background-color: #f59e0b; color: white; }
</style>
""", unsafe_allow_html=True)

# ── Load Settings & Infrastructure Clients ───────────────
@st.cache_resource
def get_infra_clients():
    settings = get_settings()
    
    # Sync PostgreSQL Engine
    sync_url = settings.database.dsn.replace("postgresql+asyncpg", "postgresql")
    engine = create_engine(sync_url)
    
    # S3 Client (MinIO)
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.minio.endpoint,
        aws_access_key_id=settings.minio.access_key,
        aws_secret_access_key=settings.minio.secret_key,
        use_ssl=settings.minio.secure,
        region_name="us-east-1"
    )
    
    return settings, engine, s3_client

settings, db_engine, s3_client = get_infra_clients()

# ── Data Fetching Helpers ────────────────────────────────
def fetch_jobs_data(limit=100) -> pd.DataFrame:
    query = f"""
        SELECT id, tenant_id, status, input_file_path, ocr_output_path, 
               extraction_output_path, confidence_score, ocr_data, extraction_data, 
               evaluation_data, ground_truth, error_message, created_at, updated_at
        FROM jobs
        ORDER BY created_at DESC
        LIMIT {limit}
    """
    with db_engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def fetch_job_by_id(job_id: str) -> dict:
    query = text("""
        SELECT id, tenant_id, status, input_file_path, ocr_output_path, 
               extraction_output_path, confidence_score, ocr_data, extraction_data, 
               evaluation_data, ground_truth, error_message, created_at, updated_at
        FROM jobs
        WHERE id = :job_id
    """)
    with db_engine.connect() as conn:
        result = conn.execute(query, {"job_id": job_id}).mappings().first()
    return dict(result) if result else None

def get_minio_object(bucket: str, path: str) -> bytes:
    # If the path starts with bucket name (e.g. "invoices/"), strip it out to get the correct object key
    if path.startswith(f"{bucket}/"):
        path = path.split("/", 1)[1]
    try:
        response = s3_client.get_object(Bucket=bucket, Key=path)
        return response["Body"].read()
    except ClientError as e:
        st.warning(f"Could not fetch {path} from MinIO: {e}")
        return None

# ── Sidebar Navigation ───────────────────────────────────
st.sidebar.image("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=256&auto=format&fit=crop", width="stretch")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🚀 Interactive Demo", "📊 Evaluation Metrics", "🖥️ Pipeline Monitor"])

# ── Render View 1: Interactive Demo ──────────────────────
if page == "🚀 Interactive Demo":
    st.title("🚀 Interactive Invoice Demo")
    st.write("Upload an invoice to watch the OCR + LLM pipeline run in real-time.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Ingestion File Upload")
        uploaded_file = st.file_uploader("Choose an invoice image or PDF", type=["png", "jpg", "jpeg", "pdf"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Document Preview", width="stretch")
            if st.button("Trigger Processing Pipeline", type="primary"):
                # Hit API Ingest Endpoint
                import os
                api_base = os.environ.get("INGESTION_API_URL", "http://localhost:8000")
                api_url = f"{api_base}/ingest"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                with st.spinner("Uploading to ingestion API..."):
                    try:
                        res = requests.post(api_url, files=files)
                        if res.status_code == 202:
                            job_data = res.json()
                            job_id = job_data.get("job_id")
                            st.success(f"Accepted! Job ID: {job_id}")
                            st.session_state["current_job_id"] = job_id
                        else:
                            st.error(f"API upload failed: {res.text}")
                    except Exception as e:
                        st.error(f"Could not connect to API Server: {e}")

    with col2:
        st.subheader("2. Processing Pipeline Live State")
        if "current_job_id" in st.session_state:
            job_id = st.session_state["current_job_id"]
            
            # Polling Loop
            status_container = st.empty()
            progress_bar = st.progress(0.0)
            
            terminal_states = ["done", "failed"]
            job = None
            
            with st.spinner("Polling pipeline progress..."):
                for attempt in range(200): # 5 minutes timeout (200 * 1.5s)
                    job = fetch_job_by_id(job_id)
                    if not job:
                        status_container.error("Job record not found in database yet.")
                        break
                    
                    status = job["status"]
                    
                    # Update status indicator
                    status_container.markdown(f"**Current Status:** `{status.upper()}`")
                    
                    # Progress bar fraction
                    state_map = {
                        "queued": 0.1,
                        "ocr_processing": 0.3,
                        "ocr_done": 0.5,
                        "extracting": 0.7,
                        "extracted": 0.9,
                        "done": 1.0,
                        "failed": 1.0
                    }
                    progress_bar.progress(state_map.get(status, 0.0))
                    
                    if status in terminal_states:
                        break
                        
                    time.sleep(1.5)
            
            if job:
                st.write("---")
                st.subheader("3. Pipeline Outputs")
                
                # Check outcome
                if job["status"] == "done":
                    tab1, tab2, tab3 = st.tabs(["📄 OCR Output Text", "🤖 LLM Structured Extraction", "🔍 Business Validations"])
                    
                    with tab1:
                        st.write(f"**Average OCR Confidence:** `{job['confidence_score']:.2%}`")
                        if job.get("ocr_data"):
                            st.text_area("OCR Raw Text Lines", job["ocr_data"].get("raw_text", ""), height=300)
                        elif job["ocr_output_path"]:
                            ocr_bytes = get_minio_object("invoices", job["ocr_output_path"])
                            if ocr_bytes:
                                try:
                                    ocr_data = json.loads(ocr_bytes.decode("utf-8"))
                                    st.text_area("OCR Raw Text Lines", ocr_data.get("raw_text", ""), height=300)
                                except Exception as e:
                                    st.error(f"Error reading OCR details: {e}")
                        else:
                            st.info("No OCR data or file path found.")
                            
                    with tab2:
                        st.write("Parsed Structured fields (Pydantic model-validated schema):")
                        if job["extraction_data"]:
                            st.json(job["extraction_data"])
                        else:
                            st.warning("Extraction output is empty.")
                            
                    with tab3:
                        st.write("### Validation Flags:")
                        
                        # Recalculate validations for demo display
                        ext_data = job["extraction_data"]
                        if ext_data:
                            # VAT check
                            summary = ext_data.get("summary", {})
                            net_str = summary.get("total_net_worth")
                            vat_str = summary.get("total_vat")
                            gross_str = summary.get("total_gross_worth")
                            
                            from src.services.output.validator import parse_raw_amount
                            net_val = parse_raw_amount(net_str)
                            vat_val = parse_raw_amount(vat_str)
                            gross_val = parse_raw_amount(gross_str)
                            
                            if net_val is not None and vat_val is not None and gross_val is not None:
                                expected_gross = net_val + vat_val
                                diff = abs(expected_gross - gross_val)
                                if diff > 0.05:
                                    st.warning(f"⚠️ **VAT Math Mismatch:** Net ({net_val}) + VAT ({vat_val}) = Expected Gross ({expected_gross:.2f}), but Extracted Gross is ({gross_val}). Diff: {diff:.2f}")
                                else:
                                    st.success("✅ **VAT Math Match:** Net + VAT ≈ Gross worth is correct.")
                            
                            # Items check
                            items = ext_data.get("items", [])
                            if items:
                                items_sum = 0.0
                                has_item_net = False
                                for item in items:
                                    item_net = parse_raw_amount(item.get("item_net_worth"))
                                    if item_net is not None:
                                        items_sum += item_net
                                        has_item_net = True
                                
                                if has_item_net and net_val is not None:
                                    diff_items = abs(items_sum - net_val)
                                    if diff_items > 0.05:
                                        st.warning(f"⚠️ **Line Items Sum Discrepancy:** Sum of line items net worth is ({items_sum:.2f}), but Invoice Total Net Worth is ({net_val:.2f}). Diff: {diff_items:.2f}")
                                    else:
                                        st.success("✅ **Line Item Net Sum Match:** Sum of item net worths equals invoice net worth.")
                            
                            # Ground truth check
                            eval_data = job["evaluation_data"]
                            if eval_data and eval_data.get("evaluated"):
                                st.write("### Evaluation against Ground Truth:")
                                passed = eval_data.get("passed")
                                if passed:
                                    st.success("🏆 **Evaluation passed!** All three mandatory fields match ground truth exactly.")
                                else:
                                    st.error("❌ **Evaluation failed.** Some mandatory fields do not match ground truth.")
                                st.write(eval_data.get("field_accuracies"))
                                
                elif job["status"] == "failed":
                    st.error(f"Pipeline processing failed: {job['error_message']}")
        else:
            st.info("Upload an invoice file and trigger the pipeline to begin.")

# ── Render View 2: Evaluation Metrics ────────────────────
elif page == "📊 Evaluation Metrics":
    st.title("📊 Evaluation & Accuracy Analytics")
    st.write("Aggregated system-level performance metrics based on dataset ground-truth evaluations.")
    
    # Fetch Data
    df_all = fetch_jobs_data(1000)
    
    # Filter Evaluated Runs
    evaluated_jobs = []
    for idx, row in df_all.iterrows():
        eval_dict = row["evaluation_data"]
        if eval_dict and eval_dict.get("evaluated"):
            evaluated_jobs.append(row)
            
    df_eval = pd.DataFrame(evaluated_jobs)
    
    if df_eval.empty:
        st.warning("No dataset runs with ground-truth evaluations found in the database yet. Trigger dataset ingestion via POST `/ingest/dataset`.")
    else:
        # Compute Analytics
        total_eval = len(df_eval)
        
        # Calculate Pass Rate
        passed_count = sum(df_eval["evaluation_data"].apply(lambda x: 1 if x.get("passed") else 0))
        pass_rate = passed_count / total_eval if total_eval > 0 else 0.0
        
        # Field accuracies
        acc_no = df_eval["evaluation_data"].apply(lambda x: x.get("field_accuracies", {}).get("invoice_no", 0.0)).mean()
        acc_date = df_eval["evaluation_data"].apply(lambda x: x.get("field_accuracies", {}).get("invoice_date", 0.0)).mean()
        acc_net = df_eval["evaluation_data"].apply(lambda x: x.get("field_accuracies", {}).get("total_net_worth", 0.0)).mean()
        
        # Render Metric Card Grid
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Evaluated Invoices</div><div class="metric-value">{total_eval}</div><div class="metric-subtitle">Dataset runs</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mandatory Pass Rate</div><div class="metric-value">{pass_rate:.1%}</div><div class="metric-subtitle">All 3 matched</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Invoice No Acc</div><div class="metric-value">{acc_no:.1%}</div><div class="metric-subtitle">Field accuracy</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Invoice Date Acc</div><div class="metric-value">{acc_date:.1%}</div><div class="metric-subtitle">Field accuracy</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Net Worth Acc</div><div class="metric-value">{acc_net:.1%}</div><div class="metric-subtitle">Field accuracy</div></div>', unsafe_allow_html=True)
            
        st.write("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("OCR Confidence Distribution")
            df_confidence = df_all[df_all["confidence_score"].notnull()]
            if not df_confidence.empty:
                fig_conf = px.histogram(
                    df_confidence,
                    x="confidence_score",
                    nbins=20,
                    title="OCR Average Confidence Scores",
                    color_discrete_sequence=["#38bdf8"],
                    labels={"confidence_score": "Confidence Score"}
                )
                fig_conf.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                )
                st.plotly_chart(fig_conf, width="stretch")
            else:
                st.info("No confidence scores to plot.")
                
        with col2:
            st.subheader("Processing Latency Distribution")
            
            # Filter terminal states with valid timestamps
            df_latency = df_all[df_all["status"].isin(["done", "failed"])].copy()
            df_latency["latency_sec"] = (df_latency["updated_at"] - df_latency["created_at"]).dt.total_seconds()
            df_latency = df_latency[df_latency["latency_sec"] > 0]
            
            if not df_latency.empty:
                fig_lat = px.histogram(
                    df_latency,
                    x="latency_sec",
                    nbins=20,
                    title="Ingestion to Output Latency (Seconds)",
                    color_discrete_sequence=["#a855f7"],
                    labels={"latency_sec": "Latency (sec)"}
                )
                fig_lat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                )
                st.plotly_chart(fig_lat, width="stretch")
            else:
                st.info("No latency data available.")

# ── Render View 3: Pipeline Monitoring & History ──────────
elif page == "🖥️ Pipeline Monitor":
    st.title("🖥️ Pipeline Health & Jobs History")
    
    # Fetch Data
    df_all = fetch_jobs_data(200)
    
    # ── Queue Health Grid ────────────────────────────────
    status_counts = df_all["status"].value_counts()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Completed (Done)</div><div class="metric-value">{status_counts.get("done", 0)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Processing</div><div class="metric-value">{status_counts.get("ocr_processing", 0) + status_counts.get("extracting", 0)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Failed</div><div class="metric-value" style="color: #ef4444;">{status_counts.get("failed", 0)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Queued</div><div class="metric-value" style="color: #f59e0b;">{status_counts.get("queued", 0)}</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    # ── Stuck Jobs Warnings ──────────────────────────────
    stuck_limit = datetime.now(timezone.utc) - timedelta(minutes=10)
    stuck_jobs = []
    
    for idx, row in df_all.iterrows():
        # SQLAlchemy dates are naive but UTC-based
        updated_at = row["updated_at"].replace(tzinfo=timezone.utc)
        if row["status"] not in ["done", "failed"] and updated_at < stuck_limit:
            stuck_jobs.append(row)
            
    if stuck_jobs:
        st.warning(f"🚨 **Alert:** Detected {len(stuck_jobs)} jobs stuck in processing for more than 10 minutes!")
        st.dataframe(pd.DataFrame(stuck_jobs)[["id", "status", "updated_at"]])
        
    # ── Searchable jobs log table ────────────────────────
    st.subheader("Jobs History Log")
    
    # Filters
    search_query = st.text_input("🔍 Search by Job ID or File Path")
    status_filter = st.selectbox("Filter by Status", ["All"] + list(df_all["status"].unique()))
    
    df_filtered = df_all.copy()
    
    if search_query:
        df_filtered = df_filtered[
            df_filtered["id"].astype(str).str.contains(search_query, case=False) |
            df_filtered["input_file_path"].str.contains(search_query, case=False)
        ]
        
    if status_filter != "All":
        df_filtered = df_filtered[df_filtered["status"] == status_filter]
        
    st.dataframe(
        df_filtered[["id", "tenant_id", "status", "input_file_path", "confidence_score", "created_at", "updated_at"]],
        width="stretch"
    )
    
    # ── Job Detail Inspector Section ─────────────────────
    st.write("---")
    st.subheader("🔎 Job Details Inspector")
    job_ids = df_filtered["id"].astype(str).tolist()
    
    if job_ids:
        selected_job_id = st.selectbox("Select a Job ID to inspect", job_ids)
        
        if selected_job_id:
            job = fetch_job_by_id(selected_job_id)
            if job:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Job Metadata")
                    st.write(f"**Status:** `{job['status']}`")
                    st.write(f"**Tenant ID:** `{job['tenant_id']}`")
                    st.write(f"**Created At:** `{job['created_at']}`")
                    st.write(f"**Updated At:** `{job['updated_at']}`")
                    if job["error_message"]:
                        st.error(f"**Error Message:** {job['error_message']}")
                        
                    # Fetch Source Image from MinIO
                    st.write("#### Document Image Source")
                    image_bytes = get_minio_object("invoices", job["input_file_path"])
                    if image_bytes:
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            st.image(img, width="stretch")
                        except Exception as e:
                            st.write(f"Could not render image preview: {e}")
                    else:
                        st.info("Source document not found in storage.")
                        
                with col2:
                    st.write("#### Raw OCR Text Output")
                    if job.get("ocr_data"):
                        st.text_area("Raw Text Lines", job["ocr_data"].get("raw_text", ""), height=250)
                    elif job["ocr_output_path"]:
                        ocr_data_bytes = get_minio_object("invoices", job["ocr_output_path"])
                        if ocr_data_bytes:
                            try:
                                ocr_raw = json.loads(ocr_data_bytes.decode("utf-8"))
                                st.text_area("Raw Text Lines", ocr_raw.get("raw_text", ""), height=250)
                            except Exception as e:
                                st.write(f"Error loading OCR text: {e}")
                    else:
                        st.info("No OCR data or file created.")
                        
                    st.write("#### LLM Parsed Extraction Output")
                    if job["extraction_data"]:
                        st.json(job["extraction_data"])
                    else:
                        st.info("No structured data extracted.")
                        
                    if job["evaluation_data"]:
                        st.write("#### Evaluator Result Metrics")
                        st.json(job["evaluation_data"])
    else:
        st.info("No jobs match the filters.")
