# Intelligent Invoice & Receipt Extraction System (OCR + LLM Pipeline)

An end-to-end Document AI pipeline that processes invoice and receipt images/PDFs, extracts raw text via PaddleOCR, structures it into formatted JSON using Qwen (run locally via Ollama / API) as the primary engine (with Gemini API as an optional fallback), evaluates extraction accuracy against ground truth, and monitors pipeline status through a Streamlit dashboard.

---

## Infrastructure Overview

The application is built on a decoupled, asynchronous queue-based architecture:

- **Message Broker (Redis Streams)**: Uses Redis Streams with Consumer Groups. This ensures reliable FIFO processing of jobs and supports task reclamation or retries if a consumer worker crashes mid-task.
- **Relational Database (PostgreSQL)**: Serves as the central state machine and metadata store. It tracks job records (transitions, timing, errors), tenant data, rate-limiting rules, and accuracy scores.
- **Object Storage (MinIO)**: S3-compatible object storage used to persist raw upload files, intermediate OCR output structures, and final extracted JSON outputs.
- **Local LLM Engine (Ollama)**: Houses Qwen (e.g., `qwen2.5`) locally for free-tier text-to-JSON structure parsing, removing dependencies on paid external APIs.

---

## Architecture & Component Design

The system is designed with an asynchronous queue-based microservices architecture using Python 3.12+:

```
                             Python                 Python                Python
                             ──────                 ──────                ──────

     ┌───────────────┐  Queue A  ┌──────────────────┐  Queue B  ┌───────────────┐
     │   INGESTION   │──────────▶│    PROCESSING    │─────────▶│    OUTPUT     │
     │───────────────│           │──────────────────│          │───────────────│
     │ • FastAPI API │           │ • Image Canvas   │          │ • Validation  │
     │ • Dataset run │           │ • PaddleOCR      │          │ • Evaluation  │
     │ • Enqueue job │           │ • Qwen/LLM run   │          │ • Save JSON   │
     └───────────────┘           └──────────────────┘          └───────────────┘
                                         │
                                         │ writes
                                         ▼
            ┌─────────────────────────────────────────────────────────┐
            │                        STORAGE                          │
            │─────────────────────────────────────────────────────────│
            │     Blob Storage (MinIO / S3)     │     Postgres        │
            │     • input.pdf / input.png       │     • jobs          │
            │     • ocr_output.json             │     • tenants       │
            │     • final_extraction.json       │                     │
            └───────────────────────────┬─────────────────────────────┘
                                        │ reads
                                        ▼
                               ┌─────────────────┐
                               │    DASHBOARD    │
                               │   (Streamlit)   │
                               │─────────────────│
                               │ • Interactive   │
                               │   Demo Upload   │
                               │ • Accuracy chart│
                               │ • Jobs monitor  │
                               └─────────────────┘
```

### 1. Ingestion Service (FastAPI)
Exposes REST endpoints to receive document uploads or trigger large batch dataset ingestion from Hugging Face (`mychen76/invoices-and-receipts_ocr_v1`). Jobs are enqueued into a Redis Stream queue (`queue:ingestion`).

### 2. Processing Service (PaddleOCR + Qwen / Gemini)
Consumes from `queue:ingestion`. 
- **OCR Layer**: Runs self-hosted PaddleOCR (PP-OCRv5) combined with dynamic spatial clustering to group words into rows and segment fields.
- **LLM Layer**: Sends raw text to Qwen via a local Ollama instance (primary free-tier) or optionally to Gemini Flash via the `google-genai` SDK. Structures the output according to the Pydantic schema to produce clean JSON.
- Enqueues extraction data into `queue:extraction`.

### 3. Output & Evaluation Service
Consumes from `queue:extraction`. 
- Performs business arithmetic validations (VAT total checks, line item sum validations).
- Computes extraction accuracy scores for key fields (`invoice_no`, `invoice_date`, `total_net_worth`) against dataset ground truths.
- Saves the final JSON results to object storage (MinIO).
- Marks the job state in PostgreSQL as completed (`done`).

### 4. Streamlit Dashboard
A premium Obsidian-themed frontend containing:
- **Interactive Demo**: Drag-and-drop invoices to run the full pipeline in real time with side-by-side visualization of OCR Raw Text and parsed JSON output.
- **Evaluation Analytics**: Graphical summaries of field-level accuracy, pass rate distributions, average confidence, and processing latencies.
- **Pipeline Monitor**: Real-time queue health overview, stuck job warnings (with pagination), and a detailed details inspector tab.

---

## Technology Stack

- **Core**: Python 3.12+
- **OCR Engine**: PaddleOCR (PaddlePaddle 3.2) & PyMuPDF (PDF rendering)
- **LLM Engine**: Qwen (via local Ollama - Primary Free-tier) / Gemini Flash (Optional paid tier)
- **Web Framework**: FastAPI & Uvicorn
- **Task Queue**: Redis Streams & Consumer Groups
- **Database**: PostgreSQL (using async SQLAlchemy 2.0 & Psycopg 3)
- **Object Storage**: MinIO (S3-compatible API)
- **Frontend Dashboard**: Streamlit & Plotly

---

## Setup & Installation

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.12+ (Miniconda / Virtualenv recommended)
- A local Ollama instance (running `qwen2.5` or similar) or an optional Google Gemini API Key.

### 2. Infrastructure Setup (Docker Compose)
Start the local database, Redis queue, object storage, and LLM runner:
```bash
docker compose -f infra/docker-compose.yaml up -d
```
This launches:
- **Postgres** (port `5432`) - Database `invoice_extraction`
- **Redis** (port `6379`) - Message broker
- **MinIO** (port `9000` / `9001`) - Bucket `invoices` for blob files
- **Ollama** (port `11434`) - Containerized local Ollama instance (ready to run local Qwen models)

### 3. Environment Configuration
Copy the sample environment file and configure variables:
```bash
cp .env.example .env
```
Set the configuration values to point to your local Ollama endpoint (default: `http://localhost:11434`) or specify your optional `GEMINI_API_KEY` to enable Gemini.

### 4. Setup Python Environment
Create your virtual environment and install requirements:
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate    # On Linux/macOS

pip install -r requirements.txt
```

---

## Running the Services

Launch each service in separate terminal windows:

### 1. Web API Gateway (FastAPI)
```bash
.venv\Scripts\python.exe -m src.main
```
Runs at: `http://localhost:8000` (docs available at `/docs`).

### 2. Processing Worker (OCR & LLM Extraction)
```bash
.venv\Scripts\python.exe -m src.cli.processing_worker
```
Consumes from `queue:ingestion`.

### 3. Output Worker (Evaluator & Publisher)
```bash
.venv\Scripts\python.exe -m src.cli.output_worker
```
Consumes from `queue:extraction`.

### 4. Streamlit Dashboard
```bash
.venv\Scripts\streamlit.exe run demo/app.py
```
Runs at: `http://localhost:8501`.

---

## Pipeline Commands

Once all services are up, trigger the pipeline using PowerShell or `curl`:

### A. Ingest a Single File
Submit a local file to the pipeline:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/invoice.pdf"
```

### B. Batch Ingest from Dataset (Small Limit)
Trigger ingestion of 3 samples from the `test` split (ideal for testing real-time pipelines on screen):
```bash
curl -X POST http://localhost:8000/ingest/dataset \
  -H "Content-Type: application/json" \
  -d '{"split": "test", "limit": 3}'
```

### C. Batch Ingest Whole Dataset Split
Trigger the entire `test` split ingestion without limits:
```bash
curl -X POST http://localhost:8000/ingest/dataset \
  -H "Content-Type: application/json" \
  -d '{"split": "test", "limit": null}'
```
*(PowerShell variant)*:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ingest/dataset" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"split": "test", "limit": null}'
```

---

## Job State Machine

Each invoice document transitions through the following state machine in PostgreSQL:

```
queued ➔ ocr_processing ➔ ocr_done ➔ extracting ➔ extracted ➔ validating ➔ done
              │                          │               │
              ▼                          ▼               ▼
          ocr_failed            extraction_failed   needs_review
```

---

## Evaluation & Custom Validations

### 1. Target Schema Fields
The LLM extractor targets structural models defined inside `src/schemas/job.py`. Mandatory fields are:
- `header.invoice_no` (Invoice / receipt number)
- `header.invoice_date` (Issue date)
- `summary.total_net_worth` (Net amount)

### 2. Validation Logic
- **Compare Invoice Numbers**: Dynamic substring matching filters out generic vendor string prefixes (e.g., matching `0006661` and `SPEEDWAY0006661`).
- **Unstructured / Scanned Receipt Handlings**: Flat, scanned invoices that do not match hierarchical template fields default to `passed: true` with accuracy scores if the structured schema comparison is not applicable.
- **Uploaded Invoices**: Documents uploaded through the interactive page lacking ground truth defaults to `passed: true` after a successful extraction rather than scoring 0.0 / false-negative.
