# Intelligent Invoice & Receipt Extraction System (OCR + LLM Pipeline)

An end-to-end Document AI pipeline designed to ingest invoice and receipt images, perform layout-aware OCR, extract structured JSON data using a local LLM, and evaluate extraction quality against Hugging Face ground truths.

---

## 1. Pipeline Overview & Functional Design

Our architecture implements the full functional specifications outlined in the technical assessment:

```
                            [ RAW IMAGE ]
                                  │
                                  ▼
                   [ Document Quality Assessor ]
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      ▼ (NONE)                    ▼ (MINIMAL)                 ▼ (MODERATE/AGGRESSIVE)
[Grayscale]                 [Light Sharpen]             [CLAHE + Gaussian Blur]
      │                           │                           │
      └───────────────────────────┼───────────────────────────┘
                                  ▼
                     [ Perspective Corrector ] (Quadrilateral Warp)
                                  │
                                  ▼
                     [ Hough Line De-skewing ]
                                  │
                     ┌────────────┴────────────┐
                     ▼ (Parallel Execution)    ▼
               [ PaddleOCR ]            [ DocLayout-YOLO ]
             (Text Detection)           (Table & Layout Regions)
                     └────────────┬────────────┘
                                  ▼
                      [ Layout Reconstructor ]
               (Spatial Clustering & Table Markdown Formatting)
                                  │
                                  ▼
                     [ OCR Post-Processor ] (Regex, Zero/Digit Corrections)
                                  │
                                  ▼
                  [ Ollama Extractor (Qwen) ] ◄─── (Pydantic Schema Guided)
                                  │
                                  ▼
                   [ Output & Evaluation Service ]
            (Pass-Rate Criteria & Business Math Validations)
```

### Part A: OCR Pipeline
* **Engine**: Self-hosted **PaddleOCR (PP-OCRv5)** running in a CPU/GPU-monitored ThreadPoolExecutor to prevent event-loop blocking.
* **Layout Preserving**: Tracks text block positions, `bbox` coordinates, and token confidence levels.
* **Spatial Clustering**: Groups scattered blocks into reading-order rows using vertical coordinate tolerances and horizontal distance thresholds.

### Part B: LLM Structured Extraction
* **Core Extractor**: Pydantic schema-guided JSON parsing using a local **Ollama** instance running `qwen2.5:3b` (primary free tier) or **Google Gemini API** (fallback/optional).
* **Schema Validation**: Ensures correct types and handles nested line items. Mandatory fields (`header.invoice_no`, `header.invoice_date`, and `summary.total_net_worth`) are strictly validated; missing optional fields default to `null`.

### Part C: Pipeline Improvement Techniques
1. **Document Quality Assessor**: Analyzes input images using edge density, sharpness metrics, and background uniformity to dynamically select the optimal preprocessing strategy (`NONE`, `MINIMAL`, `MODERATE`, `AGGRESSIVE`).
2. **Perspective Corrector**: Automatically detects document contours, extracts the bounding quadrilateral, and applies a perspective warp to normalize skewed scans.
3. **De-skewing**: Leverages Canny Edge Detection and Hough Line Transform to calculate document rotation and deskew text lines.
4. **Layout-Aware Table Extraction**: Integrates **DocLayout-YOLO (YOLOv10)** to identify tabular layout coordinates. When text rows overlap with detected tables, the layout reconstructor automatically formats them as Markdown tables, preserving line item structures.
5. **OCR Post-Processing**: Corrects common OCR misreads (e.g., matching zero `0` vs `O` inside numeric blocks, fixing `VAT 100Z` to `VAT 10%`, and repairing encoding artifacts).

### Part D: Evaluation Pipeline
* **Pass-Rate Rule**: A job is marked as `passed` only if all **3 mandatory fields** (invoice number, date, total net worth) are extracted correctly.
* **Evaluation Metrics**: Reports exact string match accuracy for `invoice_no` (with substring vendor-prefix filtering), date format matches, and Levenshtein distance metrics.

### Part E: Streamlit Product Demo & Monitor
* **Interactive Upload Sandbox**: Drag-and-drop user interface displaying side-by-side original images, deskewed preprocessed images, OCR raw text, and the final predicted JSON.
* **Analytics Dashboard**: Real-time evaluation analytics displaying overall pass rates, field-level accuracy, and latency profiles.
* **System Monitor**: Displays background worker queue states, active jobs, consumer group health, and logs.

---

## 2. Directory Structure (Domain-Driven)

The OCR processing package has been reorganized into domain-driven subdirectories for scalability and readability:

```
src/services/processing/ocr/
├── engines/
│   └── engines.py           # PaddleOcrEngine implementation & thread pooling
├── layout/
│   ├── layout.py            # Spatial row clustering & Markdown formatter
│   └── doclayout.py         # DocLayout-YOLO (YOLOv10) region analyzer
├── preprocessing/
│   ├── pre_processor.py     # Quality-aware adaptive pipeline
│   ├── perspective.py       # Document contour detection & quadrilateral warp
│   └── quality_assessor.py  # Image sharpness and background analysis
├── postprocessing/
│   └── post_processor.py    # Spell checks, digit substitution, VAT fixes
├── evaluation/
│   ├── metrics.py           # Character Error Rate (CER), Levenshtein & exact match
│   └── benchmark.py         # A/B testing pipeline comparing preprocess configs
└── __init__.py              # Unified package API exports
```

---

## 3. Technology Stack

* **Language**: Python 3.12+
* **OCR & Layout**: PaddleOCR 2.8+, PyMuPDF, `doclayout_yolo` (YOLOv10), `ultralytics`
* **Local LLM**: Ollama (`qwen2.5:3b`) / Google Gemini SDK
* **API Gateway**: FastAPI & Uvicorn (ASGI)
* **Message Broker**: Redis Streams & Consumer Groups
* **Database**: PostgreSQL (SQLAlchemy 2.0 Async)
* **Object Store**: MinIO (S3-compatible API)
* **User Interface**: Streamlit & Plotly

---

## 4. Quick Start & Setup

### 1. Run Local Infrastructure
Spin up the PostgreSQL, Redis, MinIO, and Ollama services:
```bash
docker compose -f infra/docker-compose.yaml up -d
```

### 2. Configure Environment
Create a `.env` file from the example template:
```bash
cp .env.example .env
```
Ensure `LLM_PROVIDER="ollama"` and `OLLAMA_HOST="http://localhost:11434"` (or set `GEMINI_API_KEY` to run Gemini).

### 3. Setup Virtual Environment
Install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate    # On Linux/macOS

pip install -r requirements.txt
```

### 4. Running the Pipeline Services
Launch each pipeline component in a separate terminal:

* **Web API Gateway**:
  ```bash
  .venv\Scripts\python.exe -m src.main
  ```
* **Processing Worker (OCR + LLM)**:
  ```bash
  .venv\Scripts\python.exe -m src.cli.processing_worker
  ```
* **Output Worker (Evaluation & Validation)**:
  ```bash
  .venv\Scripts\python.exe -m src.cli.output_worker
  ```
* **Streamlit Dashboard**:
  ```bash
  .venv\Scripts\streamlit.exe run demo/app.py
  ```

---

## 5. Testing & Verification

### Running Automated Tests
We maintain 100% test passing coverage across 66 unit and integration test suites:
```bash
.venv\Scripts\python.exe -m pytest -v
```
To run specific suites:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_doclayout.py
.venv\Scripts\python.exe -m pytest tests/integration/test_ingestion.py
```
