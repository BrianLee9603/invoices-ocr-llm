import ast
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from src.schemas.document import InvoiceExtraction
from src.services.output.validator import parse_raw_amount

logger = logging.getLogger(__name__)

def normalize_invoice_no(val: str) -> str:
    """Strip non-alphanumeric characters and lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', val).lower()

def parse_date(date_str: str) -> Optional[datetime]:
    """Clean and parse a date string using standard formats."""
    cleaned = re.sub(r'\s+', ' ', date_str).strip()
    # Replace alternate separators with dash to try to parse uniformly
    cleaned_norm = re.sub(r'[./]', '-', cleaned)
    
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%y-%m-%d", "%d-%m-%y", "%m-%d-%y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned_norm, fmt)
        except ValueError:
            continue
            
    # Try with raw string formats
    raw_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", 
        "%m-%d-%Y", "%m/%d/%Y", "%Y.%m.%d", "%d.%m.%Y",
        "%y-%m-%d", "%d-%m-%y", "%m-%d-%y",
        "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"
    ]
    for fmt in raw_formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None

def compare_invoice_dates(date_str1: str, date_str2: str) -> bool:
    """Compare two dates by parsing them or falling back to simple comparison."""
    d1 = parse_date(date_str1)
    d2 = parse_date(date_str2)
    if d1 is not None and d2 is not None:
        if d1.date() == d2.date():
            return True
        # Handle day/month ambiguity swap if day and month are both <= 12
        if d1.year == d2.year:
            if d1.day <= 12 and d1.month <= 12 and d2.day <= 12 and d2.month <= 12:
                if d1.day == d2.month and d1.month == d2.day:
                    return True
    return date_str1.strip().lower() == date_str2.strip().lower()

def compare_amounts(amount_str1: str, amount_str2: str) -> bool:
    """Compare two amount strings numerically."""
    val1 = parse_raw_amount(amount_str1)
    val2 = parse_raw_amount(amount_str2)
    if val1 is not None and val2 is not None:
        return abs(val1 - val2) <= 0.05
    return amount_str1.strip().lower() == amount_str2.strip().lower()

def evaluate_extraction(
    extraction: InvoiceExtraction, 
    ground_truth: Dict[str, Any]
) -> Tuple[bool, Dict[str, float]]:
    """
    Perform content-based evaluation of extraction against ground truth.
    
    Args:
        extraction: The extracted invoice data model.
        ground_truth: The raw ground truth dictionary from DB.
        
    Returns:
        (passed, field_accuracies)
        - passed: True if all three mandatory fields match the ground truth.
        - field_accuracies: Dict containing accuracy (1.0 or 0.0) for each mandatory field.
    """
    # 1. Parse ground truth data
    gt_data = {}
    gt_json = ground_truth.get("json", "")
    if gt_json:
        if isinstance(gt_json, str):
            try:
                gt_data = json.loads(gt_json)
            except json.JSONDecodeError:
                try:
                    gt_data = ast.literal_eval(gt_json)
                except Exception as exc:
                    logger.error("Failed to parse ground truth string: %s", exc)
        elif isinstance(gt_json, dict):
            gt_data = gt_json

    if not gt_data:
        logger.warning("Empty or unparsable ground truth data")
        return False, {"invoice_no": 0.0, "invoice_date": 0.0, "total_net_worth": 0.0}

    # 2. Extract ground truth targets
    gt_header = gt_data.get("header", {})
    gt_summary = gt_data.get("summary", {})

    gt_invoice_no = str(gt_header.get("invoice_no") or "")
    gt_invoice_date = str(gt_header.get("invoice_date") or "")
    gt_total_net_worth = str(gt_summary.get("total_net_worth") or "")

    # 3. Extract extracted targets
    ext_invoice_no = str(extraction.header.invoice_no or "")
    ext_invoice_date = str(extraction.header.invoice_date or "")
    ext_total_net_worth = str(extraction.summary.total_net_worth or "")

    # 4. Perform content-based semantic matches
    no_match = (normalize_invoice_no(ext_invoice_no) == normalize_invoice_no(gt_invoice_no))
    date_match = compare_invoice_dates(ext_invoice_date, gt_invoice_date)
    net_match = compare_amounts(ext_total_net_worth, gt_total_net_worth)

    accuracies = {
        "invoice_no": 1.0 if no_match else 0.0,
        "invoice_date": 1.0 if date_match else 0.0,
        "total_net_worth": 1.0 if net_match else 0.0
    }

    # Passed rule: all three fields must match in content
    passed = no_match and date_match and net_match
    
    logger.info(
        "Evaluation results: passed=%s | accuracies=%s | Extracted: no='%s' date='%s' net='%s' | GT: no='%s' date='%s' net='%s'",
        passed, accuracies, 
        ext_invoice_no, ext_invoice_date, ext_total_net_worth,
        gt_invoice_no, gt_invoice_date, gt_total_net_worth
    )

    return passed, accuracies
