import re
import logging
from typing import Optional
from src.schemas.document import InvoiceExtraction

logger = logging.getLogger(__name__)

def parse_raw_amount(amount_str: Optional[str]) -> Optional[float]:
    """
    Robust parser to convert raw formatted amount strings (e.g. '$ 24 161,60', 
    '1.234,56', 'each 444,60') into clean float values.
    """
    if not amount_str:
        return None
        
    # Remove prefix text like 'each', currency symbols, spaces, and other non-numeric chars
    cleaned = re.sub(r'[^\d,.-]', '', amount_str).strip()
    if not cleaned:
        return None

    # Handle European vs US formats:
    # 1. If both ',' and '.' exist:
    if ',' in cleaned and '.' in cleaned:
        # If dot is after comma, dot is decimal (e.g. 1,234.56)
        if cleaned.find(',') < cleaned.find('.'):
            cleaned = cleaned.replace(',', '')
        else:
            # Comma is decimal (e.g. 1.234,56)
            cleaned = cleaned.replace('.', '').replace(',', '.')
    # 2. If only ',' exists:
    elif ',' in cleaned:
        # If comma is followed by exactly 2 digits at the end of string, it is decimal (e.g. 123,45)
        if re.search(r',\d{2}$', cleaned):
            cleaned = cleaned.replace(',', '.')
        else:
            # Thousands separator
            cleaned = cleaned.replace(',', '')
            
    try:
        return float(cleaned)
    except ValueError:
        logger.warning("Failed to convert amount string '%s' (cleaned: '%s') to float", amount_str, cleaned)
        return None

def validate_extraction(extraction: InvoiceExtraction) -> bool:
    """
    Validate mandatory fields are present and optionally cross-check total amounts.
    
    Returns:
        True if the mandatory fields are present.
    """
    header = extraction.header
    summary = extraction.summary
    
    # 1. Validate mandatory fields exist and are not empty
    if not (header.invoice_no and header.invoice_no.strip()):
        logger.warning("Validation failed: invoice_no is missing or empty")
        return False
        
    if not (header.invoice_date and header.invoice_date.strip()):
        logger.warning("Validation failed: invoice_date is missing or empty")
        return False
        
    if not (summary.total_net_worth and summary.total_net_worth.strip()):
        logger.warning("Validation failed: total_net_worth is missing or empty")
        return False

    # 2. Optional VAT calculation cross-check
    net_val = parse_raw_amount(summary.total_net_worth)
    vat_val = parse_raw_amount(summary.total_vat)
    gross_val = parse_raw_amount(summary.total_gross_worth)
    
    if net_val is not None and vat_val is not None and gross_val is not None:
        expected_gross = net_val + vat_val
        diff = abs(expected_gross - gross_val)
        if diff > 0.05:
            logger.warning(
                "VAT validation warning: net (%f) + vat (%f) = expected gross (%f) "
                "differs from extracted gross (%f) by %f",
                net_val, vat_val, expected_gross, gross_val, diff
            )
            # We don't fail the extraction for VAT discrepancies since VAT is optional
            
    # 3. Line items net worth sum cross-check
    if extraction.items:
        items_net_sum = 0.0
        has_item_net_worth = False
        for item in extraction.items:
            item_net = parse_raw_amount(item.item_net_worth)
            if item_net is not None:
                items_net_sum += item_net
                has_item_net_worth = True
                
        if has_item_net_worth and net_val is not None:
            diff_items = abs(items_net_sum - net_val)
            if diff_items > 0.05:
                logger.warning(
                    "Net worth validation warning: sum of item net worths (%f) "
                    "differs from summary total_net_worth (%f) by %f",
                    items_net_sum, net_val, diff_items
                )
                
    return True
