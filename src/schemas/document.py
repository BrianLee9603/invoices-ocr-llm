from pydantic import BaseModel, Field
from typing import List, Optional, Any

class LineItem(BaseModel):
    item_desc: str = Field(description="Description of the item or service")
    item_qty: Optional[str] = Field(None, description="Quantity of items (raw string)")
    item_net_price: Optional[str] = Field(None, description="Net unit price (raw string)")
    item_net_worth: Optional[str] = Field(None, description="Total net worth for this item (raw string)")
    item_vat: Optional[str] = Field(None, description="VAT percentage, e.g. 10% (raw string)")
    item_gross_worth: Optional[str] = Field(None, description="Total gross worth for this item (raw string)")

class InvoiceHeader(BaseModel):
    invoice_no: str = Field(description="The unique invoice or receipt number")
    invoice_date: str = Field(description="The invoice date (raw string)")
    seller: Optional[str] = Field(None, description="Seller details (name and/or address)")
    client: Optional[str] = Field(None, description="Client details (name and/or address)")
    seller_tax_id: Optional[str] = Field(None, description="Tax ID of the seller")
    client_tax_id: Optional[str] = Field(None, description="Tax ID of the client")
    iban: Optional[str] = Field(None, description="IBAN bank identifier")

class InvoiceSummary(BaseModel):
    total_net_worth: str = Field(description="Total net amount of the invoice")
    total_vat: Optional[str] = Field(None, description="Total VAT amount")
    total_gross_worth: Optional[str] = Field(None, description="Total gross amount (total incl. VAT)")

class InvoiceExtraction(BaseModel):
    header: InvoiceHeader
    items: List[LineItem]
    summary: InvoiceSummary


class TextBlock(BaseModel):
    text: str
    confidence: float
    bbox: Optional[List[Any]] = None

class OcrOutput(BaseModel):
    file_name: str
    ocr_engine: str
    raw_text: str
    average_confidence: float
    text_blocks: List[TextBlock]
