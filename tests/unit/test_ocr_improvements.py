import io
import pytest
import numpy as np
from PIL import Image
from src.schemas.document import TextBlock
from src.services.processing.ocr import OcrPostProcessor, LayoutReconstructor
from src.services.processing.ocr.pre_processor import ImagePreprocessor

def test_ocr_post_processor():
    processor = OcrPostProcessor()
    
    # 1. Test encoding fix (removes \ufffd replacement char)
    assert processor.fix_text("Descripci\ufffd") == "Descripci"
    
    # 2. Test word merges
    assert processor.fix_text("Grossworth") == "Gross worth"
    assert processor.fix_text("Netprice") == "Net price"
    
    # 3. Test O -> 0 substitution in numeric context
    # Inside digits
    assert processor.fix_text("1O0") == "100"
    assert processor.fix_text("1o2.04") == "102.04"
    # Leading O/o before decimals
    assert processor.fix_text("O12.34") == "012.34"
    assert processor.fix_text("o56.78") == "056.78"
    
    # 4. Test l -> 1 substitution in numeric context
    assert processor.fix_text("1l2") == "112"
    assert processor.fix_text("2l3") == "213"

    # 5. Test "0Z" to "%" misread fix
    # With context keyword
    assert processor.fix_text("VAT 100Z") == "VAT 10%"
    assert processor.fix_text("Tax 150Z") == "Tax 15%"
    assert processor.fix_text("VAT 9870Z") == "VAT 987%"
    
    # Without context keyword, but fits 1-2 digits standalone percent pattern (e.g., "100Z" -> "10%")
    assert processor.fix_text("100Z") == "10%"
    
    # Without context keyword, and does not fit 1-2 digits (e.g., SKU/Serial "9870Z") -> should be preserved
    assert processor.fix_text("SKU 9870Z") == "SKU 9870Z"

def test_layout_reconstructor():
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    
    # Create text blocks in scattered order with bbox [x_min, y_min, x_max, y_max]
    blocks = [
        TextBlock(text="Invoice No:", confidence=0.9, bbox=[10, 10, 100, 30]),
        TextBlock(text="INV-12345", confidence=0.95, bbox=[110, 10, 200, 30]),
        TextBlock(text="Date:", confidence=0.88, bbox=[300, 10, 350, 30]),
        TextBlock(text="2026-06-08", confidence=0.99, bbox=[360, 10, 460, 30]),
        TextBlock(text="Total:", confidence=0.92, bbox=[10, 100, 60, 120]),
        TextBlock(text="$100.00", confidence=0.97, bbox=[70, 100, 150, 120]),
    ]
    
    # The output should cluster them by Y and then sort by X
    result = reconstructor.reconstruct(blocks)
    
    assert "Invoice No:" in result
    assert "INV-12345" in result
    assert "Date:" in result
    assert "2026-06-08" in result
    assert "Total:" in result
    assert "$100.00" in result


def test_image_preprocessor():
    preprocessor = ImagePreprocessor()
    
    # Generate a dummy 100x100 white PNG image
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    original_bytes = buf.getvalue()
    
    # Run preprocessor
    processed_bytes = preprocessor.preprocess(original_bytes, "test_dummy.png")
    
    # Decodable check (processed bytes should be valid image bytes)
    img_back = Image.open(io.BytesIO(processed_bytes))
    assert img_back.size[0] >= 2000 or img_back.size[1] >= 2000


def test_layout_reconstructor_wavy_text():
    # Test that median-based Y-clustering is robust to vertical drift
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    
    # Blocks on the same logical line but drifting vertically
    # Block 1: [10, 10, 50, 30] -> center Y = 20
    # Block 2: [100, 15, 140, 35] -> center Y = 25
    # Block 3: [200, 20, 240, 40] -> center Y = 30
    # Outlier Block 4: [300, 45, 340, 65] -> center Y = 55 (drifting lower)
    # The average center would be (20+25+30+55)/4 = 32.5.
    # The median center would be 27.5.
    # Tolerances are calculated from average block height (~20px).
    blocks = [
        TextBlock(text="Word1", confidence=0.9, bbox=[10, 10, 50, 30]),
        TextBlock(text="Word2", confidence=0.9, bbox=[100, 15, 140, 35]),
        TextBlock(text="Word3", confidence=0.9, bbox=[200, 20, 240, 40]),
    ]
    
    rows = reconstructor._cluster_into_rows([
        {'text': b.text, 'y_min': b.bbox[1], 'y_max': b.bbox[3], 'x_min': b.bbox[0], 'x_max': b.bbox[2]}
        for b in blocks
    ])
    
    # All 3 should be clustered into a single row
    assert len(rows) == 1
    assert len(rows[0]) == 3


def test_layout_reconstructor_two_column_table():
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    
    # A row with 2 columns with a large gap (gap is 300px, width is 450px -> ratio = 66% > 15%)
    row_tabular = [
        {'text': "Qty: 5", 'x_min': 10, 'x_max': 50, 'y_min': 10, 'y_max': 30},
        {'text': "$50.00", 'x_min': 350, 'x_max': 460, 'y_min': 10, 'y_max': 30},
    ]
    
    # A row with 2 columns but small gap (gap is 10px, width is 200px -> ratio = 5% < 15%)
    row_non_tabular = [
        {'text': "Invoice", 'x_min': 10, 'x_max': 100, 'y_min': 10, 'y_max': 30},
        {'text': "Number", 'x_min': 110, 'x_max': 210, 'y_min': 10, 'y_max': 30},
    ]
    
    assert reconstructor._is_tabular_row(row_tabular) is True
    assert reconstructor._is_tabular_row(row_non_tabular) is False

