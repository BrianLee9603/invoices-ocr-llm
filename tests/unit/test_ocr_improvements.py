import io
import pytest
import numpy as np
from PIL import Image
from src.schemas.document import TextBlock
from src.services.processing.ocr import OcrPostProcessor, LayoutReconstructor
from src.services.processing.preprocessor import ImagePreprocessor

def test_ocr_post_processor():
    processor = OcrPostProcessor()
    
    # 1. Test encoding fix (converts \ufffd replacement char to 'ó')
    assert processor.fix_text("Descripci\ufffd") == "Descripció"
    
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
    assert processor.fix_text("100Z") == "10%"
    assert processor.fix_text("150Z") == "15%"

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
    assert img_back.size[0] >= 100
    assert img_back.size[1] >= 100
