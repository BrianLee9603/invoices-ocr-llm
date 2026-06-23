import pytest
import numpy as np
from src.schemas.document import TextBlock
from src.services.processing.ocr.layout.doclayout import DocLayoutYoloAnalyzer
from src.services.processing.ocr.layout.layout import LayoutReconstructor


def test_doclayout_yolo_analyzer_dummy():
    # Test initialization
    analyzer = DocLayoutYoloAnalyzer()
    assert analyzer.model_path is not None
    # We don't necessarily need to trigger model download during lightweight unit test,
    # but since weights are already cached in our previous step, this will load quickly.
    
    # Create a dummy image (3 channels, uint8)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Run prediction (should return list, empty or not)
    regions = analyzer.detect_regions(dummy_image)
    assert isinstance(regions, list)


def test_layout_reconstructor_with_yolo_table():
    reconstructor = LayoutReconstructor(row_tolerance_ratio=0.5)
    
    # Text blocks that might not normally trigger the tabular heuristic
    # because they have only 2 columns with a small horizontal gap,
    # but they are located vertically inside a YOLO-detected table.
    blocks = [
        TextBlock(text="Item Description", confidence=0.9, bbox=[10, 100, 150, 120]),
        TextBlock(text="Price", confidence=0.95, bbox=[160, 100, 200, 120]),
        TextBlock(text="Item A", confidence=0.88, bbox=[10, 130, 150, 150]),
        TextBlock(text="$10.00", confidence=0.99, bbox=[160, 130, 200, 150]),
    ]
    
    # YOLO table bounding box covering the y-coordinates from 90 to 160
    layout_regions = [
        {
            "bbox": [5, 90, 210, 160],
            "label": "table",
            "confidence": 0.85
        }
    ]
    
    # Without layout regions, _is_tabular_row would check X gap between columns.
    # Gap is 10px, row width is 190px -> gap ratio = 5.2% < 15%.
    # Therefore, without layout_regions, they wouldn't be classified as a table.
    result_without = reconstructor.reconstruct(blocks, layout_regions=None)
    assert "|" not in result_without  # Not formatted as a markdown table
    
    # With layout regions, the rows should be recognized as being inside a table
    result_with = reconstructor.reconstruct(blocks, layout_regions=layout_regions)
    assert "|" in result_with  # Formatted as a markdown table!
