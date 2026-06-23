import io
import numpy as np
import pytest
import cv2
from PIL import Image

from src.services.processing.ocr.preprocessing.quality_assessor import (
    DocumentQualityAssessor,
    ImageSource,
    PreprocessingStrategy,
)
from src.services.processing.ocr.preprocessing.pre_processor import ImagePreprocessor


def test_quality_assessor_clean_image():
    # Create a 200x200 pure white image (classic software-produced template background)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Draw some text or lines to create edges (representing high edge density of PDF text)
    # Using cv2.putText to write some text
    for i in range(5):
        cv2.putText(
            img,
            "INVOICE ITEM DESCRIPTION",
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
    
    assessor = DocumentQualityAssessor()
    report = assessor.assess(img, "test_clean.png")
    
    # A clean white image with computer text should be classified as SOFTWARE_PDF
    assert report.source == ImageSource.SOFTWARE_PDF
    assert report.strategy in (PreprocessingStrategy.NONE, PreprocessingStrategy.MINIMAL)
    assert report.sharpness_score > 0.5
    assert report.bg_uniformity > 0.8


def test_quality_assessor_noisy_image():
    # Create a noisy dark gray image (representing degraded camera shot)
    np.random.seed(42)
    img = np.random.randint(100, 150, size=(200, 200, 3), dtype=np.uint8)
    
    assessor = DocumentQualityAssessor()
    report = assessor.assess(img, "test_noisy.png")
    
    # A noisy, dark, low contrast image should trigger aggressive preprocessing
    assert report.source == ImageSource.CAMERA_DEGRADED
    assert report.strategy == PreprocessingStrategy.AGGRESSIVE


def test_preprocessor_with_routing():
    preprocessor = ImagePreprocessor()
    
    # Generate a dummy 200x200 white PNG image
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    original_bytes = buf.getvalue()
    
    # Run preprocessor
    processed_bytes = preprocessor.preprocess(original_bytes, "test_routing.png")
    
    # The preprocessor output should decode to a valid image, upscaled to at least target min resolution
    img_back = Image.open(io.BytesIO(processed_bytes))
    assert img_back.size[0] >= 2000 or img_back.size[1] >= 2000
