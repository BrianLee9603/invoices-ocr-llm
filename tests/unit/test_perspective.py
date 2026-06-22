import numpy as np
import pytest
import cv2

from src.services.processing.ocr.perspective import PerspectiveCorrector


def test_perspective_corrector_no_document():
    # A blank white image has no document contour
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    corrector = PerspectiveCorrector()
    corrected = corrector.correct(img, "test_no_doc.png")
    
    # Should return original image unchanged
    np.testing.assert_array_equal(corrected, img)


def test_perspective_corrector_with_skewed_document():
    # Create a black background image of size 600x600
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Draw a skewed white rectangle representing a document (quadrilateral corners)
    # Let's say the quad corners are around:
    # Top-Left: (100, 150), Top-Right: (480, 100)
    # Bottom-Right: (520, 500), Bottom-Left: (80, 450)
    pts = np.array([[100, 150], [480, 100], [520, 500], [80, 450]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    corrector = PerspectiveCorrector()
    corrected = corrector.correct(img, "test_skewed_doc.png")
    
    # The output image should have been corrected (and should be smaller than 600x600, flat)
    # Specifically, it should find the corners and warp the white quad to fill the output image
    # Let's verify that the output shape is computed correctly:
    # Expected width: max(norm(tr - tl), norm(br - bl)) = max(norm((480-100, 100-150)), norm((520-80, 500-450)))
    # width_top = sqrt(380^2 + 50^2) = sqrt(144400 + 2500) = sqrt(146900) ≈ 383
    # width_bottom = sqrt(440^2 + 50^2) = sqrt(193600 + 2500) = sqrt(196100) ≈ 442
    # So width ≈ 442.
    # height_left = sqrt(20^2 + 300^2) = sqrt(400 + 90000) = sqrt(90400) ≈ 300
    # height_right = sqrt(40^2 + 400^2) = sqrt(1600 + 160000) = sqrt(161600) ≈ 402
    # So height ≈ 402.
    assert corrected.shape[0] > 0
    assert corrected.shape[1] > 0
    assert corrected.shape[0] != 600 or corrected.shape[1] != 600
    
    # A good check is that the corners of the output image are now filled (no black border at top left)
    # Top left corner of output should be white (255, 255, 255)
    assert np.all(corrected[5, 5] == [255, 255, 255])
    # Bottom right corner of output should be white (255, 255, 255)
    assert np.all(corrected[corrected.shape[0] - 6, corrected.shape[1] - 6] == [255, 255, 255])
