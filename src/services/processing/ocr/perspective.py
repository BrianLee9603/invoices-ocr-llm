"""
Perspective correction for camera-captured document images.

Two-tier approach:
1. Fast: OpenCV contour-based (find document boundary → homography transform)
2. Robust fallback: PaddleOCR UVDoc model-based unwarping (if integrated/available)
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """
    Correct perspective distortion in document images.
    
    Uses contour detection to find the document boundary,
    then applies a perspective transform to "flatten" the document.
    """
    
    _MIN_CONTOUR_AREA_RATIO = 0.15  # Document must be ≥15% of image area
    _APPROX_EPSILON_RATIO = 0.02    # cv2.approxPolyDP epsilon
    
    def correct(self, img: np.ndarray, filename: str = "") -> np.ndarray:
        """
        Attempt perspective correction.
        
        Returns corrected image if successful, original image otherwise.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Step 1: Find document contour
        corners = self._find_document_corners(gray, filename)
        if corners is None:
            logger.debug("[%s] No document boundary found, skipping perspective correction", filename)
            return img
        
        # Step 2: Compute target rectangle dimensions
        width, height = self._compute_output_dimensions(corners)
        
        # Step 3: Apply perspective transform
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        corrected = cv2.warpPerspective(
            img, M, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(img.shape) == 3 else 255,
        )
        
        logger.info("[%s] Perspective correction applied (output: %dx%d)", filename, width, height)
        return corrected
    
    def _find_document_corners(
        self, gray: np.ndarray, filename: str
    ) -> Optional[np.ndarray]:
        """
        Find the 4 corners of the document using contour detection.
        
        Returns ordered corners: [top-left, top-right, bottom-right, bottom-left]
        or None if no valid document boundary found.
        """
        h, w = gray.shape
        image_area = h * w
        
        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest contours
            area = cv2.contourArea(contour)
            
            # Must be at least 15% of image area
            if area < image_area * self._MIN_CONTOUR_AREA_RATIO:
                continue
            
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self._APPROX_EPSILON_RATIO * peri, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                ordered = self._order_corners(corners)
                
                logger.debug("[%s] Found document boundary (area ratio: %.2f)",
                            filename, area / image_area)
                return ordered
        
        return None
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # We need to sort points based on coordinates.
        # top-left: smallest x+y
        # bottom-right: largest x+y
        # top-right: smallest y-x (or smallest x-y, let's use sum & diff carefully)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]    # Top-left: smallest sum
        rect[2] = pts[np.argmax(s)]    # Bottom-right: largest sum
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right: smallest diff (x - y is largest or y - x is smallest, diff=y-x so negative value. Argmin gets smallest y-x which is top-right)
        rect[3] = pts[np.argmax(diff)]  # Bottom-left: largest diff (y-x is largest)
        
        return rect
    
    def _compute_output_dimensions(self, corners: np.ndarray) -> Tuple[int, int]:
        """Compute output width and height from corner coordinates."""
        tl, tr, br, bl = corners
        
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        height = int(max(height_left, height_right))
        
        return width, height
