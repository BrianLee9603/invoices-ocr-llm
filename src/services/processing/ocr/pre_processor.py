"""
Image preprocessing pipeline for invoice/receipt images.

Applies adaptive quality-aware enhancements to improve OCR accuracy,
especially on camera-captured receipts with uneven lighting, noise,
or perspective distortions.
"""

import logging
import cv2
import numpy as np

from src.services.processing.ocr.quality_assessor import (
    DocumentQualityAssessor,
    PreprocessingStrategy,
    QualityReport,
)
from src.services.processing.ocr.perspective import PerspectiveCorrector

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Adaptive image preprocessor using quality-aware routing.

    Instead of a binary camera/software split, uses DocumentQualityAssessor
    to classify images and apply the minimum necessary preprocessing strategy.
    """

    def __init__(self):
        self._assessor = DocumentQualityAssessor()
        self._perspective_corrector = PerspectiveCorrector()
        # Resolution limits
        self._min_long_side = 2000
        self._max_long_side = 4000

    def preprocess(self, image_bytes: bytes, filename: str = "") -> bytes:
        """
        Apply preprocessing pipeline and return enhanced image bytes.

        Args:
            image_bytes: Raw image file bytes.
            filename: For logging purposes.

        Returns:
            Preprocessed image as PNG bytes.
        """
        # Decode
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("[%s] Could not decode image, returning original", filename)
            return image_bytes

        h, w = img.shape[:2]
        logger.debug("[%s] Image decoded: %dx%d", filename, w, h)

        # Quality assessment
        report = self._assessor.assess(img, filename)

        # Apply strategy-appropriate preprocessing
        if report.strategy == PreprocessingStrategy.NONE:
            processed = self._to_grayscale_bgr(img)
        elif report.strategy == PreprocessingStrategy.MINIMAL:
            processed = self._preprocess_minimal(img, filename)
        elif report.strategy == PreprocessingStrategy.MODERATE:
            processed = self._preprocess_moderate(img, filename)
        else:  # AGGRESSIVE
            processed = self._preprocess_aggressive(img, filename)

        # Resolution normalization (both min AND max)
        processed = self._normalize_resolution(processed, filename)

        # Encode back to PNG (lossless)
        success, encoded = cv2.imencode(".png", processed)
        if not success:
            logger.warning("[%s] Failed to encode preprocessed image, returning original", filename)
            return image_bytes

        logger.debug("[%s] Preprocessing complete. Output size: %.1f KB", filename, len(encoded) / 1024)
        return encoded.tobytes()

    def _to_grayscale_bgr(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale, then back to BGR to maintain channel consistency."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _preprocess_minimal(self, img: np.ndarray, filename: str) -> np.ndarray:
        """Light preprocessing (grayscale + light sharpening) for clean images."""
        logger.debug("[%s] Applying minimal preprocessing (light sharpening)", filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Light sharpening to counteract compression artifacts
        sharpening_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _preprocess_moderate(self, img: np.ndarray, filename: str) -> np.ndarray:
        """Moderate preprocessing (CLAHE + blur + deskew) for standard camera/scans."""
        logger.debug("[%s] Applying moderate preprocessing pipeline", filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # De-skew
        skew_angle = self._detect_skew(blurred, filename)
        if abs(skew_angle) > 0.5:
            blurred = self._rotate_image(blurred, -skew_angle, filename)

        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    def _preprocess_aggressive(self, img: np.ndarray, filename: str) -> np.ndarray:
        """Aggressive preprocessing (CLAHE + denoise + deskew + perspective correction)."""
        logger.debug("[%s] Applying aggressive preprocessing pipeline", filename)
        
        # Start with moderate preprocessing
        processed = self._preprocess_moderate(img, filename)
        
        # Apply perspective correction
        processed = self._perspective_corrector.correct(processed, filename)
        
        return processed

    def _detect_skew(self, gray_img: np.ndarray, filename: str) -> float:
        """
        Detect skew angle using Hough Line Transform on grayscale image.

        Returns angle in degrees. Positive = clockwise rotation.
        """
        # Create edge image for line detection
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=gray_img.shape[1] // 4,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            logger.debug("[%s] No lines detected for skew correction", filename)
            return 0.0

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within ±30° of horizontal)
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median angle to be robust against outliers
        skew_angle = float(np.median(angles))
        logger.debug("[%s] Detected skew angle: %.2f°", filename, skew_angle)
        return skew_angle

    def _rotate_image(self, img: np.ndarray, angle: float, filename: str) -> np.ndarray:
        """Rotate image by the given angle (degrees) around its center."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Use white background fill for rotated areas
        bg_color = 255 if len(img.shape) == 2 else (255, 255, 255)
        rotated = cv2.warpAffine(
            img, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color,
        )
        logger.debug("[%s] Image rotated by %.2f°", filename, angle)
        return rotated

    def _normalize_resolution(self, img: np.ndarray, filename: str) -> np.ndarray:
        """
        Normalize image resolution.
        Upscales small images (long side < 2000) and downscales huge images (long side > 4000).
        """
        h, w = img.shape[:2]
        long_side = max(h, w)

        if long_side < self._min_long_side:
            scale = self._min_long_side / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug("[%s] Upscaled from %dx%d → %dx%d (scale=%.2f)", filename, w, h, new_w, new_h, scale)
        elif long_side > self._max_long_side:
            scale = self._max_long_side / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug("[%s] Downscaled from %dx%d → %dx%d (scale=%.2f)", filename, w, h, new_w, new_h, scale)

        return img
