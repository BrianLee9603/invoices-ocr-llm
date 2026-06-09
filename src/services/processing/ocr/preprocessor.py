"""
Image preprocessing pipeline for invoice/receipt images.

Applies adaptive enhancements to improve OCR accuracy, especially on
camera-captured receipts with uneven lighting, noise, or slight skew.
Software-produced invoices get minimal processing to avoid degradation.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocess invoice images before OCR to improve text recognition quality.

    Pipeline:
      1. Decode raw bytes → OpenCV image
      2. Detect image type (software-produced vs camera-captured)
      3. Apply type-specific preprocessing
      4. Encode back to bytes
    """

    # --- Thresholds for detection ---
    # Camera-captured images typically have lower edge density and more noise
    _EDGE_DENSITY_THRESHOLD = 0.06  # Software invoices have sharp, dense edges
    _NOISE_THRESHOLD = 5.0  # Laplacian variance — higher = more detail/noise

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

        # Detect image type
        is_camera = self._is_camera_captured(img, filename)
        logger.info(
            "[%s] Image type detected: %s",
            filename,
            "camera-captured" if is_camera else "software-produced",
        )

        if is_camera:
            img = self._preprocess_camera(img, filename)
        else:
            img = self._preprocess_software(img, filename)

        # Upscale small images for better OCR (target ~2000px on longest side)
        img = self._ensure_minimum_resolution(img, filename, target_long_side=2000)

        # Encode back to PNG (lossless)
        success, encoded = cv2.imencode(".png", img)
        if not success:
            logger.warning("[%s] Failed to encode preprocessed image, returning original", filename)
            return image_bytes

        logger.debug("[%s] Preprocessing complete. Output size: %.1f KB", filename, len(encoded) / 1024)
        return encoded.tobytes()

    def _is_camera_captured(self, img: np.ndarray, filename: str) -> bool:
        """
        Heuristic to distinguish camera-captured from software-produced images.

        Software-produced invoices have:
        - Clean white backgrounds (high mean brightness)
        - Sharp, consistent edges (high edge density)
        - Low color variance in background areas

        Camera-captured receipts have:
        - Uneven backgrounds (shadows, table surfaces)
        - Perspective distortion
        - Lower contrast text
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Check edge density — software images have clean, sharp edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # 2. Check background uniformity — software images have uniform white bg
        # Sample corners (likely background)
        h, w = gray.shape
        corner_size = min(h, w) // 10
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:],
        ]
        corner_std = np.mean([np.std(c) for c in corners])
        corner_mean = np.mean([np.mean(c) for c in corners])

        logger.debug(
            "[%s] Detection metrics: edge_density=%.4f, corner_std=%.1f, corner_mean=%.1f",
            filename, edge_density, corner_std, corner_mean,
        )

        # Software-produced: high edge density, bright uniform corners
        # Camera-captured: lower edge density, variable corners, darker backgrounds
        if corner_std > 15 or corner_mean < 200:
            return True  # Camera — non-uniform or dark background
        if edge_density < self._EDGE_DENSITY_THRESHOLD:
            return True  # Camera — fewer clean edges

        return False

    def _preprocess_camera(self, img: np.ndarray, filename: str) -> np.ndarray:
        """
        Aggressive preprocessing for camera-captured receipts/invoices.

        Steps:
        1. Convert to grayscale
        2. Apply CLAHE for contrast enhancement
        3. Adaptive thresholding to handle uneven lighting
        4. Morphological noise removal
        5. De-skew if needed
        """
        logger.debug("[%s] Applying camera-captured preprocessing pipeline", filename)

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        logger.debug("[%s] CLAHE applied", filename)

        # 3. Slight Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 4. Adaptive thresholding — handles uneven lighting well
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=11,
        )
        logger.debug("[%s] Adaptive thresholding applied", filename)

        # 5. Morphological cleanup: remove small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        logger.debug("[%s] Morphological cleanup done", filename)

        # 6. De-skew
        skew_angle = self._detect_skew(cleaned, filename)
        if abs(skew_angle) > 0.5:
            cleaned = self._rotate_image(cleaned, -skew_angle, filename)

        # Convert back to BGR for consistent output
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return result

    def _preprocess_software(self, img: np.ndarray, filename: str) -> np.ndarray:
        """
        Light preprocessing for software-produced invoices.

        These are already clean — we just:
        1. Sharpen slightly to counteract any JPEG compression artifacts
        2. Ensure good contrast
        """
        logger.debug("[%s] Applying software-produced preprocessing (light)", filename)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Light sharpening to counteract compression artifacts
        sharpening_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Otsu thresholding — works well on clean images with bimodal histograms
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return result

    def _detect_skew(self, binary_img: np.ndarray, filename: str) -> float:
        """
        Detect skew angle using Hough Line Transform.

        Returns angle in degrees. Positive = clockwise rotation.
        """
        # Invert for line detection (text = white on black background)
        inverted = cv2.bitwise_not(binary_img)

        # Detect lines
        lines = cv2.HoughLinesP(
            inverted, 1, np.pi / 180,
            threshold=100,
            minLineLength=binary_img.shape[1] // 4,
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

    def _ensure_minimum_resolution(
        self, img: np.ndarray, filename: str, target_long_side: int = 2000
    ) -> np.ndarray:
        """Upscale small images so OCR models have enough detail to work with."""
        h, w = img.shape[:2]
        long_side = max(h, w)

        if long_side >= target_long_side:
            return img

        scale = target_long_side / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("[%s] Upscaled from %dx%d → %dx%d (scale=%.2f)", filename, w, h, new_w, new_h, scale)
        return upscaled
