"""
Document Image Quality Assessor.

Classifies input images into quality tiers and recommends
preprocessing strategies based on multiple image features.
Uses a multi-feature scoring system instead of brittle binary heuristics.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageSource(Enum):
    """Detected source of the document image."""
    SOFTWARE_PDF = "software_pdf"        # Clean, digitally generated
    SCANNER = "scanner"                  # High-quality scan
    CAMERA_CLEAN = "camera_clean"        # Good camera capture (modern phone, good lighting)
    CAMERA_DEGRADED = "camera_degraded"  # Poor camera capture (noise, blur, skew, shadows)


class PreprocessingStrategy(Enum):
    """Recommended preprocessing approach."""
    NONE = "none"                # Pass through unchanged (already optimal)
    MINIMAL = "minimal"          # Light sharpening only
    MODERATE = "moderate"        # CLAHE + light denoise + deskew
    AGGRESSIVE = "aggressive"    # Full pipeline: CLAHE + denoise + deskew + perspective correction


@dataclass
class QualityReport:
    """Comprehensive quality assessment of a document image."""
    source: ImageSource
    strategy: PreprocessingStrategy
    
    # Individual quality scores (0.0 = worst, 1.0 = best)
    sharpness_score: float = 0.0       # Laplacian variance normalized
    contrast_score: float = 0.0        # Dynamic range / histogram spread
    brightness_score: float = 0.0      # Mean brightness normalized
    noise_score: float = 0.0           # Estimated SNR
    skew_estimate_deg: float = 0.0     # Estimated rotation angle
    edge_density: float = 0.0          # Ratio of edge pixels
    bg_uniformity: float = 0.0         # Background consistency
    
    # Composite score
    overall_score: float = 0.0         # Weighted combination


class DocumentQualityAssessor:
    """
    Assess document image quality using multiple features.
    
    Unlike the previous binary camera/software heuristic, this uses
    a multi-dimensional feature vector to classify images into 4 tiers
    and recommend appropriate preprocessing strategies.
    """
    
    # Thresholds calibrated for invoice/receipt images
    _SHARPNESS_THRESHOLD_HIGH = 500.0    # Laplacian var above this = very sharp
    _SHARPNESS_THRESHOLD_LOW = 100.0     # Below this = blurry
    _BRIGHTNESS_IDEAL_RANGE = (120, 220) # Ideal mean brightness for documents
    _EDGE_DENSITY_SOFTWARE = 0.08        # Software-generated have higher edge density
    _BG_UNIFORMITY_THRESHOLD = 10.0      # Corner std below this = uniform bg
    _NOISE_SNR_THRESHOLD = 25.0          # SNR above this = clean image
    
    def assess(self, img: np.ndarray, filename: str = "") -> QualityReport:
        """
        Compute quality features and classify the image.
        
        Args:
            img: BGR image (OpenCV format)
            filename: For logging
            
        Returns:
            QualityReport with source classification and recommended strategy.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # --- Feature extraction ---
        sharpness = self._compute_sharpness(gray)
        contrast = self._compute_contrast(gray)
        brightness = float(np.mean(gray))
        noise_snr = self._estimate_noise_snr(gray)
        edge_density = self._compute_edge_density(gray)
        bg_uniformity_std, bg_mean = self._compute_bg_uniformity(gray)
        
        # --- Normalize to 0-1 scores ---
        sharpness_score = min(sharpness / self._SHARPNESS_THRESHOLD_HIGH, 1.0)
        contrast_score = min(contrast / 200.0, 1.0)  # 200 = ideal dynamic range
        
        lo, hi = self._BRIGHTNESS_IDEAL_RANGE
        if lo <= brightness <= hi:
            brightness_score = 1.0
        else:
            dist = min(abs(brightness - lo), abs(brightness - hi))
            brightness_score = max(0.0, 1.0 - dist / 100.0)
        
        noise_score = min(noise_snr / self._NOISE_SNR_THRESHOLD, 1.0)
        bg_uniformity_score = max(0.0, 1.0 - bg_uniformity_std / 50.0)
        
        # --- Classification ---
        source = self._classify_source(
            sharpness, edge_density, bg_uniformity_std, bg_mean,
            brightness, noise_snr, contrast
        )
        
        # --- Strategy selection ---
        strategy = self._select_strategy(
            source, sharpness_score, noise_score, brightness_score
        )
        
        # --- Composite score ---
        overall = (
            0.30 * sharpness_score +
            0.20 * contrast_score +
            0.15 * brightness_score +
            0.20 * noise_score +
            0.15 * bg_uniformity_score
        )
        
        report = QualityReport(
            source=source,
            strategy=strategy,
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            noise_score=noise_score,
            edge_density=edge_density,
            bg_uniformity=bg_uniformity_score,
            overall_score=overall,
        )
        
        logger.info(
            "[%s] Quality assessment: source=%s, strategy=%s, overall=%.2f "
            "(sharp=%.2f, contrast=%.2f, bright=%.2f, noise=%.2f, bg=%.2f)",
            filename, source.value, strategy.value, overall,
            sharpness_score, contrast_score, brightness_score,
            noise_score, bg_uniformity_score,
        )
        
        return report
    
    def _compute_sharpness(self, gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Difference between 95th and 5th percentile pixel values."""
        p5, p95 = np.percentile(gray, [5, 95])
        return float(p95 - p5)
    
    def _estimate_noise_snr(self, gray: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio using median filter."""
        denoised = cv2.medianBlur(gray, 5)
        noise = gray.astype(float) - denoised.astype(float)
        noise_std = np.std(noise)
        if noise_std < 1e-6:
            return 100.0  # Essentially no noise
        signal_std = np.std(gray.astype(float))
        return float(signal_std / noise_std)
    
    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Fraction of edge pixels using Canny."""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges) / edges.size)
    
    def _compute_bg_uniformity(self, gray: np.ndarray) -> Tuple[float, float]:
        """Compute background uniformity by sampling corner regions."""
        h, w = gray.shape
        corner_size = min(h, w) // 10
        if corner_size < 5:
            return 0.0, float(np.mean(gray))
        
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:],
        ]
        corner_std = float(np.mean([np.std(c) for c in corners]))
        corner_mean = float(np.mean([np.mean(c) for c in corners]))
        return corner_std, corner_mean
    
    def _classify_source(
        self, sharpness: float, edge_density: float,
        bg_std: float, bg_mean: float, brightness: float,
        snr: float, contrast: float
    ) -> ImageSource:
        """
        Multi-feature classification into 4 source categories.
        
        Uses a scoring system rather than hard thresholds to handle
        ambiguous cases (e.g., dark PDFs, high-quality camera shots).
        """
        software_score = 0.0
        scanner_score = 0.0
        camera_clean_score = 0.0
        camera_degraded_score = 0.0
        
        # High edge density → likely software-generated (crisp vector text)
        if edge_density > 0.08:
            software_score += 2.0
        elif edge_density > 0.05:
            software_score += 1.0
            scanner_score += 0.5
        else:
            camera_clean_score += 0.5
            camera_degraded_score += 0.5
        
        # Very uniform bright background → software or scanner
        if bg_std < 5.0 and bg_mean > 230:
            software_score += 2.0
        elif bg_std < 10.0 and bg_mean > 200:
            software_score += 1.0
            scanner_score += 1.5
        elif bg_std < 15.0:
            scanner_score += 0.5
            camera_clean_score += 1.0
        else:
            camera_degraded_score += 1.5
        
        # Very sharp → software or good scanner
        if sharpness > 500:
            software_score += 1.0
            scanner_score += 0.5
        elif sharpness > 200:
            scanner_score += 1.0
            camera_clean_score += 0.5
        elif sharpness < 100:
            camera_degraded_score += 2.0
        
        # High SNR → clean image
        if snr > 30:
            software_score += 0.5
            scanner_score += 0.5
        elif snr < 15:
            camera_degraded_score += 1.5
        
        # Low contrast → likely camera with bad lighting
        if contrast < 100:
            camera_degraded_score += 1.0
        
        # Pick highest score
        scores = {
            ImageSource.SOFTWARE_PDF: software_score,
            ImageSource.SCANNER: scanner_score,
            ImageSource.CAMERA_CLEAN: camera_clean_score,
            ImageSource.CAMERA_DEGRADED: camera_degraded_score,
        }
        return max(scores, key=scores.get)
    
    def _select_strategy(
        self, source: ImageSource,
        sharpness: float, noise: float, brightness: float
    ) -> PreprocessingStrategy:
        """Select preprocessing strategy based on source and quality scores."""
        if source == ImageSource.SOFTWARE_PDF:
            # Software PDFs are already clean
            if sharpness > 0.8 and noise > 0.8:
                return PreprocessingStrategy.NONE
            return PreprocessingStrategy.MINIMAL
        
        if source == ImageSource.SCANNER:
            # Scanners produce clean but sometimes slightly noisy images
            if sharpness > 0.7 and noise > 0.7:
                return PreprocessingStrategy.MINIMAL
            return PreprocessingStrategy.MODERATE
        
        if source == ImageSource.CAMERA_CLEAN:
            return PreprocessingStrategy.MODERATE
        
        # CAMERA_DEGRADED — needs full treatment
        return PreprocessingStrategy.AGGRESSIVE
