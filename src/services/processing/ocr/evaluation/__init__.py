"""
OCR Evaluation package.
Provides metrics and benchmarking tools to measure OCR performance.
"""

from src.services.processing.ocr.evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    field_exact_match,
    OcrMetrics,
)
from src.services.processing.ocr.evaluation.benchmark import (
    OcrBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    "character_error_rate",
    "word_error_rate",
    "field_exact_match",
    "OcrMetrics",
    "OcrBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
]
