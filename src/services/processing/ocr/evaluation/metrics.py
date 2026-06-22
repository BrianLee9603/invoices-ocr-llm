"""
OCR evaluation metrics for invoice extraction.

Three levels of evaluation:
1. Character-level: CER (Character Error Rate)
2. Word-level: WER (Word Error Rate) 
3. Field-level: Exact match rate per invoice field
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class OcrMetrics:
    """Comprehensive OCR quality metrics."""
    cer: float              # Character Error Rate (0 = perfect)
    wer: float              # Word Error Rate (0 = perfect)
    field_accuracy: Dict[str, float]  # Per-field exact match rate
    consistency_score: float  # Business logic consistency (subtotal + tax = total)
    num_samples: int


def character_error_rate(prediction: str, ground_truth: str) -> float:
    """
    Compute CER using Levenshtein distance.
    
    CER = (insertions + deletions + substitutions) / len(ground_truth)
    """
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    
    n, m = len(ground_truth), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ground_truth[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    
    return dp[n][m] / n


def word_error_rate(prediction: str, ground_truth: str) -> float:
    """Compute WER using word-level Levenshtein distance."""
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    
    n, m = len(gt_words), len(pred_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    
    return dp[n][m] / n


def field_exact_match(
    predicted_fields: Dict[str, str],
    ground_truth_fields: Dict[str, str],
    normalize: bool = True
) -> Dict[str, bool]:
    """
    Check exact match for each field.
    
    If normalize=True, strips whitespace and lowercases before comparison.
    """
    results = {}
    for field_name, gt_value in ground_truth_fields.items():
        pred_value = predicted_fields.get(field_name, "")
        
        if normalize:
            gt_clean = re.sub(r'\s+', ' ', gt_value.strip().lower())
            pred_clean = re.sub(r'\s+', ' ', pred_value.strip().lower())
            results[field_name] = gt_clean == pred_clean
        else:
            results[field_name] = gt_value == pred_value
    
    return results
