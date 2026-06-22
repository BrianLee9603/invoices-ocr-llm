from src.services.processing.ocr.evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    field_exact_match,
)


def test_character_error_rate():
    # Perfect match
    assert character_error_rate("hello", "hello") == 0.0
    
    # 1 insertion
    assert character_error_rate("hello", "hell") == 0.25
    
    # 1 deletion
    assert character_error_rate("hell", "hello") == 0.2
    
    # 1 substitution
    assert character_error_rate("hella", "hello") == 0.2
    
    # Empty ground truth
    assert character_error_rate("hello", "") == 1.0
    assert character_error_rate("", "") == 0.0


def test_word_error_rate():
    # Perfect match
    assert word_error_rate("hello world", "hello world") == 0.0
    
    # 1 mismatch
    assert word_error_rate("hello there world", "hello world") == 0.5
    
    # Empty ground truth
    assert word_error_rate("hello world", "") == 1.0
    assert word_error_rate("", "") == 0.0


def test_field_exact_match():
    pred = {"total": " $100.00 ", "date": "2026-06-08"}
    gt = {"total": "$100.00", "date": "2026-06-08"}
    
    # Normalized (should strip spaces, match)
    res_normalized = field_exact_match(pred, gt, normalize=True)
    assert res_normalized["total"] is True
    assert res_normalized["date"] is True
    
    # Raw comparison (should fail on "total" due to whitespace)
    res_raw = field_exact_match(pred, gt, normalize=False)
    assert res_raw["total"] is False
    assert res_raw["date"] is True
