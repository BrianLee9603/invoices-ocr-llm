import logging
import re

logger = logging.getLogger(__name__)


class OcrPostProcessor:
    """Fix common OCR character substitution errors.
    
    PaddleOCR sometimes confuses visually similar characters:
      - O ↔ 0 in numeric contexts
      - l ↔ 1 in numeric contexts  
      - Merged words (e.g. 'Grossworth' → 'Gross worth')
      - Encoding artifacts (e.g. replacement characters)
    """

    # Patterns where O/o should likely be 0 (digits surrounding it)
    _O_TO_ZERO = re.compile(r'(?<=\d)[Oo](?=\d)')
    # Patterns where l should likely be 1 (digits surrounding it)
    _L_TO_ONE = re.compile(r'(?<=\d)l(?=\d)')
    # Fix standalone 'O' that should be '0' at start of numeric sequences
    _LEADING_O = re.compile(r'\b[Oo](\d+[,.]\d+)\b')
    # Fix common OCR split: "10" + "0Z" → should be "10%" (the % gets misread)
    _PERCENT_FIX = re.compile(r'\b(\d+)0Z\b')
    # Normalize excessive whitespace in numbers
    _MULTI_SPACE_IN_NUM = re.compile(r'(\d)\s{2,}(\d)')
    # Known word merges to fix
    _WORD_MERGES = {
        'Grossworth': 'Gross worth',
        'grossworth': 'gross worth',
        'Networth': 'Net worth',
        'networth': 'net worth',
        'Netprice': 'Net price',
        'netprice': 'net price',
    }

    def fix_text(self, text: str, confidence: float = 1.0) -> str:
        """Apply heuristic corrections to OCR text.
        
        More aggressive corrections are applied to low-confidence blocks.
        """
        if not text:
            return text

        original = text

        # Fix replacement characters (common with accented chars)
        text = text.replace('\ufffd', 'ó')  # Common: Descripció

        # Fix known word merges
        for merged, fixed in self._WORD_MERGES.items():
            text = text.replace(merged, fixed)

        # Fix O→0 in numeric context
        text = self._O_TO_ZERO.sub('0', text)
        text = self._LEADING_O.sub(r'0\1', text)

        # Fix l→1 in numeric context
        text = self._L_TO_ONE.sub('1', text)

        # Fix the common PaddleOCR "0Z" instead of "%" pattern
        text = self._PERCENT_FIX.sub(r'\g<1>%', text)

        # Normalize multiple spaces in numbers
        text = self._MULTI_SPACE_IN_NUM.sub(r'\1 \2', text)

        if text != original:
            logger.debug("OCR post-fix: '%s' → '%s'", original, text)

        return text
