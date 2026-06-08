import abc
import asyncio
import logging
import os
import re
import tempfile
import time
import psutil
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.schemas.document import OcrOutput, TextBlock

logger = logging.getLogger(__name__)


# =============================================================================
# OCR Post-Processing: Fix common character recognition errors
# =============================================================================

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


# =============================================================================
# Layout Reconstruction: Rebuild document structure from bbox coordinates
# =============================================================================

class LayoutReconstructor:
    """Reconstruct document layout from OCR text blocks using spatial clustering.

    Takes scattered OCR blocks with bbox coordinates and reconstructs the
    original reading order, grouping text into rows and detecting table regions.
    This is critical for invoice extraction because PaddleOCR returns blocks
    in arbitrary order, destroying table structure.

    The output is a formatted string with section markers (HEADER / TABLE / SUMMARY)
    and markdown-style table formatting for tabular regions.
    """

    def __init__(self, row_tolerance_ratio: float = 0.5):
        """
        Args:
            row_tolerance_ratio: How much vertical overlap (as a fraction of
                average block height) is needed to consider two blocks on the
                same row. Higher = more lenient grouping.
        """
        self._row_tolerance_ratio = row_tolerance_ratio

    def reconstruct(self, text_blocks: List[TextBlock], post_processor: Optional[OcrPostProcessor] = None) -> str:
        """Cluster blocks into rows, detect tables, return layout-formatted text.

        Args:
            text_blocks: OCR text blocks with bbox coordinates.
            post_processor: Optional post-processor to fix OCR errors.

        Returns:
            Layout-reconstructed text with section markers and table formatting.
        """
        if not text_blocks:
            return ""

        # Filter blocks with valid bboxes
        valid_blocks = []
        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue
            if post_processor:
                text = post_processor.fix_text(text, block.confidence)
            if block.bbox and len(block.bbox) >= 4:
                valid_blocks.append({
                    'text': text,
                    'confidence': block.confidence,
                    'x_min': block.bbox[0],
                    'y_min': block.bbox[1],
                    'x_max': block.bbox[2],
                    'y_max': block.bbox[3],
                })
            else:
                valid_blocks.append({
                    'text': text,
                    'confidence': block.confidence,
                    'x_min': 0,
                    'y_min': 0,
                    'x_max': 0,
                    'y_max': 0,
                })

        if not valid_blocks:
            return ""

        # Step 1: Cluster blocks into rows by Y coordinate
        rows = self._cluster_into_rows(valid_blocks)

        # Step 2: Detect table vs non-table regions
        sections = self._detect_sections(rows)

        # Step 3: Format output
        return self._format_sections(sections)

    def _cluster_into_rows(self, blocks: list) -> list:
        """Group text blocks into rows based on vertical position overlap."""
        # Sort by y_min (top of block)
        sorted_blocks = sorted(blocks, key=lambda b: b['y_min'])

        # Calculate average block height for tolerance
        heights = [b['y_max'] - b['y_min'] for b in sorted_blocks if b['y_max'] > b['y_min']]
        avg_height = sum(heights) / len(heights) if heights else 20
        tolerance = avg_height * self._row_tolerance_ratio

        rows = []
        current_row = [sorted_blocks[0]]
        current_y_center = (sorted_blocks[0]['y_min'] + sorted_blocks[0]['y_max']) / 2

        for block in sorted_blocks[1:]:
            block_y_center = (block['y_min'] + block['y_max']) / 2

            if abs(block_y_center - current_y_center) <= tolerance:
                # Same row
                current_row.append(block)
                # Update row center as running average
                current_y_center = sum(
                    (b['y_min'] + b['y_max']) / 2 for b in current_row
                ) / len(current_row)
            else:
                # New row — save current and start new
                current_row.sort(key=lambda b: b['x_min'])  # Left-to-right within row
                rows.append(current_row)
                current_row = [block]
                current_y_center = block_y_center

        # Don't forget the last row
        current_row.sort(key=lambda b: b['x_min'])
        rows.append(current_row)

        return rows

    def _detect_sections(self, rows: list) -> list:
        """Detect HEADER, TABLE, and SUMMARY sections based on row structure."""
        sections = []
        
        # Analyze column count distribution to find table regions
        # A table region has multiple consecutive rows with similar column counts (>= 3)
        table_start = None
        table_header_row = None
        in_table = False
        consecutive_multi_col = 0

        for i, row in enumerate(rows):
            num_cols = len(row)
            
            # Check for section markers
            row_text = ' '.join(b['text'] for b in row).strip().upper()
            
            if row_text in ('ITEMS', 'SUMMARY', 'TOTAL'):
                # Explicit section boundary
                if in_table:
                    sections.append({
                        'type': 'table',
                        'header_row': table_header_row,
                        'rows': rows[table_start:i],
                    })
                    in_table = False
                    consecutive_multi_col = 0
                sections.append({'type': 'marker', 'text': row_text, 'row_index': i})
                continue

            if num_cols >= 3:
                consecutive_multi_col += 1
                if consecutive_multi_col >= 2 and not in_table:
                    # Start of a table region (retroactively include first multi-col row)
                    in_table = True
                    table_start = i - consecutive_multi_col + 1
                    table_header_row = rows[table_start] if table_start < len(rows) else None
            else:
                if in_table and consecutive_multi_col < 2:
                    # Single-col row in table context — might be a multi-line description
                    # Keep it in the table if surrounded by table rows
                    pass
                elif in_table:
                    sections.append({
                        'type': 'table',
                        'header_row': table_header_row,
                        'rows': rows[table_start:i],
                    })
                    in_table = False
                consecutive_multi_col = 0

            if not in_table:
                sections.append({'type': 'text', 'row': row, 'row_index': i})

        # Close any open table
        if in_table:
            sections.append({
                'type': 'table',
                'header_row': table_header_row,
                'rows': rows[table_start:],
            })

        return sections

    def _format_sections(self, sections: list) -> str:
        """Format detected sections into readable structured text."""
        output_lines = []
        current_section = "HEADER"

        for section in sections:
            if section['type'] == 'marker':
                marker_text = section['text']
                if 'ITEM' in marker_text:
                    current_section = "ITEMS"
                elif 'SUMMAR' in marker_text:
                    current_section = "SUMMARY"
                output_lines.append(f"\n--- {marker_text} ---")

            elif section['type'] == 'table':
                # Format as markdown-style table
                table_rows = section['rows']
                if not table_rows:
                    continue

                # Determine max columns
                max_cols = max(len(r) for r in table_rows)

                # Build table
                formatted_rows = []
                for row in table_rows:
                    cells = [b['text'] for b in row]
                    # Pad to max_cols
                    while len(cells) < max_cols:
                        cells.append('')
                    formatted_rows.append(cells)

                # Calculate column widths
                col_widths = [0] * max_cols
                for row_cells in formatted_rows:
                    for j, cell in enumerate(row_cells):
                        col_widths[j] = max(col_widths[j], len(cell))

                # Output with alignment
                for k, row_cells in enumerate(formatted_rows):
                    line = '| ' + ' | '.join(
                        cell.ljust(col_widths[j]) for j, cell in enumerate(row_cells)
                    ) + ' |'
                    output_lines.append(line)
                    # Add separator after first row (header)
                    if k == 0:
                        sep = '|' + '|'.join(
                            '-' * (col_widths[j] + 2) for j in range(max_cols)
                        ) + '|'
                        output_lines.append(sep)

            elif section['type'] == 'text':
                row = section['row']
                # Join blocks in the row with appropriate spacing
                line = '    '.join(b['text'] for b in row)
                output_lines.append(line)

        return '\n'.join(output_lines)

class OcrEngine(ABC):
    """Abstract base class for OCR Engines."""

    @abstractmethod
    async def process(self, image_bytes: bytes, filename: str) -> OcrOutput:
        """
        Process the image bytes and return structured OCR output.
        """
        pass

class PaddleOcrEngine(OcrEngine):
    """OCR Engine using PaddleOCR with optional image preprocessing and layout reconstruction."""

    def __init__(self, preprocess: bool = True, layout_reconstruction: bool = True):
        self._ocr = None
        self._model_initialized = False
        self._preprocess = preprocess
        self._layout_reconstruction = layout_reconstruction
        self._preprocessor = None
        self._layout_reconstructor = LayoutReconstructor()
        self._post_processor = OcrPostProcessor()
        
        if preprocess:
            from src.services.processing.preprocessor import ImagePreprocessor
            self._preprocessor = ImagePreprocessor()

    def _get_ocr(self):
        # Lazy initialization so import errors or GPU setup happens on first use
        if self._ocr is None:
            logger.info("[PaddleOCR] Model initialization starting (lazy load)...")
            init_start = time.perf_counter()
            
            from paddleocr import PaddleOCR
            # Disable logger noise from paddleocr
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            
            import_time = time.perf_counter() - init_start
            logger.debug("[PaddleOCR] Import PaddleOCR module took %.2fs", import_time)
            
            create_start = time.perf_counter()
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
            create_time = time.perf_counter() - create_start
            logger.info("[PaddleOCR] Model initialization COMPLETE - creation took %.2fs", create_time)
            
            self._model_initialized = True
        else:
            logger.debug("[PaddleOCR] Model already loaded, reusing instance")
        
        return self._ocr

    def _run_ocr_with_monitoring(self, ocr_instance, temp_file_path: str, filename: str):
        """Run OCR with detailed logging of system resources and processing stages."""
        logger.debug("[%s] Step 2a: Getting initial system resource state...", filename)
        process = psutil.Process(os.getpid())
        
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent_before = process.cpu_percent(interval=0.1)
        logger.debug(
            "[%s] Step 2a DONE - Before OCR: CPU=%.1f%%, Memory=%.1f MB",
            filename,
            cpu_percent_before,
            mem_before,
        )
        
        logger.debug("[%s] Step 2b: Calling ocr_instance.ocr() - this may take time...", filename)
        step2b_start = time.perf_counter()
        
        result = ocr_instance.ocr(temp_file_path)
        
        step2b_time = time.perf_counter() - step2b_start
        logger.debug("[%s] Step 2b DONE (%.2fs) - ocr() call completed", filename, step2b_time)
        
        logger.debug("[%s] Step 2c: Getting final system resource state...", filename)
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent_after = process.cpu_percent(interval=0.1)
        mem_delta = mem_after - mem_before
        
        logger.debug(
            "[%s] Step 2c DONE - After OCR: CPU=%.1f%%, Memory=%.1f MB (delta: %+.1f MB)",
            filename,
            cpu_percent_after,
            mem_after,
            mem_delta,
        )
        
        return result

    async def process(self, image_bytes: bytes, filename: str) -> OcrOutput:
        loop = asyncio.get_running_loop()
        process_start = time.perf_counter()
        
        # Step 0: Image preprocessing (optional)
        if self._preprocessor:
            logger.debug("[%s] Step 0: Preprocessing image...", filename)
            preprocess_start = time.perf_counter()
            image_bytes = await loop.run_in_executor(
                None,
                lambda: self._preprocessor.preprocess(image_bytes, filename)
            )
            preprocess_time = time.perf_counter() - preprocess_start
            logger.info("[%s] Step 0 DONE (%.2fs) - Image preprocessed", filename, preprocess_time)
        else:
            preprocess_time = 0.0

        # Write to a temporary file because paddleocr works best with paths
        logger.debug("[%s] Writing image bytes (size=%.2f MB) to temp file", filename, len(image_bytes) / (1024*1024))
        write_start = time.perf_counter()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        write_time = time.perf_counter() - write_start
        logger.debug("[%s] Temp file written (%.2fs) at %s", filename, write_time, temp_file_path)

        try:
            # Run PaddleOCR in a thread pool to avoid blocking the event loop
            logger.debug("[%s] Step 1: Ensure OCR model is loaded...", filename)
            model_start = time.perf_counter()
            ocr_instance = await loop.run_in_executor(None, self._get_ocr)
            model_time = time.perf_counter() - model_start
            logger.debug("[%s] Step 1 DONE (%.2fs) - model ready", filename, model_time)
            
            logger.debug("[%s] Step 2: Running PaddleOCR.ocr() on temp file...", filename)
            ocr_start = time.perf_counter()
            result = await loop.run_in_executor(
                None,
                lambda: self._run_ocr_with_monitoring(ocr_instance, temp_file_path, filename)
            )
            ocr_time = time.perf_counter() - ocr_start
            logger.info(
                "[%s] Step 2 DONE (%.2fs) - OCR processed, result has %d pages/sections",
                filename,
                ocr_time,
                len(result) if result is not None else 0,
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        logger.debug("[%s] Step 3: Parsing OCR results into structured format...", filename)
        parse_start = time.perf_counter()
        
        text_blocks = []
        confidences = []
        raw_text_lines = []

        # PaddleOCR returns a list of results, one per image/page. We process the first page.
        if result and result[0]:
            page_res = result[0]
            if isinstance(page_res, dict):
                # Paddlex 3.6+ dictionary format
                rec_texts = page_res.get("rec_texts", [])
                rec_scores = page_res.get("rec_scores", [])
                rec_boxes = page_res.get("rec_boxes", [])
                logger.debug("[%s] Format: dictionary, items=%d", filename, len(rec_texts))
                
                # Convert numpy array to list if needed
                if hasattr(rec_boxes, "tolist"):
                    rec_boxes = rec_boxes.tolist()
                
                for i, (text, confidence, bbox) in enumerate(zip(rec_texts, rec_scores, rec_boxes)):
                    int_bbox = [int(x) for x in bbox] if bbox else None
                    text_blocks.append(
                        TextBlock(
                            text=text,
                            confidence=float(confidence),
                            bbox=int_bbox
                        )
                    )
                    confidences.append(confidence)
                    raw_text_lines.append(text)
                    if i < 3 or i >= len(rec_texts) - 3:
                        logger.debug("[%s]   Item %d: text_len=%d, conf=%.3f", filename, i, len(text), float(confidence))
            else:
                # Legacy list-of-lines format
                logger.debug("[%s] Format: legacy list, items=%d", filename, len(page_res))
                for i, line in enumerate(page_res):
                    bbox_points, (text, confidence) = line
                    
                    # Convert 4-point bbox [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [xmin, ymin, xmax, ymax]
                    xs = [pt[0] for pt in bbox_points]
                    ys = [pt[1] for pt in bbox_points]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

                    text_blocks.append(
                        TextBlock(
                            text=text,
                            confidence=float(confidence),
                            bbox=bbox
                        )
                    )
                    confidences.append(confidence)
                    raw_text_lines.append(text)
                    if i < 3 or i >= len(page_res) - 3:
                        logger.debug("[%s]   Line %d: text_len=%d, conf=%.3f", filename, i, len(text), float(confidence))
        
        parse_time = time.perf_counter() - parse_start
        logger.debug("[%s] Step 3 DONE (%.2fs) - extracted %d text blocks", filename, parse_time, len(text_blocks))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Step 4: Layout reconstruction (replaces flat newline join)
        if self._layout_reconstruction and text_blocks:
            logger.debug("[%s] Step 4: Reconstructing layout from %d text blocks...", filename, len(text_blocks))
            layout_start = time.perf_counter()
            raw_text = self._layout_reconstructor.reconstruct(text_blocks, self._post_processor)
            layout_time = time.perf_counter() - layout_start
            logger.info("[%s] Step 4 DONE (%.4fs) - Layout reconstructed", filename, layout_time)
        else:
            # Fallback: flat newline join with post-processing
            raw_text = "\n".join(
                self._post_processor.fix_text(line) for line in raw_text_lines
            )
            layout_time = 0.0

        total_time = time.perf_counter() - process_start
        logger.info(
            "[%s] ✓ OCR COMPLETE - Total time: %.2fs | Breakdown: preprocess=%.2fs, write=%.2fs, model_load=%.2fs, ocr=%.2fs, parse=%.2fs, layout=%.4fs | Text blocks: %d | Avg confidence: %.3f",
            filename,
            total_time,
            preprocess_time,
            write_time,
            model_time,
            ocr_time,
            parse_time,
            layout_time,
            len(text_blocks),
            avg_conf,
        )

        return OcrOutput(
            file_name=filename,
            ocr_engine="paddleocr",
            raw_text=raw_text,
            average_confidence=avg_conf,
            text_blocks=text_blocks
        )

class DoclingOcrEngine(OcrEngine):
    """OCR Engine using IBM's Docling (exports directly to Markdown)."""

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    async def process(self, image_bytes: bytes, filename: str) -> OcrOutput:
        loop = asyncio.get_running_loop()

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            converter = await loop.run_in_executor(None, self._get_converter)
            result = await loop.run_in_executor(
                None,
                lambda: converter.convert(temp_file_path)
            )
            markdown_text = result.document.export_to_markdown()
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        # Docling does not output simple OCR blocks with confidence scores directly.
        # We parse the output by line and assign a default high confidence.
        text_blocks = []
        raw_text_lines = markdown_text.splitlines()
        for line in raw_text_lines:
            if line.strip():
                text_blocks.append(
                    TextBlock(
                        text=line,
                        confidence=0.95,
                        bbox=None
                    )
                )

        return OcrOutput(
            file_name=filename,
            ocr_engine="docling",
            raw_text=markdown_text,
            average_confidence=0.95,
            text_blocks=text_blocks
        )

def create_ocr_engine(engine_name: str, preprocess: bool = True, layout_reconstruction: bool = True) -> OcrEngine:
    if engine_name.lower() == "paddleocr":
        return PaddleOcrEngine(preprocess=preprocess, layout_reconstruction=layout_reconstruction)
    elif engine_name.lower() == "docling":
        return DoclingOcrEngine()
    else:
        raise ValueError(f"Unknown OCR Engine: {engine_name}")
