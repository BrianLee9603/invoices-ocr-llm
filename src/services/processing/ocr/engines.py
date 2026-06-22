import abc
import asyncio
import logging
import os
import tempfile
import time
import psutil
from abc import ABC, abstractmethod
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from src.schemas.document import OcrOutput, TextBlock
from src.services.processing.ocr.layout import LayoutReconstructor
from src.services.processing.ocr.post_processor import OcrPostProcessor
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


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
        
        # Load app settings
        settings = get_settings().processing
        self._ocr_version = settings.ocr_version
        self._det_limit_side_len = settings.det_limit_side_len
        self._max_workers = settings.ocr_max_workers
        self._fallback_enabled = settings.fallback_enabled
        
        # Setup fixed-size ThreadPoolExecutor for PaddleOCR to manage memory
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="ocr")
        
        if preprocess:
            from src.services.processing.ocr.pre_processor import ImagePreprocessor
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
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                ocr_version=self._ocr_version,
                det_limit_side_len=self._det_limit_side_len,
            )
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
        original_image_bytes = image_bytes
        
        # Step 0: Image preprocessing (optional)
        if self._preprocessor:
            logger.debug("[%s] Step 0: Preprocessing image...", filename)
            preprocess_start = time.perf_counter()
            image_bytes = await loop.run_in_executor(
                self._executor,
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
            ocr_instance = await loop.run_in_executor(self._executor, self._get_ocr)
            model_time = time.perf_counter() - model_start
            logger.debug("[%s] Step 1 DONE (%.2fs) - model ready", filename, model_time)
            
            logger.debug("[%s] Step 2: Running PaddleOCR.ocr() on temp file...", filename)
            ocr_start = time.perf_counter()
            result = await loop.run_in_executor(
                self._executor,
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

        # Check for fallback retry if 0 text blocks detected
        if not text_blocks and self._preprocess and self._fallback_enabled:
            logger.warning(
                "[%s] OCR returned 0 text blocks — retrying WITHOUT preprocessing",
                filename,
            )
            # Re-run OCR on raw image bytes, reuse same engine context but disable preprocessing
            fallback_engine = PaddleOcrEngine(preprocess=False, layout_reconstruction=self._layout_reconstruction)
            fallback_engine._ocr = self._ocr
            fallback_engine._model_initialized = self._model_initialized
            fallback_engine._executor = self._executor
            return await fallback_engine.process(original_image_bytes, filename)

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
