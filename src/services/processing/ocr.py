import abc
import asyncio
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

from src.schemas.document import OcrOutput, TextBlock

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
    """OCR Engine using PaddleOCR."""

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        # Lazy initialization so import errors or GPU setup happens on first use
        if self._ocr is None:
            from paddleocr import PaddleOCR
            # Disable logger noise from paddleocr
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
        return self._ocr

    async def process(self, image_bytes: bytes, filename: str) -> OcrOutput:
        loop = asyncio.get_running_loop()
        
        # Write to a temporary file because paddleocr works best with paths
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            # Run PaddleOCR in a thread pool to avoid blocking the event loop
            ocr_instance = await loop.run_in_executor(None, self._get_ocr)
            result = await loop.run_in_executor(
                None, 
                lambda: ocr_instance.ocr(temp_file_path)
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

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
                
                # Convert numpy array to list if needed
                if hasattr(rec_boxes, "tolist"):
                    rec_boxes = rec_boxes.tolist()
                
                for text, confidence, bbox in zip(rec_texts, rec_scores, rec_boxes):
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
            else:
                # Legacy list-of-lines format
                for line in page_res:
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

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        raw_text = "\n".join(raw_text_lines)

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

def create_ocr_engine(engine_name: str) -> OcrEngine:
    if engine_name.lower() == "paddleocr":
        return PaddleOcrEngine()
    elif engine_name.lower() == "docling":
        return DoclingOcrEngine()
    else:
        raise ValueError(f"Unknown OCR Engine: {engine_name}")
