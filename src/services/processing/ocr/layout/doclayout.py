import logging
import os
from typing import List, Dict, Any
import numpy as np
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

logger = logging.getLogger(__name__)


class DocLayoutYoloAnalyzer:
    """
    Layout Analyzer using DocLayout-YOLO (YOLOv10) to detect document regions:
    text, title, table, key-value, other.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize DocLayout-YOLO. If model_path is None, downloads the recommended
        weights from Hugging Face.
        """
        if model_path is None:
            logger.info("Fetching DocLayout-YOLO model weights from Hugging Face...")
            try:
                model_path = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                )
                logger.info(f"DocLayout-YOLO model cached at: {model_path}")
            except Exception as e:
                logger.exception("Failed to download DocLayout-YOLO weights from Hugging Face")
                raise RuntimeError(f"Failed to fetch model weights: {e}") from e

        self.model_path = model_path
        self._model = None

    @property
    def model(self) -> YOLOv10:
        """Lazy load the YOLO model to save memory until usage."""
        if self._model is None:
            logger.info("Initializing DocLayout-YOLO model: %s", self.model_path)
            try:
                self._model = YOLOv10(self.model_path)
                import torch
                # Enable GPU if torch supports it (even if CPU-only torch, it won't crash)
                if torch.cuda.is_available():
                    self._model.to("cuda")
                    logger.info("DocLayout-YOLO initialized successfully on GPU (CUDA)")
                else:
                    logger.info("DocLayout-YOLO initialized successfully on CPU")
            except Exception as e:
                logger.exception("Failed to load YOLO model")
                raise RuntimeError(f"Failed to initialize DocLayout-YOLO: {e}") from e
        return self._model

    def detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect layout regions in the image.

        Args:
            image: OpenCV BGR image.

        Returns:
            A list of dictionaries representing detected regions:
            [
                {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'label': str,
                    'confidence': float
                },
                ...
            ]
        """
        if image is None or image.size == 0:
            logger.warning("Empty image passed to detect_regions")
            return []

        try:
            # Predict with recommended image size
            results = self.model.predict(image, imgsz=1024, verbose=False)
            if not results:
                return []

            result = results[0]
            boxes = result.boxes
            names = result.names  # Class map, e.g., {0: 'text', 1: 'title', ...}

            regions = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                xmin, ymin, xmax, ymax = [int(val) for val in xyxy]

                regions.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "label": label.lower(),
                    "confidence": confidence
                })

            logger.info("DocLayout-YOLO detected %d regions in document", len(regions))
            return regions
        except Exception as e:
            logger.exception("Error during DocLayout-YOLO region detection")
            return []
