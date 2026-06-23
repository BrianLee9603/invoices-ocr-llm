"""
A/B testing framework for OCR preprocessing configurations.

Runs multiple preprocessing configurations against the same image set,
measures CER/WER, and generates comparison reports.
"""

import asyncio
import json
import logging
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.services.processing.ocr.engines.engines import PaddleOcrEngine
from src.services.processing.ocr.evaluation.metrics import (
    character_error_rate,
    word_error_rate,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark variant."""
    name: str
    preprocess: bool
    layout_reconstruction: bool
    description: str = ""


@dataclass
class BenchmarkResult:
    """Results for a single image + config combination."""
    config_name: str
    filename: str
    cer: float
    wer: float
    avg_confidence: float
    num_blocks: int
    latency_ms: float
    raw_text: str


class OcrBenchmark:
    """
    Run A/B benchmarks comparing preprocessing configurations.
    
    Usage:
        benchmark = OcrBenchmark()
        benchmark.add_config("baseline", preprocess=True, layout=True)
        benchmark.add_config("no_preprocess", preprocess=False, layout=True)
        results = await benchmark.run("data/sample/", "data/ground_truth/")
    """
    
    def __init__(self):
        self._configs: List[BenchmarkConfig] = []
    
    def add_config(
        self, name: str, preprocess: bool = True,
        layout: bool = True, description: str = ""
    ):
        self._configs.append(BenchmarkConfig(
            name=name, preprocess=preprocess,
            layout_reconstruction=layout, description=description,
        ))
    
    async def run(
        self,
        image_dir: str,
        ground_truth_dir: Optional[str] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run all configs against all images in the directory."""
        image_path = Path(image_dir)
        images = list(image_path.glob("*.jpg")) + \
                 list(image_path.glob("*.png")) + \
                 list(image_path.glob("*.jpeg"))
        
        if not images:
            logger.warning("No images found in directory: %s", image_dir)
            return {}

        all_results: Dict[str, List[BenchmarkResult]] = {}
        
        for config in self._configs:
            logger.info("Running benchmark config: %s", config.name)
            # Create engine with the custom settings
            engine = PaddleOcrEngine(
                preprocess=config.preprocess,
                layout_reconstruction=config.layout_reconstruction,
            )
            
            results = []
            for img_path in images:
                logger.info("Processing image: %s with config: %s", img_path.name, config.name)
                image_bytes = img_path.read_bytes()
                
                # Load ground truth if available
                gt_text = ""
                if ground_truth_dir:
                    gt_path = Path(ground_truth_dir) / f"{img_path.stem}.txt"
                    if gt_path.exists():
                        gt_text = gt_path.read_text(encoding="utf-8")
                
                # Run OCR with timing
                start = time.perf_counter()
                try:
                    ocr_output = await engine.process(image_bytes, img_path.name)
                    latency = (time.perf_counter() - start) * 1000
                    
                    # Compute metrics
                    cer = character_error_rate(ocr_output.raw_text, gt_text) if gt_text else -1.0
                    wer = word_error_rate(ocr_output.raw_text, gt_text) if gt_text else -1.0
                    
                    results.append(BenchmarkResult(
                        config_name=config.name,
                        filename=img_path.name,
                        cer=cer,
                        wer=wer,
                        avg_confidence=ocr_output.average_confidence,
                        num_blocks=len(ocr_output.text_blocks),
                        latency_ms=latency,
                        raw_text=ocr_output.raw_text,
                    ))
                except Exception as e:
                    logger.error("Failed to run OCR for %s: %s", img_path.name, str(e))
            
            all_results[config.name] = results
            
            if results:
                # Log aggregate stats
                avg_conf = sum(r.avg_confidence for r in results) / len(results)
                avg_latency = sum(r.latency_ms for r in results) / len(results)
                
                valid_cer = [r.cer for r in results if r.cer >= 0]
                valid_wer = [r.wer for r in results if r.wer >= 0]
                
                avg_cer = sum(valid_cer) / len(valid_cer) if valid_cer else -1.0
                avg_wer = sum(valid_wer) / len(valid_wer) if valid_wer else -1.0
                
                logger.info(
                    "Config '%s' Summary: %d images, avg_confidence=%.3f, avg_latency=%.0fms, avg_cer=%.3f, avg_wer=%.3f",
                    config.name, len(results), avg_conf, avg_latency, avg_cer, avg_wer,
                )
        
        return all_results


async def main():
    parser = argparse.ArgumentParser(description="Run OCR Preprocessing configurations benchmark.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--gt-dir", type=str, default=None, help="Directory containing ground truth text files.")
    args = parser.parse_args()

    benchmark = OcrBenchmark()
    benchmark.add_config("baseline_with_preprocess", preprocess=True, layout=True)
    benchmark.add_config("raw_no_preprocess", preprocess=False, layout=True)

    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    results = await benchmark.run(args.image_dir, args.gt_dir)
    print(f"Benchmark run complete on {args.image_dir}")


if __name__ == "__main__":
    asyncio.run(main())
