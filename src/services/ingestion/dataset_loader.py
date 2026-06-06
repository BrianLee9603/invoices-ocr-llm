"""
HuggingFace dataset loader for ``mychen76/invoices-and-receipts_ocr_v1``.

Yields (image_bytes, doc_id, parsed_data_dict) tuples, converting PIL
images to PNG bytes on the fly.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

DATASET_NAME = "mychen76/invoices-and-receipts_ocr_v1"


def load_hf_dataset(
    split: str = "test",
    limit: int | None = None,
) -> Generator[tuple[bytes, str, dict[str, Any]], None, None]:
    """
    Load the HuggingFace dataset and yield individual samples.

    Args:
        split:  Dataset split — ``train``, ``test``, or ``valid``.
        limit:  Maximum number of samples. ``None`` = entire split.

    Yields:
        ``(image_bytes, doc_id, parsed_data)`` — PNG bytes, document ID,
        and the parsed ground-truth dictionary.
    """
    # Import here to avoid heavy startup cost when not using dataset
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading HuggingFace dataset '%s' (split=%s)...", DATASET_NAME, split)
    ds = load_dataset(DATASET_NAME, split=split)

    count = 0
    for row in ds:
        if limit is not None and count >= limit:
            break

        # Convert PIL Image → PNG bytes
        img = row["image"]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Parse ground truth JSON string → dict
        doc_id = row.get("id", f"hf_{count}")
        raw_parsed = row.get("parsed_data", "{}")
        if isinstance(raw_parsed, str):
            try:
                parsed_data = json.loads(raw_parsed)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in parsed_data for doc %s", doc_id)
                parsed_data = {"raw": raw_parsed}
        else:
            parsed_data = raw_parsed

        yield image_bytes, doc_id, parsed_data
        count += 1

    logger.info("Loaded %d samples from '%s' (split=%s).", count, DATASET_NAME, split)
