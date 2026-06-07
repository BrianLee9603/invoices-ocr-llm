import logging
import json
from ollama import AsyncClient
from src.schemas.document import OcrOutput, InvoiceExtraction

logger = logging.getLogger(__name__)

class OllamaExtractor:
    """LLM Structured Extractor using local Ollama instance."""

    def __init__(self, host: str, model_name: str):
        self._host = host
        self._model_name = model_name
        self._client = AsyncClient(host=host)

    async def extract(self, ocr_output: OcrOutput) -> InvoiceExtraction:
        """
        Extract structured invoice data from raw OCR text using Ollama.
        """
        system_prompt = (
            "You are a highly precise document extraction AI. Your task is to extract structured JSON "
            "data from the provided OCR text of an invoice or receipt. The output MUST conform strictly to "
            "the requested JSON schema.\n"
            "Guidelines:\n"
            "1. Extract header details, line items, and summary totals accurately.\n"
            "2. Map any missing fields to null.\n"
            "3. If a value is present, extract it as close as possible to the raw string representation.\n"
            "4. Follow the Pydantic schema precisely."
        )

        user_content = (
            f"Document Filename: {ocr_output.file_name}\n"
            f"OCR Engine Used: {ocr_output.ocr_engine}\n\n"
            f"OCR / Parse Content:\n"
            f"-------------------\n"
            f"{ocr_output.raw_text}\n"
            f"-------------------\n"
        )

        try:
            logger.info("Sending extraction request to Ollama (%s) with model %s", self._host, self._model_name)
            response = await self._client.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                format=InvoiceExtraction.model_json_schema(),
                options={"temperature": 0.0}
            )

            content = response.message.content
            logger.debug("Received raw response from Ollama: %s", content)

            # Validate the JSON matches the schema
            try:
                extraction = InvoiceExtraction.model_validate_json(content)
                return extraction
            except Exception as val_exc:
                logger.warning("First attempt schema validation failed: %s. Retrying with explicit request...", val_exc)
                # Retry once
                response_retry = await self._client.chat(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"The response was invalid JSON or did not match the schema: {val_exc}. Please fix it and return valid JSON adhering strictly to the schema."}
                    ],
                    format=InvoiceExtraction.model_json_schema(),
                    options={"temperature": 0.0}
                )
                return InvoiceExtraction.model_validate_json(response_retry.message.content)

        except Exception as exc:
            logger.exception("Failed to extract data via Ollama: %s", exc)
            raise
