import logging
from abc import ABC, abstractmethod
from typing import Optional

from ollama import AsyncClient
from google import genai
from google.genai import types

from src.schemas.document import OcrOutput, InvoiceExtraction

logger = logging.getLogger(__name__)


class LlmExtractor(ABC):
    """Abstract interface for LLM extraction providers."""

    @abstractmethod
    async def extract(self, ocr_output: OcrOutput) -> InvoiceExtraction:
        """Extract structured data from raw OCR output."""
        pass


class OllamaExtractor(LlmExtractor):
    """LLM Structured Extractor using local Ollama instance."""

    def __init__(self, host: str, model_name: str):
        self._host = host
        self._model_name = model_name
        self._client = AsyncClient(host=host)

    async def extract(self, ocr_output: OcrOutput) -> InvoiceExtraction:
        """
        Extract structured invoice data from raw OCR text using Ollama.
        """
        system_prompt = _build_system_prompt()
        user_content = _build_user_content(ocr_output)

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


class GeminiExtractor(LlmExtractor):
    """LLM Structured Extractor using Google Gemini API."""

    def __init__(self, api_key: str, model_name: str):
        self._api_key = api_key
        self._model_name = model_name
        self._client = genai.Client(api_key=api_key)

    async def extract(self, ocr_output: OcrOutput) -> InvoiceExtraction:
        """
        Extract structured invoice data from raw OCR text using Google Gemini API.
        """
        system_prompt = _build_system_prompt()
        user_content = _build_user_content(ocr_output)

        try:
            logger.info("Sending extraction request to Gemini with model %s", self._model_name)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=InvoiceExtraction,
                temperature=0.0
            )

            # Using client.aio for asynchronous generation
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=user_content,
                config=config
            )

            content = response.text
            logger.debug("Received raw response from Gemini: %s", content)

            if not content:
                raise ValueError("Received empty response from Gemini API.")

            # Validate the JSON matches the schema
            try:
                extraction = InvoiceExtraction.model_validate_json(content)
                return extraction
            except Exception as val_exc:
                logger.warning("First attempt schema validation failed: %s. Retrying with explicit request...", val_exc)
                # Retry once
                retry_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=InvoiceExtraction,
                    temperature=0.0
                )
                retry_content = [
                    user_content,
                    content,
                    f"The response was invalid JSON or did not match the schema: {val_exc}. Please fix it and return valid JSON adhering strictly to the schema."
                ]
                response_retry = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=retry_content,
                    config=retry_config
                )
                return InvoiceExtraction.model_validate_json(response_retry.text)

        except Exception as exc:
            logger.exception("Failed to extract data via Gemini: %s", exc)
            raise


def create_extractor(settings) -> LlmExtractor:
    """Factory function to create an LLM Extractor based on configuration settings."""
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaExtractor(
            host=settings.ollama.host,
            model_name=settings.ollama.model
        )
    elif provider == "gemini":
        if not settings.gemini.api_key:
            raise ValueError(
                "Gemini API key is required but missing from configuration. "
                "Please set GEMINI_API_KEY in your environment/settings."
            )
        return GeminiExtractor(
            api_key=settings.gemini.api_key,
            model_name=settings.gemini.model
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def _build_system_prompt() -> str:
    """Build a detailed layout-aware system prompt with rules and few-shot examples."""
    return (
        "You are a highly precise document extraction AI. Your task is to extract structured information "
        "from the OCR text of invoices and receipts.\n\n"
        "You must return a JSON object that adheres strictly to the target JSON schema.\n\n"
        "CRITICAL EXTRACTION RULES:\n"
        "1. MANDATORY FIELDS:\n"
        "   - `header.invoice_no`: The invoice or receipt number. Look for labels like 'Invoice No', 'Inv #', 'Receipt No', 'Document No', 'Invoice Number', etc.\n"
        "   - `header.invoice_date`: The invoice date exactly as written. Do NOT reformat it (e.g. keep '12.04.2024' or '03/O5/2O17' or '2017-05-03' as is).\n"
        "   - `summary.total_net_worth`: The total net amount before taxes.\n\n"
        "2. OPTIONAL FIELDS:\n"
        "   - Extract optional fields ONLY if they are clearly present in the document.\n"
        "   - Set optional fields to null if they cannot be found or are not present. Do not make up values.\n"
        "   - Optional fields include: `header.seller`, `header.client`, `header.seller_tax_id`, `header.client_tax_id`, `header.iban`, `items`, `summary.total_vat`, `summary.total_gross_worth`.\n\n"
        "3. FORMAT PRESERVATION:\n"
        "   - Do NOT clean, normalize, or reformat numbers, dates, or tax IDs.\n"
        "   - Keep all currency symbols, spaces, commas, decimals, and separators EXACTLY as they appear in the OCR text.\n"
        "   - Example: If the total net worth is printed as '24 161,60' or '$ 2,500.00', extract it exactly as '24 161,60' or '$ 2,500.00'. Do NOT convert to standard float format or clean spaces.\n\n"
        "4. LINE ITEMS:\n"
        "   - Each item in the `items` list must correspond to a single invoice line item.\n"
        "   - Extract item description, quantity, net unit price, net worth, VAT, and gross worth exactly as they appear.\n"
        "   - Group text lines belonging to the same item together.\n\n"
        "FEW-SHOT EXAMPLE:\n\n"
        "Input OCR Text:\n"
        "--- HEADER REGION ---\n"
        "Invoice No: INV-100234       Date: 12.04.2024\n"
        "Seller: Acme Industrial Corp   Client: Globex Corporation\n"
        "Seller Tax ID: DE123456789    Client Tax ID: DE987654321\n"
        "IBAN: DE89370400440532013000\n\n"
        "--- TABLE REGION ---\n"
        "| Description | Qty | Net Price | Net Worth | VAT | Gross Worth |\n"
        "| Paper Clips Box | 10 | 2.50 | 25.00 | 19% | 29.75 |\n"
        "| Heavy Duty Stapler | 2 | 15.00 | 30.00 | 19% | 35.70 |\n\n"
        "--- SUMMARY REGION ---\n"
        "Total Net Worth: 55.00\n"
        "Total VAT: 10.45\n"
        "Total Gross Worth: 65.45\n\n"
        "Expected Output JSON:\n"
        "{\n"
        "  \"header\": {\n"
        "    \"invoice_no\": \"INV-100234\",\n"
        "    \"invoice_date\": \"12.04.2024\",\n"
        "    \"seller\": \"Acme Industrial Corp\",\n"
        "    \"client\": \"Globex Corporation\",\n"
        "    \"seller_tax_id\": \"DE123456789\",\n"
        "    \"client_tax_id\": \"DE987654321\",\n"
        "    \"iban\": \"DE89370400440532013000\"\n"
        "  },\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"item_desc\": \"Paper Clips Box\",\n"
        "      \"item_qty\": \"10\",\n"
        "      \"item_net_price\": \"2.50\",\n"
        "      \"item_net_worth\": \"25.00\",\n"
        "      \"item_vat\": \"19%\",\n"
        "      \"item_gross_worth\": \"29.75\"\n"
        "    },\n"
        "    {\n"
        "      \"item_desc\": \"Heavy Duty Stapler\",\n"
        "      \"item_qty\": \"2\",\n"
        "      \"item_net_price\": \"15.00\",\n"
        "      \"item_net_worth\": \"30.00\",\n"
        "      \"item_vat\": \"19%\",\n"
        "      \"item_gross_worth\": \"35.70\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"total_net_worth\": \"55.00\",\n"
        "    \"total_vat\": \"10.45\",\n"
        "    \"total_gross_worth\": \"65.45\"\n"
        "  }\n"
        "}"
    )


def _build_user_content(ocr_output: OcrOutput) -> str:
    """Build user message content with metadata and layout-reconstructed text."""
    return (
        f"Please extract the structured invoice information from the following OCR text.\n\n"
        f"File Name: {ocr_output.file_name}\n"
        f"OCR Engine: {ocr_output.ocr_engine}\n\n"
        f"OCR Text:\n"
        f"{ocr_output.raw_text}"
    )


