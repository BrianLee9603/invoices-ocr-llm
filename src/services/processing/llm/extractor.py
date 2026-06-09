import logging
from abc import ABC, abstractmethod
from typing import Optional

from ollama import AsyncClient
from google import genai
from google.genai import types

from src.schemas.document import OcrOutput, InvoiceExtraction
from src.services.processing.llm.prompts import _build_system_prompt, _build_user_content
from src.exceptions import TransientError, PersistentError

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
                options={
                    "temperature": 0.0,
                    "num_ctx": 2048,
                    "num_predict": 512
                }
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
                try:
                    response_retry = await self._client.chat(
                        model=self._model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": f"The response was invalid JSON or did not match the schema: {val_exc}. Please fix it and return valid JSON adhering strictly to the schema."}
                        ],
                        format=InvoiceExtraction.model_json_schema(),
                        options={
                            "temperature": 0.0,
                            "num_ctx": 2048,
                            "num_predict": 512
                        }
                    )
                    return InvoiceExtraction.model_validate_json(response_retry.message.content)
                except Exception as retry_val_exc:
                    logger.error("Second attempt schema validation failed: %s", retry_val_exc)
                    raise PersistentError(f"Ollama extracted JSON failed validation: {retry_val_exc}") from retry_val_exc

        except (TransientError, PersistentError):
            raise
        except Exception as exc:
            exc_name = type(exc).__name__
            if "Connect" in exc_name or "Timeout" in exc_name or "MaxRetry" in exc_name or "HTTP" in exc_name or "ResponseError" in exc_name:
                logger.warning("Transient network failure during Ollama extraction: %s", exc)
                raise TransientError(f"Ollama transient failure: {exc}") from exc
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
                try:
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
                except Exception as retry_val_exc:
                    logger.error("Second attempt schema validation failed: %s", retry_val_exc)
                    raise PersistentError(f"Gemini extracted JSON failed validation: {retry_val_exc}") from retry_val_exc

        except (TransientError, PersistentError):
            raise
        except Exception as exc:
            exc_name = type(exc).__name__
            if "APIError" in exc_name or "Connect" in exc_name or "Timeout" in exc_name or "ResourceExhausted" in exc_name or "ServiceUnavailable" in exc_name:
                logger.warning("Transient error during Gemini extraction: %s", exc)
                raise TransientError(f"Gemini API transient failure: {exc}") from exc
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
