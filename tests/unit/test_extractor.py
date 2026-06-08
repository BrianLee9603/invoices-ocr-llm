import pytest
from unittest.mock import MagicMock, patch
from src.config.settings import AppSettings, OllamaSettings, GeminiSettings
from src.services.processing.extractor import create_extractor, OllamaExtractor, GeminiExtractor

def test_create_extractor_ollama():
    settings = MagicMock(spec=AppSettings)
    settings.llm_provider = "ollama"
    settings.ollama = MagicMock(spec=OllamaSettings)
    settings.ollama.host = "http://localhost:11434"
    settings.ollama.model = "qwen2.5:1.5b"
    
    extractor = create_extractor(settings)
    assert isinstance(extractor, OllamaExtractor)
    assert extractor._host == "http://localhost:11434"
    assert extractor._model_name == "qwen2.5:1.5b"

def test_create_extractor_gemini():
    settings = MagicMock(spec=AppSettings)
    settings.llm_provider = "gemini"
    settings.gemini = MagicMock(spec=GeminiSettings)
    settings.gemini.api_key = "test-api-key"
    settings.gemini.model = "gemini-2.5-flash"
    
    with patch("google.genai.Client") as mock_client:
        extractor = create_extractor(settings)
        assert isinstance(extractor, GeminiExtractor)
        assert extractor._model_name == "gemini-2.5-flash"
        assert extractor._api_key == "test-api-key"

def test_create_extractor_gemini_missing_key():
    settings = MagicMock(spec=AppSettings)
    settings.llm_provider = "gemini"
    settings.gemini = MagicMock(spec=GeminiSettings)
    settings.gemini.api_key = None
    
    with pytest.raises(ValueError, match="Gemini API key is required"):
        create_extractor(settings)

def test_create_extractor_invalid_provider():
    settings = MagicMock(spec=AppSettings)
    settings.llm_provider = "invalid"
    
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_extractor(settings)
