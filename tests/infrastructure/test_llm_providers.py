"""Tests for LLM provider abstractions."""

from __future__ import annotations

from typing import Sequence

import pytest

from synthetic_teleology.infrastructure.llm import (
    LLMConfig,
    LLMConnectionError,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
)


class TestLLMConfig:
    """Test LLMConfig validation and creation."""

    def test_default_config(self) -> None:
        config = LLMConfig()
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_custom_config(self) -> None:
        config = LLMConfig(
            model="claude-opus-4-6",
            temperature=0.3,
            max_tokens=2048,
            system_prompt="You are a helper.",
        )
        assert config.model == "claude-opus-4-6"
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.system_prompt == "You are a helper."

    def test_validate_empty_model_raises(self) -> None:
        config = LLMConfig(model="")
        with pytest.raises(ValueError, match="model must not be empty"):
            config.validate()

    def test_validate_invalid_temperature_raises(self) -> None:
        config = LLMConfig(model="test", temperature=3.0)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_validate_invalid_max_tokens_raises(self) -> None:
        config = LLMConfig(model="test", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

    def test_validate_valid_config(self) -> None:
        config = LLMConfig(model="test-model", temperature=1.0, max_tokens=100)
        config.validate()  # Should not raise

    def test_frozen(self) -> None:
        config = LLMConfig(model="test")
        with pytest.raises(AttributeError):
            config.model = "other"  # type: ignore[misc]

    def test_stop_sequences(self) -> None:
        config = LLMConfig(
            model="test",
            stop_sequences=("STOP", "END"),
        )
        assert config.stop_sequences == ("STOP", "END")

    def test_extra_params(self) -> None:
        config = LLMConfig(model="test", extra={"top_p": 0.9})
        assert config.extra["top_p"] == 0.9


class TestLLMMessage:
    """Test LLMMessage data structure."""

    def test_creation(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self) -> None:
        msg = LLMMessage(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "Hi there"}

    def test_frozen(self) -> None:
        msg = LLMMessage(role="user", content="test")
        with pytest.raises(AttributeError):
            msg.role = "assistant"  # type: ignore[misc]


class TestLLMResponse:
    """Test LLMResponse data structure."""

    def test_creation(self) -> None:
        resp = LLMResponse(
            text="Hello world",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
        )
        assert resp.text == "Hello world"
        assert resp.model == "test-model"
        assert resp.usage["input_tokens"] == 10
        assert resp.finish_reason == "stop"

    def test_defaults(self) -> None:
        resp = LLMResponse(text="content")
        assert resp.model == ""
        assert resp.usage == {}
        assert resp.finish_reason == ""

    def test_frozen(self) -> None:
        resp = LLMResponse(text="content")
        with pytest.raises(AttributeError):
            resp.text = "other"  # type: ignore[misc]


class TestLLMExceptions:
    """Test LLM exception hierarchy."""

    def test_base_error(self) -> None:
        err = LLMError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_connection_error(self) -> None:
        err = LLMConnectionError("cannot connect")
        assert isinstance(err, LLMError)

    def test_rate_limit_error(self) -> None:
        err = LLMRateLimitError("too many requests")
        assert isinstance(err, LLMError)

    def test_response_error(self) -> None:
        err = LLMResponseError("bad response")
        assert isinstance(err, LLMError)


class _MockProvider(LLMProvider):
    """A minimal mock LLM provider for testing the abstract interface."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        return LLMResponse(
            text="mock response",
            model=config.model,
        )

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        return LLMResponse(
            text="async mock response",
            model=config.model,
        )


class TestLLMProvider:
    """Test LLM provider interface with a mock."""

    def test_provider_name(self) -> None:
        provider = _MockProvider()
        assert provider.provider_name == "mock"

    def test_generate(self) -> None:
        provider = _MockProvider()
        config = LLMConfig(model="test-model", temperature=0.5, max_tokens=100)
        messages = [LLMMessage(role="user", content="Hello")]
        response = provider.generate(messages, config)
        assert response.text == "mock response"
        assert response.model == "test-model"

    def test_validate_config_delegates(self) -> None:
        provider = _MockProvider()
        config = LLMConfig(model="", temperature=0.5)
        with pytest.raises(ValueError, match="model must not be empty"):
            provider.validate_config(config)

    def test_validate_config_valid(self) -> None:
        provider = _MockProvider()
        config = LLMConfig(model="test-model")
        provider.validate_config(config)  # Should not raise

    @pytest.mark.asyncio
    async def test_generate_async(self) -> None:
        provider = _MockProvider()
        config = LLMConfig(model="test-model", temperature=0.5, max_tokens=100)
        messages = [LLMMessage(role="user", content="Hello")]
        response = await provider.generate_async(messages, config)
        assert response.text == "async mock response"
