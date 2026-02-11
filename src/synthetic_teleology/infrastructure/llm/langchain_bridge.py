"""Bridge from legacy LLMProvider to LangChain BaseChatModel.

Provides ``LLMProviderToChatModel`` for users who have existing custom
``LLMProvider`` implementations and want to use them with the new
LLM-first architecture.

Example
-------
::

    from synthetic_teleology.infrastructure.llm import AnthropicProvider
    from synthetic_teleology.infrastructure.llm.langchain_bridge import LLMProviderToChatModel
    from synthetic_teleology.graph import GraphBuilder

    legacy_provider = AnthropicProvider(api_key="...")
    model = LLMProviderToChatModel(provider=legacy_provider, model_name="claude-opus-4-6")

    app, state = (
        GraphBuilder("agent")
        .with_model(model)
        .with_goal("Analyze market trends")
        .build()
    )
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class LLMProviderToChatModel(BaseChatModel):
    """Wraps a legacy ``LLMProvider`` as a LangChain ``BaseChatModel``.

    This bridge allows existing custom LLMProvider implementations to be
    used with the new LLM-first ``GraphBuilder`` and service classes.

    Parameters
    ----------
    provider:
        A legacy ``LLMProvider`` instance.
    model_name:
        The model identifier to use with the provider.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default max tokens per response.
    """

    provider: Any
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return the provider name."""
        try:
            return f"legacy-{self.provider.provider_name}"
        except AttributeError:
            return "legacy-llm-provider"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the legacy provider."""
        from synthetic_teleology.infrastructure.llm import LLMConfig, LLMMessage

        # Convert LangChain messages to legacy format
        legacy_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                legacy_messages.append(LLMMessage(role="system", content=msg.content))
            elif isinstance(msg, HumanMessage):
                legacy_messages.append(LLMMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                legacy_messages.append(LLMMessage(role="assistant", content=msg.content))
            else:
                legacy_messages.append(LLMMessage(role="user", content=str(msg.content)))

        config = LLMConfig(
            model=self.model_name,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop_sequences=tuple(stop or []),
        )

        response = self.provider.generate(legacy_messages, config)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response.text),
                    generation_info={
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    },
                )
            ]
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider_type": self._llm_type,
        }
