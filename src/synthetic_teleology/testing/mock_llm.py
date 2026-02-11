"""Mock LLM for testing and examples.

Provides a ``MockStructuredChatModel`` that supports ``with_structured_output``
by returning pre-configured Pydantic model instances.  Works with any
``BaseChatModel``-based service (evaluators, planners, revisers, etc.).
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, ConfigDict


class _StructuredOutputRunnable(RunnableSerializable):
    """A runnable that returns a pre-configured Pydantic instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instance: Any

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.instance


class MockStructuredChatModel(BaseChatModel):
    """A mock chat model that supports with_structured_output.

    Usage::

        model = MockStructuredChatModel(
            structured_responses=[my_pydantic_instance_1, my_pydantic_instance_2],
        )
        # Each call to the chain will return the next response in order.
        # After exhausting the list, it cycles back to the start.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    structured_responses: list[Any] = []
    _call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock-structured"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self._call_index % len(self.structured_responses) if self.structured_responses else 0
        resp = self.structured_responses[idx] if self.structured_responses else ""
        self._call_index += 1

        text = resp.model_dump_json() if isinstance(resp, BaseModel) else str(resp)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that yields pre-configured structured responses."""
        model_ref = self

        class _MultiStructuredRunnable(RunnableSerializable):
            """Returns responses in sequence, cycling."""

            model_config = ConfigDict(arbitrary_types_allowed=True)

            def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
                nonlocal model_ref
                idx = model_ref._call_index % len(model_ref.structured_responses)
                resp = model_ref.structured_responses[idx]
                model_ref._call_index += 1
                return resp

        return _MultiStructuredRunnable()
