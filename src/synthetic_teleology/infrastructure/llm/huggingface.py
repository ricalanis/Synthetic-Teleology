"""HuggingFace local inference provider for the Synthetic Teleology LLM layer.

Uses the ``transformers`` library's ``pipeline`` API for local model inference
without any API calls.  Supports text-generation pipelines with configurable
model, tokenizer, and generation parameters.

Requires the ``transformers`` and ``torch`` packages.  If not installed,
a clear error is raised at instantiation time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from synthetic_teleology.infrastructure.llm import (
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
)

logger = logging.getLogger(__name__)

# Attempt imports at module level
try:
    import transformers as _transformers
    import torch as _torch

    _HAS_TRANSFORMERS = True
except ImportError:
    _transformers = None  # type: ignore[assignment]
    _torch = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False


def _check_transformers_available() -> None:
    """Raise a clear error if transformers/torch are not installed."""
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "The 'transformers' and 'torch' packages are required for "
            "HuggingFaceLocalProvider. Install them with: "
            "pip install transformers torch"
        )


class HuggingFaceLocalProvider(LLMProvider):
    """LLM provider for local HuggingFace model inference.

    Loads a model via ``transformers.pipeline("text-generation", ...)``
    and runs inference entirely on the local machine.  No network calls
    are made after model loading.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"microsoft/phi-2"``,
        ``"meta-llama/Llama-2-7b-chat-hf"``).
    device:
        Device to run inference on.  ``"auto"`` uses accelerate to
        distribute across available devices.  Defaults to ``"auto"``.
    torch_dtype:
        PyTorch dtype for model weights.  ``"auto"`` lets the model
        config decide.  Defaults to ``"auto"``.
    trust_remote_code:
        Whether to trust and execute code from the model repo.
        Defaults to ``False``.
    pipeline_kwargs:
        Additional keyword arguments passed to
        ``transformers.pipeline()``.
    lazy_load:
        If ``True``, defer model loading until the first ``generate``
        call.  Defaults to ``True``.

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        pipeline_kwargs: dict[str, Any] | None = None,
        lazy_load: bool = True,
    ) -> None:
        _check_transformers_available()

        self._model_name = model_name
        self._device = device
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._pipeline_kwargs = pipeline_kwargs or {}
        self._pipeline: Any = None
        self._is_loaded = False

        if not lazy_load:
            self._load_pipeline()

    @property
    def provider_name(self) -> str:
        """Return ``'huggingface'``."""
        return "huggingface"

    @property
    def is_loaded(self) -> bool:
        """Whether the model pipeline has been loaded."""
        return self._is_loaded

    def _load_pipeline(self) -> None:
        """Load the HuggingFace pipeline. Thread-safe via simple flag check."""
        if self._is_loaded:
            return

        logger.info(
            "HuggingFaceLocalProvider: loading model %r (device=%s, dtype=%s)",
            self._model_name,
            self._device,
            self._torch_dtype,
        )

        # Resolve torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": _torch.float16,
            "float32": _torch.float32,
            "bfloat16": _torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(self._torch_dtype, "auto")

        try:
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self._model_name,
                "trust_remote_code": self._trust_remote_code,
                **self._pipeline_kwargs,
            }

            if self._device != "auto":
                pipeline_kwargs["device"] = self._device
            else:
                pipeline_kwargs["device_map"] = "auto"

            if resolved_dtype != "auto":
                pipeline_kwargs["torch_dtype"] = resolved_dtype

            self._pipeline = _transformers.pipeline(**pipeline_kwargs)
            self._is_loaded = True

            logger.info(
                "HuggingFaceLocalProvider: model %r loaded successfully",
                self._model_name,
            )

        except Exception as exc:
            raise LLMError(
                f"Failed to load HuggingFace model {self._model_name!r}: {exc}"
            ) from exc

    def generate(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate a response using the local HuggingFace pipeline.

        Parameters
        ----------
        messages:
            Conversation history.
        config:
            Per-call configuration.

        Returns
        -------
        LLMResponse
        """
        self.validate_config(config)
        self._load_pipeline()

        prompt = self._build_prompt(messages, config.system_prompt)

        try:
            # Run the pipeline
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_tokens,
                "temperature": max(0.01, config.temperature),  # Avoid 0
                "do_sample": config.temperature > 0.01,
                "return_full_text": False,
            }

            if config.stop_sequences:
                # HuggingFace uses eos_token_id or stopping_criteria
                # For simplicity, we handle stop sequences in post-processing
                pass

            # Pass through any extra generation kwargs
            for key, value in config.extra.items():
                generation_kwargs[key] = value

            results = self._pipeline(
                prompt,
                **generation_kwargs,
            )

            if not results:
                raise LLMResponseError("HuggingFace pipeline returned empty results")

            generated_text = results[0].get("generated_text", "")

            # Post-process: apply stop sequences
            if config.stop_sequences:
                for stop_seq in config.stop_sequences:
                    idx = generated_text.find(stop_seq)
                    if idx != -1:
                        generated_text = generated_text[:idx]

            # Estimate token counts
            tokenizer = getattr(self._pipeline, "tokenizer", None)
            usage: dict[str, int] = {}
            if tokenizer is not None:
                try:
                    input_tokens = len(tokenizer.encode(prompt))
                    output_tokens = len(tokenizer.encode(generated_text))
                    usage = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                except Exception:
                    pass

            return LLMResponse(
                text=generated_text,
                model=self._model_name,
                usage=usage,
                finish_reason="stop",
                raw={"pipeline": type(self._pipeline).__name__},
            )

        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMError(
                f"HuggingFace pipeline inference failed: {exc}"
            ) from exc

    async def generate_async(
        self,
        messages: Sequence[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Asynchronously generate a response by wrapping sync inference.

        Since HuggingFace pipelines are synchronous, this method runs
        the synchronous ``generate`` in a thread pool via
        ``asyncio.to_thread``.
        """
        return await asyncio.to_thread(self.generate, messages, config)

    # -- internal helpers -----------------------------------------------------

    def _build_prompt(
        self,
        messages: Sequence[LLMMessage],
        system_prompt: str = "",
    ) -> str:
        """Convert messages to a single prompt string.

        Uses a simple chat template format compatible with most models.
        If the pipeline's tokenizer has a ``apply_chat_template`` method,
        we use that instead.
        """
        # Try using the tokenizer's chat template
        tokenizer = getattr(self._pipeline, "tokenizer", None) if self._pipeline else None
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                chat_messages = []
                if system_prompt:
                    chat_messages.append({
                        "role": "system",
                        "content": system_prompt,
                    })
                for msg in messages:
                    chat_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

                return tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.debug(
                    "HuggingFaceLocalProvider: apply_chat_template failed, "
                    "falling back to manual formatting"
                )

        # Manual formatting
        parts: list[str] = []

        if system_prompt:
            parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}\n")

        parts.append("Assistant:")

        return "".join(parts)

    def unload(self) -> None:
        """Unload the model from memory.

        Useful for freeing GPU memory when the provider is no longer needed.
        """
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._is_loaded = False

            if _HAS_TRANSFORMERS and _torch is not None:
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

            logger.info(
                "HuggingFaceLocalProvider: model %r unloaded",
                self._model_name,
            )

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"HuggingFaceLocalProvider("
            f"model={self._model_name!r}, "
            f"device={self._device!r}, "
            f"status={status})"
        )

    def __del__(self) -> None:
        """Best-effort cleanup."""
        try:
            self.unload()
        except Exception:
            pass
