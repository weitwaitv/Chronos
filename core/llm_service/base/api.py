"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""
import logging
from abc import ABC, abstractmethod
from typing import Protocol, Union, runtime_checkable

from core.llm_service.openai.prompt.base import (
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)

logger = logging.getLogger(__name__)


class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        raise NotImplementedError()


@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
            the appropriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """


class DummyCompletionResult(CompletionResult):
    def get_completions(self) -> list[str]:
        return ["This is a dummy response."]


class DummyCompletionFn(CompletionFn):
    def __call__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> CompletionResult:
        return DummyCompletionResult()
