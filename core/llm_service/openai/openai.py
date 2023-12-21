import time
from typing import Any, Optional, Union

from core.evals.module.base import CompletionFnSpec
from core.llm_service.base.api import CompletionResult
from core.llm_service.base.base import openai_chat_completion_create_retrying
from core.llm_service.openai.prompt.base import (
    ChatCompletionPrompt,
    OpenAICreateChatPrompt,
    Prompt,
)
from core.record import record_sampling


class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "message" in choice:
                    try:
                        completions.append(choice["message"]["content"])
                    except KeyError:
                        completions.append("")
        return completions


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
        return completions


class OpenAIChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        is_fn: Optional[bool] = True,
        n_ctx: Optional[int] = None,
        extra_options=None,
        **kwargs,
    ):
        if extra_options is None:
            extra_options = {}
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.is_fn = is_fn
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (
                    isinstance(prompt, list)
                    and all(isinstance(token, int) for token in prompt)
                )
                or (
                    isinstance(prompt, list)
                    and all(isinstance(token, str) for token in prompt)
                )
                or (
                    isinstance(prompt, list)
                    and all(isinstance(msg, dict) for msg in prompt)
                )
            ), (
                f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list["
                f"str] or list[dict[str, str]]"
            )

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt = prompt.to_formatted_prompt()
        start_time = time.time()
        result = openai_chat_completion_create_retrying(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        end_time = time.time()
        result = OpenAIChatCompletionResult(
            raw_data=result, prompt=openai_create_prompt
        )

        if self.is_fn:
            record_sampling(
                prompt=result.prompt,
                sampled=result.get_completions(),
                response_time=end_time - start_time,
            )
        return result
