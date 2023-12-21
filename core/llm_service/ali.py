from typing import Any, Union

from dashscope import Generation
from dashscope.api_entities.dashscope_response import GenerationResponse

from core.llm_service.base.api import CompletionResult
from core.record import record_sampling


class ALiCompletionResult(CompletionResult):
    def __init__(self, response):
        self.response: GenerationResponse = response

    def get_completions(self) -> list[str]:
        if self.response.output["text"]:
            return [self.response.output["text"]]
        else:
            return [""]


class ALiCompletionFn:
    def __init__(self, model, api_key, extra_options=None):
        if extra_options is None:
            extra_options = {}
        self.model = model
        self.api_key: str = api_key
        self.extra_options = extra_options

    def __call__(
            self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> ALiCompletionResult:
        prompt_content = ";".join([content["content"] for content in prompt])
        response = Generation.call(
            model=self.model,
            prompt=prompt_content,
            api_key=self.api_key,
            **self.extra_options
        )
        result = ALiCompletionResult(response)
        record_sampling(prompt=prompt_content, sampled=result.get_completions())
        return result
