from typing import Dict, Union, Any

from core.llm_service.base.api import CompletionResult
from core.llm_service.base.base import request
from core.record import record_sampling


class FnForExampleCompletionResult(CompletionResult):
    def __init__(self, response: Union[Dict, None]):
        self.response = response

    def get_completions(self) -> list[str]:
        if self.response:
            return [self.response["data"]["text"]]
        else:
            return [""]


class FnForExampleCompletionFn:
    def __init__(self, api_key, url, extra_options=None):
        if extra_options is None:
            extra_options = {}
        self.api_key = api_key
        self.url = url
        self.extra_options = extra_options

    def __call__(
        self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> FnForExampleCompletionResult:
        response = request(
            method="get",
            url=self.url,
            params={"api_key": self.api_key, "prompt": prompt},
        )
        result = FnForExampleCompletionResult(response)
        record_sampling(prompt=prompt, sampled=result.get_completions())
        return result
