import time
from typing import Any, Union

from volcengine.maas import MaasService, MaasException

from core.llm_service.base.api import CompletionResult
from core.record import record_sampling


class FangZhouCompletionResult(CompletionResult):
    def __init__(self, response):
        self.response = response

    def get_completions(self) -> list[str]:
        try:
            txt = self.response.choice.message.content
        except AttributeError:
            txt = ""
        return [txt]


class FangZhouCompletionFn:
    def __init__(
        self, model, host, region, access_key, secret_access_key, extra_options=None
    ):
        if extra_options is None:
            extra_options = {}
        self.host = host
        self.region: str = region
        self.access_key = access_key
        self.secret_access_key = secret_access_key
        self.model = model
        self.extra_options = extra_options

    def __call__(
        self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> FangZhouCompletionResult:
        maas = MaasService(self.host, self.region)
        maas.set_ak(self.access_key)
        maas.set_sk(self.secret_access_key)
        req = {
            "model": {
                "name": self.model,
            },
            "messages": prompt,
        }
        req.update(self.extra_options)
        try:
            start_time = time.time()
            response = maas.chat(req)
            end_time = time.time()
        except MaasException as e:
            print(e)
            response = {}
            start_time = 0
            end_time = 0
        response_time = end_time - start_time
        result = FangZhouCompletionResult(response)
        record_sampling(
            prompt=prompt, sampled=result.get_completions(), response_time=response_time
        )
        return result
