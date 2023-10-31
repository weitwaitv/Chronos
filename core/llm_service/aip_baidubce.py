import json
from typing import Any, Union

from core.llm_service.base.api import CompletionResult
from core.llm_service.base.base import request
from core.record import record_sampling


class AipBaidubceCompletionResult(CompletionResult):
    def __init__(self, response: dict):
        self.response = response

    def get_completions(self) -> list[str]:
        res = []
        if "result" in self.response:
            res.append(self.response["result"])
        return res


class AipBaidubceCompletionFn:
    def __init__(self, url, client_id, client_secret, extra_options=None):
        if extra_options is None:
            extra_options = {}
        self.url = url
        self.client_id = client_id
        self.client_secret = client_secret
        self.extra_options = extra_options

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"

        payload = json.dumps("")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        response = request(
            "POST",
            url,
            headers=headers,
            data=payload,
            params={"client_id": self.client_id, "client_secret": self.client_secret},
        )
        return response.get("access_token")

    def __call__(
        self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> AipBaidubceCompletionResult:
        prompt_content = ";".join([content["content"] for content in prompt])

        data = json.dumps(
            {"messages": [{"role": "user", "content": prompt_content}]}.update(
                self.extra_options
            )
        )

        headers = {"Content-Type": "application/json"}
        response = request(
            "POST",
            self.url,
            headers=headers,
            params={"access_token": self.get_access_token()},
            data=data,
        )
        print(response)
        result = AipBaidubceCompletionResult(response)
        record_sampling(prompt=prompt_content, sampled=result.get_completions())

        return result
